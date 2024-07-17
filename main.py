import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
import torchvision.transforms as transforms
from PIL import Image

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1)
    return epe

def multi_scale_loss(pred_flows: list, gt_flow: torch.Tensor, scales: list):
    total_loss = 0
    for i, pred_flow in enumerate(pred_flows):
        scale = scales[i]
        scaled_gt_flow = torch.nn.functional.interpolate(gt_flow, size=pred_flow.shape[2:], mode='bilinear', align_corners=False)
        epe = compute_epe_error(pred_flow, scaled_gt_flow)
        total_loss += epe.mean()
    return total_loss / len(pred_flows)

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

def get_image_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop((480, 640), scale=(0.8, 1.0)),
    ])

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform:
            data['event_volume'] = self.transform(data['event_volume'])
        return data

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch

def get_random_subset(dataset, subset_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:subset_size])

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4,
        sequence_length=2
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()

    image_transforms = get_image_transforms()
    train_set = CustomDataset(train_set, transform=image_transforms)
    test_set = CustomDataset(test_set, transform=image_transforms)

    subset_size = int(len(train_set) * 0.15)

    collate_fn = train_collate
    test_data = DataLoader(test_set,
                           batch_size=args.data_loader.test.batch_size,
                           shuffle=args.data_loader.test.shuffle,
                           collate_fn=collate_fn,
                           drop_last=False)

    model = EVFlowNet(args.train).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001802580559563173, weight_decay=0.00012418527210641904)  # 修正: 最適なハイパーパラメータを使用

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=16)

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, 'model_114_checkpoint.pth')

    start_epoch = 0
    if os.path.exists(checkpoint_file):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, filename=checkpoint_file)
        print(f"Resuming training from epoch {start_epoch}")     #####
        # 学習率を変更
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000001
        print(f"Changed learning rate to {param_group['lr']}")  
    best_val_loss = float('inf')

    scales = [0.5, 0.25, 0.125, 0.0625]

    model.train()
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))

        # 学習率を表示
        for param_group in optimizer.param_groups:
            print(f"Learning rate: {param_group['lr']}")
            print(f"weight_decay: {param_group['weight_decay']}")

        random.seed()  # シードをリセットしてランダムにする
        train_subset = get_random_subset(train_set, subset_size)
        train_data = DataLoader(train_subset,
                                batch_size=args.data_loader.train.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)

        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            ground_truth_flow = batch["flow_gt"].to(device)

            B, S, C, H, W = event_image.shape
            event_image = event_image.view(B * S, C, H, W)

            flows = model(event_image)
            assert len(flows) == len(scales), f"Expected {len(scales)} scales, but got {len(flows)} outputs"
            flows = [flow.view(B, *flow.shape[1:]) for flow in flows]

            loss: torch.Tensor = multi_scale_loss(flows, ground_truth_flow, scales)
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_file)

        scheduler.step(avg_loss)  # 修正: 平均損失をスケジューラに渡す

        # 学習率のログ出力
        for param_group in optimizer.param_groups:
            print(f"Scheduler adjusted learning rate to: {param_group['lr']}")

    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)

            B, S, C, H, W = event_image.shape
            event_image = event_image.view(B * S, C, H, W)

            batch_flow = model(event_image)[0]
            batch_flow = torch.nn.functional.interpolate(batch_flow, size=(480, 640), mode='bilinear', align_corners=False)
            batch_flow = batch_flow.view(B, S, *batch_flow.shape[1:]).mean(dim=1)

            flow = torch.cat((flow, batch_flow), dim=0)
        print("test done")

    file_name = "submission_114-"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
