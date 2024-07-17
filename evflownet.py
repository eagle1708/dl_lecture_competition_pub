import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet, self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels=8, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)  # 修正: in_channels=8
        self.encoder2 = general_conv2d(in_channels=_BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels=2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels=4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.dropout = nn.Dropout(p=0.5)  # ドロップアウトレイヤーを追加

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                                                         out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                                                         out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                                                         out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                                                         out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

        # Adding Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(_BASE_CHANNELS)
        self.batch_norm2 = nn.BatchNorm2d(2*_BASE_CHANNELS)
        self.batch_norm3 = nn.BatchNorm2d(4*_BASE_CHANNELS)
        self.batch_norm4 = nn.BatchNorm2d(8*_BASE_CHANNELS)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        inputs = self.batch_norm1(inputs)  # Batch Normalization
        skip_connections['skip0'] = inputs.clone()
        inputs = self.dropout(inputs)  # ドロップアウトを追加
        inputs = self.encoder2(inputs)
        inputs = self.batch_norm2(inputs)  # Batch Normalization
        skip_connections['skip1'] = inputs.clone()
        inputs = self.dropout(inputs)  # ドロップアウトを追加
        inputs = self.encoder3(inputs)
        inputs = self.batch_norm3(inputs)  # Batch Normalization
        skip_connections['skip2'] = inputs.clone()
        inputs = self.dropout(inputs)  # ドロップアウトを追加
        inputs = self.encoder4(inputs)
        inputs = self.batch_norm4(inputs)  # Batch Normalization
        skip_connections['skip3'] = inputs.clone()
        inputs = self.dropout(inputs)  # ドロップアウトを追加
        
        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        return [flow_dict['flow0'], flow_dict['flow1'], flow_dict['flow2'], flow_dict['flow3']]  # 修正: リストとして返す
