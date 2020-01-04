# coding=utf-8
"""
@File   : model.py
@Time   : 2020/01/04
@Author : Zengrui Zhao
"""
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.heads import SegmentationHead
import torch.nn as nn
from torchsummary import summary
import torch

class model(nn.Module):
    def __init__(self):
        super().__init__()
        net = smp.Unet()
        self.encoder = net.encoder
        self.decoder = net.decoder
        self.segmentation_head = net.segmentation_head
        self.horizontalVertical = SegmentationHead(in_channels=16,
                                                   out_channels=2,
                                                   kernel_size=3)

    def _regression(self):
        self.regression = self.decoder
        print(self.regression)

    def forward(self, x):
        features = self.encoder(x)
        decoder = self.decoder(*features)
        segmentation = self.segmentation_head(decoder)
        horizontalVertical = self.horizontalVertical(decoder)
        return [segmentation, horizontalVertical]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model().to(device)
    print(net)
    summary(net, (3, 256, 256))