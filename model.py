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
        net = smp.Unet(classes=1,
                       encoder_depth=5,
                       decoder_channels=[1024, 512, 256, 128, 64],
                       encoder_name='se_resnext101_32x4d',
                       activation='sigmoid')
        self.encoder = net.encoder
        self.decoder = net.decoder
        self.segmentation_head = net.segmentation_head
        self.horizontalVertical_head = SegmentationHead(in_channels=64,
                                                        out_channels=2,
                                                        kernel_size=3)

    def forward(self, x):
        features = self.encoder(x)
        segmentation = self.segmentation_head(self.decoder(*features))
        horizontalVertical = self.horizontalVertical_head(self.decoder(*features))

        return [segmentation, horizontalVertical]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model().to(device)
    print(net)
    summary(net, (3, 256, 256))