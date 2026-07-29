import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.layers.conv_blocks import conv_block, up_conv
from models.layers.attention import SEBlock
from models.layers.pyramid import CustomPyramidModule


class UNet(nn.Module):
    """UNet with ResNet18 encoder + Custom Pyramid Module"""
    def __init__(self, num_classes=11):
        super(UNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (B, 64, H/2, W/2)
        self.pool0 = resnet.maxpool  # (B, 64, H/4, W/4)
        self.encoder1 = resnet.layer1  # (B, 64, H/4, W/4)
        self.encoder2 = resnet.layer2  # (B, 128, H/8, W/8)
        self.encoder3 = resnet.layer3  # (B, 256, H/16, W/16)
        self.encoder4 = resnet.layer4  # (B, 512, H/32, W/32)

        self.pyramid = CustomPyramidModule(512, out_channels=512)

        self.up4 = up_conv(512, 256)
        self.dec4 = conv_block(512, 256)
        self.se4 = SEBlock(256)

        self.up3 = up_conv(256, 128)
        self.dec3 = conv_block(256, 128)
        self.se3 = SEBlock(128)

        self.up2 = up_conv(128, 64)
        self.dec2 = conv_block(128, 64)
        self.se2 = SEBlock(64)

        self.up1 = up_conv(64, 32)
        self.dec1 = conv_block(96, 32)  # 64 from encoder0 + 32 upsampled
        self.se1 = SEBlock(32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool0(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.pyramid(e4)

        d4 = self.up4(center)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d4 = self.se4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d3 = self.se3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d2 = self.se2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)
        d1 = self.se1(d1)

        out = self.final(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out
