import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.layers.pyramid import PSPModule


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network"""
    def __init__(self, num_classes=11, backbone='resnet50'):
        super(PSPNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.ppm = PSPModule(in_channels=512)
        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ppm(x)
        x = self.final(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x
