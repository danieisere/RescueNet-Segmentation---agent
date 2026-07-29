import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPyramidModule(nn.Module):
    """自定义金字塔模块，用于UNet的瓶颈层"""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), out_channels=512):
        super(CustomPyramidModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pyramid_feats = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        out = torch.cat([x] + pyramid_feats, dim=1)
        return self.project(out)


class PSPModule(nn.Module):
    """Pyramid Scene Parsing模块，用于PSPNet"""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction_channels=512):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * reduction_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
