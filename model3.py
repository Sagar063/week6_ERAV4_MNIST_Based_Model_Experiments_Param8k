
# model3.py
# Capacity increase, corrected pooling location, strong tiny model aiming for <=8k params + 99.4% target in <=15 epochs (dependent on training setup).
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code 7 - Increase capacity moderately (still small compared to big baseline)
class Model3_CapacityUp(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # 14x14

            nn.Conv2d(16, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # 7x7

            nn.Conv2d(24, 10, 1, bias=False),  # to logits channels
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
        x = self.net(x)          # [B,10,1,1]
        return x.view(x.size(0), -1)

# Code 8 - Correct MaxPooling Location (pool after sufficient convs to capture patterns)
class Model3_PoolFixed(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.MaxPool2d(2),               # after two convs
            nn.Conv2d(12, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # down to 7x7
            nn.Conv2d(16, num_classes, 1, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        return x.view(x.size(0), -1)

# Code 9+10 - Tiny model targeting <= 8k params + strong training setup (aug + StepLR)
# Depthwise-separable style to reduce parameters, GAP classifier.
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class Model3_TinyTarget(nn.Module):
    """
    Very small CNN aiming < 8k params:
    Channels: 1->12->16->16->10 (via 1x1 to logits) + GAP
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.MaxPool2d(2),
            DWConv(12, 16),
            DWConv(16, 16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, num_classes, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)   # [B,10,1,1]
        return x.view(x.size(0), -1)
