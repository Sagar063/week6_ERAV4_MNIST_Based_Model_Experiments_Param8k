
# model2.py
# Variants that add BN, Dropout, BN+Dropout, and BN+Dropout+GAP to the light model.
import torch
import torch.nn as nn
import torch.nn.functional as F

def _light_block_bn_do():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Dropout(0.05),
        nn.Conv2d(8, 12, 3, padding=1, bias=False),
        nn.BatchNorm2d(12),
        nn.ReLU(inplace=True),
        nn.Conv2d(12, 12, 3, padding=1, bias=False),
        nn.BatchNorm2d(12),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class Model2_Light_BN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*7*7, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class Model2_Light_DO(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.05),
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(12*7*7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class Model2_Light_BN_DO(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = _light_block_bn_do()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(12*7*7, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class Model2_Light_BN_DO_GAP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = _light_block_bn_do()
        self.classifier = nn.Sequential(
            nn.Conv2d(12, num_classes, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        x = self.features(x)              # [B,12,7,7]
        x = self.classifier(x)            # [B,10,1,1]
        return x.view(x.size(0), -1)      # logits
