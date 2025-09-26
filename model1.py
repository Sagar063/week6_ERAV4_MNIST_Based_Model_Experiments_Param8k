
# model1.py
# Two baselines: Big (~194k params) and Light (~10.7k params)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1_Big(nn.Module):
    """
    Intentionally larger baseline (~194k params) without BN/DO/GAP.
    1x28x28 -> conv stacks -> linear classifier.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 1->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 14x14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Model1_Light(nn.Module):
    """
    Compact baseline (~10.7k params), no BN/DO/GAP.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Keep channels small
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # 1->8
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 14x14

            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*7*7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x
