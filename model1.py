
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
        self.num_classes = num_classes
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
        # ---- Fully-convolutional head (valid 3x3 convs): 7->5->3->1 ----
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # (N,128,5,5)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0), # (N,256,3,3)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=0),  # (N,C,1,1)
            # no activation here; these are logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)           # (N,64,7,7)
        x = self.head(x)           # (N,num_classes,1,1)
        x = x.view(x.size(0), self.num_classes)  # (N,num_classes)
        return x  # raw logits; use CrossEntropyLoss



class Model1_Skeleton(nn.Module):
    """
    Mid-sized fully-convolutional baseline (no FC, no GAP).
    Mirrors professor's transition-block style:
      28x28 -> ... -> 11x11 (pool) -> 7x7 -> 1x1 via 7x7 conv.
    Returns raw logits (N, num_classes).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0, bias=True),  # 28 -> 26
            nn.ReLU(inplace=True),
        )

        # Convolution Block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, bias=True),  # 26 -> 24
            nn.ReLU(inplace=True),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, bias=True), # 24 -> 22
            nn.ReLU(inplace=True),
        )

        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # 22 -> 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, padding=0, bias=True), # 11 -> 11
            nn.ReLU(inplace=True),
        )

        # Convolution Block 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, bias=True),  # 11 -> 9
            nn.ReLU(inplace=True),
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, bias=True), # 9 -> 7
            nn.ReLU(inplace=True),
        )

        # Output Block
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, padding=0, bias=True),  # 7 -> 7
            nn.ReLU(inplace=True),
        )
        self.convblock8 = nn.Conv2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=7, padding=0, bias=True
        )  # 7 -> 1 (no activation; logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convblock1(x)  # (N,32,26,26)
        x = self.convblock2(x)  # (N,64,24,24)
        x = self.convblock3(x)  # (N,128,22,22)
        x = self.pool1(x)       # (N,128,11,11)
        x = self.convblock4(x)  # (N,32,11,11)
        x = self.convblock5(x)  # (N,64,9,9)
        x = self.convblock6(x)  # (N,128,7,7)
        x = self.convblock7(x)  # (N,num_classes,7,7)
        x = self.convblock8(x)  # (N,num_classes,1,1)
        x = x.view(x.size(0), self.num_classes)  # (N,num_classes)
        return x  # raw logits; use nn.CrossEntropyLoss


class Model1_Light(nn.Module):
    """
    Fully-convolutional LIGHT baseline (no FC, no GAP).
    Stem keeps small channels; head uses a single 7x7 valid conv to collapse 7x7 -> 1x1.
    Returns raw logits (N, num_classes).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Stem: produces (N, C, 7, 7). Keep channels compact.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # (N,8,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),   # (N,8,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (N,8,14,14)

            nn.Conv2d(8, 12, kernel_size=3, padding=1),  # (N,12,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), # (N,12,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (N,12,7,7)
        )

        # Head: single 7x7 conv (valid) to get (N, num_classes, 1, 1)
        self.head = nn.Conv2d(12, num_classes, kernel_size=7, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                # (N,12,7,7)
        x = self.head(x)                # (N,num_classes,1,1)
        x = x.view(x.size(0), self.num_classes)  # (N,num_classes)
        return x  # raw logits; pair with nn.CrossEntropyLoss()

