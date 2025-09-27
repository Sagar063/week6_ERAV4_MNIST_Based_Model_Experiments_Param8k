
# model3.py
# Capacity increase, corrected pooling location, strong tiny model aiming for <=8k params + 99.4% target in <=15 epochs (dependent on training setup).
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- tiny helpers (same behavior as your _WithGAP body) ---
def _act():
    return nn.ReLU(inplace=True)

def _maybe_bn(out_ch: int, use_bn: bool):
    return nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

def _maybe_do(p: float, use_do: bool):
    return nn.Dropout2d(p) if (use_do and p > 0) else nn.Identity()

class _ConvBNActDO(nn.Module):
    """Conv -> (BN) -> ReLU -> (Dropout2d). bias=False when BN is used."""
    def __init__(self, in_ch: int, out_ch: int, k: int, pad: int,
                 use_bn: bool, use_do: bool, p_drop: float):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad,
                              bias=not use_bn)
        self.bn   = _maybe_bn(out_ch, use_bn)
        self.act  = _act()
        self.do   = _maybe_do(p_drop, use_do)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x

# Code 7 - Increase capacity moderately (still small compared to big baseline)
class Model3_CapacityUp(nn.Module):
    """
    _WithGAP + extra capacity like Net:
      28->26->24->22 -> pool -> 11 -> 9 -> 7
      -> [EXTRA 3x3 valid: c3->c4] -> 5
      -> 1x1 to num_classes -> AdaptiveAvgPool2d(1) -> logits.
    BN/Dropout2d placement matches your _WithGAP style (after every conv).
    """
    def __init__(
        self,
        num_classes: int = 10,
        use_bn: bool = True,
        use_do: bool = True,
        p_drop: float = 0.05,
        c1: int = 10, c2: int = 10, c3: int = 20, c4: int = 32, c_bottleneck: int = 10
    ):
        super().__init__()
        self.num_classes = num_classes

        # Input + Block 1 (valid 3x3s)
        self.conv1 = _ConvBNActDO(1,  c1, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 28->26
        self.conv2 = _ConvBNActDO(c1, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 26->24
        self.conv3 = _ConvBNActDO(c2, c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 24->22

        # Transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 22->11
        self.conv4 = _ConvBNActDO(c3, c_bottleneck, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 11->11

        # Block 2 (valid 3x3s)
        self.conv5 = _ConvBNActDO(c_bottleneck, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 11->9
        self.conv6 = _ConvBNActDO(c2,          c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 9->7

        # >>> EXTRA capacity like Net: add a 3x3 valid conv at 7x7 to reach 5x5
        self.extra3x3 = _ConvBNActDO(c3, c4, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 7->5

        # Project to class maps at 5x5 (keep BN+ReLU [+Dropout] like _WithGAP), then GAP to 1x1
        #self.proj_logits = _ConvBNActDO(c4, num_classes, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 5->5
        self.proj_logits = nn.Conv2d(c4, num_classes, kernel_size=1, padding=0, bias=True)  # 5->5
        self.gap         = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)       # (B,c3,7,7)
        x = self.extra3x3(x)    # (B,c4,5,5)
        x = self.proj_logits(x) # (B,num_classes,5,5)
        x = self.gap(x)         # (B,num_classes,1,1)
        return x.view(x.size(0), self.num_classes)  # raw logits; use CrossEntropyLoss

class Model3_PoolFixed(nn.Module):
    """
    Fixed 5x5 pool head (no FC, no adaptive pooling).
    Shape: 28->26->24->22 -> pool -> 11 -> 9 -> 7 -> (extra 3x3) -> 5 -> 1x1->K -> AvgPool2d(5) -> 1.
    ~7.6–7.7k params with the default channels below (including BN).
    """
    def __init__(self, num_classes: int = 10,
                 use_bn: bool = True, use_do: bool = True, p_drop: float = 0.05,
                 c1: int = 8, c2: int = 10, c3: int = 18, c_bottleneck: int = 12, c4: int = 14):
                 #c1: int = 6, c2: int = 8, c3: int = 20, c_bottleneck: int = 12, c4: int = 17): #6,8,20,12,17
        super().__init__()
        self.num_classes = num_classes

        # Block 1 (valid 3x3s): 28->26->24->22
        self.conv1 = _ConvBNActDO(1,  c1, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)
        self.conv2 = _ConvBNActDO(c1, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)
        self.conv3 = _ConvBNActDO(c2, c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)

        # Transition: 22->11, then 1x1 bottleneck at 11x11
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = _ConvBNActDO(c3, c_bottleneck, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)

        # Block 2 (valid 3x3s): 11->9->7
        self.conv5 = _ConvBNActDO(c_bottleneck, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)
        self.conv6 = _ConvBNActDO(c2,          c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)

        # Extra capacity at 7x7 → 5x5
        self.extra3x3 = _ConvBNActDO(c3, c4, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 7->5

        # Project to class maps at 5x5, then FIXED pool to 1x1
        #self.proj_logits = _ConvBNActDO(c4, num_classes, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)
        self.proj_logits = nn.Conv2d(c4, num_classes, kernel_size=1, padding=0, bias=True)  # 5->5
        self.fixed_pool  = nn.AvgPool2d(kernel_size=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)        # (B,c1,26,26)
        x = self.conv2(x)        # (B,c2,24,24)
        x = self.conv3(x)        # (B,c3,22,22)
        x = self.pool1(x)        # (B,c3,11,11)
        x = self.conv4(x)        # (B,c_bottleneck,11,11)
        x = self.conv5(x)        # (B,c2,9,9)
        x = self.conv6(x)        # (B,c3,7,7)
        x = self.extra3x3(x)     # (B,c4,5,5)
        x = self.proj_logits(x)  # (B,num_classes,5,5)
        x = self.fixed_pool(x)   # (B,num_classes,1,1)
        return x.squeeze(-1).squeeze(-1)  # (B,num_classes)




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
    Widened TinyTarget (~7.33k params):
      Stem: 1 -> 20 -> 24 (3x3, pad=1)
      Body: MaxPool -> DWConv(24->32) -> DWConv(32->32) -> MaxPool
            -> 1x1 to logits -> AdaptiveAvgPool2d(1)
    Returns raw logits (B, num_classes).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 20, 3, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.MaxPool2d(2),
            DWConv(24, 32),
            DWConv(32, 32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, num_classes, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)   # [B, num_classes, 1, 1]
        return x.view(x.size(0), -1)
