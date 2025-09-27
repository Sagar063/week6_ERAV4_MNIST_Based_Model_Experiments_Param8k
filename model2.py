# model2.py
# Fully-convolutional LIGHT variants (no FC).
# Variants:
#   - Model2_Light_BN          : BN after every conv, no Dropout, no GAP (7x7 collapse conv)
#   - Model2_Light_DO          : Dropout2d after every conv, no BN, no GAP
#   - Model2_Light_BN_DO       : BN + Dropout2d after every conv, no GAP
#   - Model2_Light_BN_DO_GAP   : BN + Dropout2d after every conv, WITH GAP (1x1 -> AdaptiveAvgPool2d(1))
#
# All models return raw logits of shape (B, num_classes); use nn.CrossEntropyLoss.

import torch
import torch.nn as nn


# --------------------- helpers ---------------------

def _act():
    return nn.ReLU(inplace=True)

def _maybe_bn(out_ch: int, use_bn: bool):
    return nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

def _maybe_do(p: float, use_do: bool):
    return nn.Dropout2d(p) if (use_do and p > 0) else nn.Identity()

class _ConvBNActDO(nn.Module):
    """
    Conv -> (BN) -> ReLU -> (Dropout2d)
    - bias=False when BN is used; True otherwise.
    """
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


class _LightSkeletonBody_NoGAP(nn.Module):
    """
    Skeleton (no GAP): 28->26->24->22 -> pool -> 11 -> 9 -> 7 -> (1x1 proj) -> (7x7 collapse) -> 1
    - For DO variants, Dropout2d is after EVERY conv (including 1x1 proj), but NOT after the final 7x7 logits conv.
    """
    def __init__(self, num_classes: int, use_bn: bool, use_do: bool, p_drop: float,
                 c1: int = 10, c2: int = 10, c3: int = 20, c_bottleneck: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Input + Block 1 (valid 3x3)
        self.conv1 = _ConvBNActDO(1,  c1, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 28->26
        self.conv2 = _ConvBNActDO(c1, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 26->24
        self.conv3 = _ConvBNActDO(c2, c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 24->22

        # Transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 22->11
        self.conv4 = _ConvBNActDO(c3, c_bottleneck, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 11->11

        # Block 2
        self.conv5 = _ConvBNActDO(c_bottleneck, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 11->9
        self.conv6 = _ConvBNActDO(c2,          c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 9->7

        # Output: 1x1 projection to num_classes (+ReLU [+Dropout2d if DO]), then 7x7 collapse conv (logits)
        self.proj_logits = _ConvBNActDO(c3, num_classes, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 7->7
        self.collapse    = nn.Conv2d(num_classes, num_classes, kernel_size=7, padding=0, bias=False)  # 7->1 (no BN/act/dropout)

    def forward(self, x):
        x = self.conv1(x)     # (B,c1,26,26)
        x = self.conv2(x)     # (B,c2,24,24)
        x = self.conv3(x)     # (B,c3,22,22)
        x = self.pool1(x)     # (B,c3,11,11)
        x = self.conv4(x)     # (B,c_bottleneck,11,11)
        x = self.conv5(x)     # (B,c2,9,9)
        x = self.conv6(x)     # (B,c3,7,7)
        x = self.proj_logits(x)  # (B,num_classes,7,7)
        x = self.collapse(x)     # (B,num_classes,1,1)
        return x.squeeze(-1).squeeze(-1)  # (B,num_classes)


class _LightSkeletonBody_WithGAP(nn.Module):
    """
    Skeleton with GAP: 28->26->24->22 -> pool -> 11 -> 9 -> 7 -> (1x1 to num_classes) -> GAP -> 1
    - DO variants: Dropout2d after EVERY conv (including the 1x1 to num_classes).
    """
    def __init__(self, num_classes: int, use_bn: bool, use_do: bool, p_drop: float,
                 c1: int = 10, c2: int = 10, c3: int = 20, c_bottleneck: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Input + Block 1
        self.conv1 = _ConvBNActDO(1,  c1, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 28->26
        self.conv2 = _ConvBNActDO(c1, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 26->24
        self.conv3 = _ConvBNActDO(c2, c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 24->22

        # Transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 22->11
        self.conv4 = _ConvBNActDO(c3, c_bottleneck, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 11->11

        # Block 2
        self.conv5 = _ConvBNActDO(c_bottleneck, c2, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 11->9
        self.conv6 = _ConvBNActDO(c2,          c3, k=3, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 9->7

        # Output: 1x1 to num_classes (+ReLU [+Dropout2d if DO]) then AdaptiveAvgPool2d(1)
        #self.proj_logits = _ConvBNActDO(c3, num_classes, k=1, pad=0, use_bn=use_bn, use_do=use_do, p_drop=p_drop)  # 7->7
        self.proj_logits = nn.Conv2d(c3, num_classes, kernel_size=1, padding=0, bias=True)
        self.gap         = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)     # (B,c1,26,26)
        x = self.conv2(x)     # (B,c2,24,24)
        x = self.conv3(x)     # (B,c3,22,22)
        x = self.pool1(x)     # (B,c3,11,11)
        x = self.conv4(x)     # (B,c_bottleneck,11,11)
        x = self.conv5(x)     # (B,c2,9,9)
        x = self.conv6(x)     # (B,c3,7,7)
        x = self.proj_logits(x)     # (B,num_classes,7,7)
        x = self.gap(x)             # (B,num_classes,1,1)
        return x.squeeze(-1).squeeze(-1)  # (B,num_classes)


# --------------------- variants ---------------------

class Model2_Light_BN(nn.Module):
    """BN after every conv, NO Dropout. No GAP (uses 7x7 collapse conv)."""
    def __init__(self, num_classes: int = 10,
                c1: int = 10, c2: int = 10, c3: int = 20, c_bottleneck: int = 10):
        super().__init__()
        self.body = _LightSkeletonBody_NoGAP(num_classes=num_classes, use_bn=True, use_do=False, p_drop=0.0,
                                             c1=c1, c2=c2, c3=c3, c_bottleneck=c_bottleneck)
    def forward(self, x):
        return self.body(x)


class Model2_Light_DO(nn.Module):
    """Dropout2d AFTER EVERY CONV (p_drop default 0.05), NO BN. No GAP."""
    def __init__(self, num_classes: int = 10, p_drop: float = 0.05,
                 c1: int = 10, c2: int = 10, c3: int = 20, c_bottleneck: int = 10):
        super().__init__()
        self.body = _LightSkeletonBody_NoGAP(num_classes=num_classes, use_bn=False, use_do=True, p_drop=p_drop,
                                             c1=c1, c2=c2, c3=c3, c_bottleneck=c_bottleneck)
    def forward(self, x):
        return self.body(x)


class Model2_Light_BN_DO(nn.Module):
    """BN + Dropout2d AFTER EVERY CONV. No GAP."""
    def __init__(self, num_classes: int = 10, p_drop: float = 0.05,
                 c1: int = 10, c2: int = 10, c3: int = 20, c_bottleneck: int = 10):
        super().__init__()
        self.body = _LightSkeletonBody_NoGAP(num_classes=num_classes, use_bn=True, use_do=True, p_drop=p_drop,
                                             c1=c1, c2=c2, c3=c3, c_bottleneck=c_bottleneck)
    def forward(self, x):
        return self.body(x)


class Model2_Light_BN_DO_GAP(nn.Module):
    """BN + Dropout2d AFTER EVERY CONV, WITH GAP (1x1 -> AdaptiveAvgPool2d(1))."""
    def __init__(self, num_classes: int = 10, p_drop: float = 0.05,
                 c1: int = 10, c2: int = 10, c3: int = 20, c_bottleneck: int = 10):
        super().__init__()
        self.body = _LightSkeletonBody_WithGAP(num_classes=num_classes, use_bn=True, use_do=True, p_drop=p_drop,
                                               c1=c1, c2=c2, c3=c3, c_bottleneck=c_bottleneck)
    def forward(self, x):
        return self.body(x)


__all__ = [
    "Model2_Light_BN",
    "Model2_Light_DO",
    "Model2_Light_BN_DO",
    "Model2_Light_BN_DO_GAP",
]
