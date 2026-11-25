import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Standard 2x(Conv3x3 -> BN -> ReLU) block used in CNN and U-Net."""

    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=0.0):
        super().__init__()
        mid = mid_ch or out_ch
        layers = [
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualConvBlock(nn.Module):
    """Two 3x3 convs with a residual skip connection."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.drop(self.conv1(x)))
        return self.relu(out + self.shortcut(x))


def match_size(x, ref):
    """Pad or resize feature map `x` so its HxW matches `ref` before skip concat."""
    dy = ref.size(2) - x.size(2)
    dx = ref.size(3) - x.size(3)
    if dy == 0 and dx == 0:
        return x
    if dy >= 0 and dx >= 0:
        return F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
    return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
