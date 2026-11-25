import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import DoubleConv


class LiverSegCNN(nn.Module):
    """Plain encoder-decoder CNN without skip connections (baseline to compare with U-Net)."""

    def __init__(self, in_channels=1, num_classes=3, base_channels=32, dropout=0.1):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3, dropout=dropout)

        self.bottleneck = DoubleConv(c3, c4, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]

        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.bottleneck(x)

        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))

        out = self.head(x)
        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
