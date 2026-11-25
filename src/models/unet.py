import torch
import torch.nn as nn

from src.models.blocks import DoubleConv, ResidualConvBlock, match_size


class UpStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, block_cls, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = block_cls(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = match_size(self.up(x), skip)
        return self.conv(torch.cat([skip, x], dim=1))


class _UNetBackbone(nn.Module):
    """4-level U-Net shared by standard UNet and residual CNNUNet."""

    def __init__(self, block_cls, in_channels=1, num_classes=3, base_channels=32, bilinear=True, dropout=0.1):
        super().__init__()
        c1, c2, c3, c4, c5 = [base_channels * m for m in (1, 2, 4, 8, 16)]

        self.pool = nn.MaxPool2d(2)
        self.enc1 = block_cls(in_channels, c1)
        self.enc2 = block_cls(c1, c2)
        self.enc3 = block_cls(c2, c3, dropout=dropout * 0.5)
        self.enc4 = block_cls(c3, c4, dropout=dropout)
        self.bottleneck = block_cls(c4, c5, dropout=dropout)

        self.up4 = UpStage(c5, c4, c4, block_cls, bilinear=bilinear, dropout=dropout * 0.5)
        self.up3 = UpStage(c4, c3, c3, block_cls, bilinear=bilinear)
        self.up2 = UpStage(c3, c2, c2, block_cls, bilinear=bilinear)
        self.up1 = UpStage(c2, c1, c1, block_cls, bilinear=bilinear)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))

        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)


class UNet(_UNetBackbone):
    """Classic 2D U-Net with DoubleConv blocks and skip connections."""

    def __init__(self, in_channels=1, num_classes=3, base_channels=32, bilinear=True, dropout=0.1):
        super().__init__(DoubleConv, in_channels, num_classes, base_channels, bilinear, dropout)


class CNNUNet(_UNetBackbone):
    """Residual CNN encoder/decoder inside a U-Net skip-connection topology."""

    def __init__(self, in_channels=1, num_classes=3, base_channels=32, bilinear=True, dropout=0.1):
        super().__init__(ResidualConvBlock, in_channels, num_classes, base_channels, bilinear, dropout)
