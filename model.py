import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """连续两次卷积的基础模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class PI_UNet_Transient(nn.Module):
    """23通道满血版 PI-UNet 架构"""
    def __init__(self, in_channels=23, out_channels=10, features=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features, features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 2, features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 4, features * 8))
        self.up1 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features * 8, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)
        self.up3 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 2, features)
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.dec1(torch.cat([x3, self.up1(x4)], dim=1))
        x = self.dec2(torch.cat([x2, self.up2(x)], dim=1))
        x = self.dec3(torch.cat([x1, self.up3(x)], dim=1))
        return torch.relu(self.outc(x))
