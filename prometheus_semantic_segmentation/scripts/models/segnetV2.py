import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv_in = ConvBlock(3, 32)
        self.down1 = DownSample(32, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.bottleneck = DownSample(256, 256)
        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)
        self.up3 = UpSample(64, 32)
        self.up4 = UpSample(32, 32)
        self.conv_out = OutConv(32, num_classes)


    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        mask = self.conv_out(x)

        return mask
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.act(self.conv(x)))
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = ConvBlock(in_channels*2, out_channels)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x 
    
