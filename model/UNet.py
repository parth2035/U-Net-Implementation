import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(UNet, self).__init__()

        #self.feature_map_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.contract_1 = Contracting_Block(in_channels, hidden_channels)
        self.contract_2 = Contracting_Block(hidden_channels, hidden_channels * 2)
        self.contract_3 = Contracting_Block(hidden_channels * 2, hidden_channels * 4)
        self.contract_4 = Contracting_Block(hidden_channels * 4, hidden_channels * 8)

        self.mid1 = nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=3)
        self.mid2 = nn.Conv2d(hidden_channels * 16, hidden_channels * 16, kernel_size=3)

        self.expanding_1 = Expanding_Block(hidden_channels * 16)
        self.expanding_2 = Expanding_Block(hidden_channels * 8)
        self.expanding_3 = Expanding_Block(hidden_channels * 4)
        self.expanding_4 = Expanding_Block(hidden_channels * 2)

        self.feature_map_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.contract_1(x)
        x2 = self.contract_2(x1)
        x3 = self.contract_3(x2)
        x4 = self.contract_4(x3)

        x4 = self.mid2(self.mid1(x4))
        print(x4.size())
        x5 = self.expanding_1(x4, x3)
        print(x5.size())
        x6 = self.expanding_2(x5, x2)
        print(x6.size())
        x7 = self.expanding_3(x6, x1)
        x8 = self.expanding_4(x7, x)
        x = self.feature_map_out(x8)

        return x


class Contracting_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Contracting_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        return x


class Expanding_Block(nn.Module):
    def __init__(self, in_channels):
        super(Expanding_Block, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x, cc):
        x = self.conv3(self.upsample(x))
        print(x.size())
        cropped_image = self.crop(x, cc)

        x = torch.cat((cropped_image, x), dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x

    def crop(self, x, cc):
        mid_height = cc.size(2) // 2

        min_height = mid_height - x.size(2) // 2
        max_height = min_height + x.size(2)

        cropped_image = cc[:, :, min_height:max_height, min_height:max_height]

        return cropped_image
