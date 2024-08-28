import torch.nn.functional as F
import torch
import torch.nn as nn

from einops import rearrange

from involution import involution

# in_channels_1=in_channels_1, in_channels_2=in_channels_2,out_channels=num_classes

FM = 16

class Feature_HSI(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Feature_HSI, self).__init__()
        self.stride = stride
        self.F_1 = nn.Conv2d(in_channels, 16, kernel_size, stride, padding, dilation, groups, bias)
        self.F_2 = nn.Conv2d(16, 32, kernel_size, stride, padding, dilation, groups, bias)
        self.F_3 = nn.Conv2d(32, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.F_3(self.F_2(self.F_1(X_h)))
        return X_h


class Involutin_module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Involutin_module, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.f_inv_12 = involution(in_channels, kernel_size, 1)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h=self.f_inv_12(X_h)
        X_h = self.relu(self.bn_h(self.f_inv_11(X_h)))
        return X_h


class MHFM(nn.Module):
    def __init__(self, in_channels, patch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MHFM, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0, dilation, groups, bias)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels * 2, 3, 1, 0, dilation, groups, bias)

        self.conv2_1 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0, dilation, groups, bias)
        self.conv2_2 = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 0, dilation, groups, bias)
        self.conv2_3 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0, dilation, groups, bias)
        self.conv2_4 = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 0, dilation, groups, bias)

        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias)
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias)
        self.conv3_4 = nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias)

        self.conv4_2 = nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias)
        self.conv4_3 = nn.Conv2d(in_channels, in_channels, 3, 1, 0, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        x1_1 = self.conv1_1(X_h)
        x1_2 = self.conv1_2(X_h)

        x2_1 = self.conv2_1(x1_1)
        x2_2 = self.conv2_2(x1_1)
        x2_3 = self.conv2_3(x1_2)
        x2_4 = self.conv2_4(x1_2)

        x3_1 = self.conv3_1(x2_1)
        x3_2 = x3_1 + x2_2
        x3_3 = self.conv3_3(x2_3)
        x3_4 = self.conv3_4(x2_4)

        x4_2 = self.conv4_2(x3_2)
        x4_3 = x4_2 + x3_3
        x4_3 = self.conv4_3(x4_3)
        x4_4 = x4_3 + x3_4

        x = self.relu(self.bn_h(x4_4))
        return x


class lidar_conv(nn.Module):
    def __init__(self, in_channels, patch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(lidar_conv, self).__init__()

        self.lidar_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=FM, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv2d(FM, FM * 2, 3, 1, 0),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(),
            nn.MaxPool2d(1),
            nn.Dropout(0.4),
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 0),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(1),
            nn.Dropout(0.5),
        )

    def forward(self, X_l):
        x = self.lidar_conv(X_l)
        return x


class fusion_main(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, patch):
        super(fusion_main, self).__init__()
        self.Feature_HSI = Feature_HSI(in_channels_1, 64, kernel_size=1, stride=1, padding=0)
        self.Involutin_module = Involutin_module(in_channels_1, 64, kernel_size=3, stride=1, padding=1)

        self.MHFM = MHFM(64, patch, kernel_size=3, stride=1, padding=0)

        self.lidar_conv = lidar_conv(in_channels=in_channels_2, patch=patch, kernel_size=3, stride=1, padding=0)
        self.linear = nn.Sequential(
            nn.Linear(64 * (patch - 6) * (patch - 6), num_classes * 8),
            nn.BatchNorm1d(num_classes*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

        )
        self.linear3 = nn.Sequential(
            nn.Linear(num_classes*8, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.linear1 = nn.Sequential(
            nn.Linear(64 * (patch - 6) * (patch - 6), num_classes * 8),
            nn.BatchNorm1d(num_classes * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

        )
        self.linear4 = nn.Sequential(
            nn.Linear(num_classes * 8, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(64 * (patch - 6) * (patch - 6), num_classes * 8),
            nn.BatchNorm1d(num_classes * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

        )
        self.linear5 = nn.Sequential(
            nn.Linear(num_classes * 8, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.Weight_Alpha = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.Weight_Beta = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.FusionLayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.FusionLayer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x1, x2):

        feature_hsi = self.Feature_HSI(x1)
        Involutin_opeator = self.Involutin_module(x1)
        feature_spctral = Involutin_opeator * feature_hsi
        MHFM_HSI = self.MHFM(feature_spctral)
        pre_Lidar = self.lidar_conv(x2)

        # 方式2
        weight_alpha = F.softmax(self.Weight_Alpha, dim=0)
        weight_beta = F.softmax(self.Weight_Beta, dim=0)
        out0 = self.FusionLayer(weight_alpha[0] * MHFM_HSI + weight_alpha[1] * pre_Lidar)
        out0 = rearrange(out0, 'n c h w -> n (c h w)')
        out0 = self.linear(out0)
        out0 = self.linear3(out0)

        out1 = self.FusionLayer1(torch.maximum(MHFM_HSI, pre_Lidar))
        out1 = rearrange(out1, 'n c h w -> n (c h w)')
        out1 = self.linear1(out1)
        out1 = self.linear4(out1)

        out2 = self.FusionLayer2(torch.concat([MHFM_HSI, pre_Lidar], dim=1))
        out2 = rearrange(out2, 'n c h w -> n (c h w)')
        out2 = self.linear2(out2)
        out2 = self.linear5(out2)

        out = out0 + out1 + out2
        return out, out0, out1, out2, weight_beta
