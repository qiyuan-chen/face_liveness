#!usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2021/10/13 10:54
# @Author: Chen Jizhi
# @E-mail: chenjizhi@ruijie.com
# @File: cdcn.py
# @Software: PyCharm


import math
import torch
import torch.nn.functional as F
from torch import nn


## Centeral-difference (second order, with 9 parameters and a const theta for 3x3 kernel) 2D Convolution
## | a1 a2 a3 |   | w1 w2 w3 |
## | a4 a5 a6 | * | w4 w5 w6 | --> output = \sum_{i=1}^{9}(ai * wi) - \sum_{i=1}^{9}wi * a5 --> Conv2d (k=3) - Conv2d (k=1)
## | a7 a8 a9 |   | w7 w8 w9 |
##   --> output =
## | a1 a2 a3 |   |  w1  w2  w3 |
## | a4 a5 a6 | * |  w4  w5  w6 |  -  | a | * | w\_sum |     (kernel_size=1x1, padding=0)
## | a7 a8 a9 |   |  w7  w8  w9 |


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CDCN(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN, self).__init__()
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.downsample32x32 = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)

    def forward(self, x):  # x [3, 224, 224]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)  # x [128, 112, 112]
        x_Block1_32x32 = self.downsample32x32(x_Block1)  # x [128, 28, 28]

        x_Block2 = self.Block2(x_Block1)  # x [128, 56, 56]
        x_Block2_32x32 = self.downsample32x32(x_Block2)  # x [128, 28, 28]

        x_Block3 = self.Block3(x_Block2)  # x [128, 28, 28]
        x_Block3_32x32 = self.downsample32x32(x_Block3)  # x [128, 28, 28]

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)  # x [128*3, 28, 28]

        x = self.lastconv1(x_concat)  # x [128, 28, 28]
        x = self.lastconv2(x)  # x [64, 28, 28]
        x = self.lastconv3(x)  # x [1, 28, 28]

        map_x = x.squeeze(1)

        return map_x, x_concat


class CDCNpp(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCNpp, self).__init__()
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            basic_conv(128, int(128 * 1.6), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128 * 1.6)),
            nn.ReLU(),
            basic_conv(int(128 * 1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128 * 1.2)),
            nn.ReLU(),
            basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128 * 1.4), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128 * 1.4)),
            nn.ReLU(),
            basic_conv(int(128 * 1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128 * 1.2)),
            nn.ReLU(),
            basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Original

        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(size=(28, 28), mode='bilinear')

    def forward(self, x):  # x [3, 224, 224]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)
        x_Block3_SA = attention3 * x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)

        map_x = self.lastconv1(x_concat)
        map_x = map_x.squeeze(1)
        return map_x, x_concat


class CDCN_Head(nn.Module):
    def __init__(self, num_classes, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN_Head, self).__init__()
        self.Block1 = nn.Sequential(
            basic_conv(384, 384, kernel_size=1, stride=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            basic_conv(384, 384, kernel_size=3, stride=1, padding=1, bias=False, theta=theta, groups=384),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            basic_conv(384, 768, kernel_size=1, stride=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.Block2 = nn.Sequential(
            basic_conv(768, 768, kernel_size=1, stride=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            basic_conv(768, 768, kernel_size=3, stride=1, padding=1, bias=False, theta=theta, groups=768),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            basic_conv(768, 1280, kernel_size=1, stride=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=1280, out_features=640, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=640, out_features=num_classes, bias=True)

    def forward(self, x):
        feature_list = [x]
        x = self.Block1(x)
        feature_list.append(x)
        x = self.Block2(x)
        feature_list.append(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x, feature_list


class CDCN_v1(nn.Module):
    def __init__(self, num_classes):
        super(CDCN_v1, self).__init__()
        self.backbone = CDCN()
        self.head = CDCN_Head(num_classes)

    def forward(self, x):
        _, feature = self.backbone(x)
        x, feature_list = self.head(feature)
        return x, feature_list


class CDCN_v2(nn.Module):
    def __init__(self, num_classes):
        super(CDCN_v2, self).__init__()
        self.backbone = CDCNpp()
        self.head = CDCN_Head(num_classes)

    def forward(self, x):
        _, feature = self.backbone(x)
        x, feature_list = self.head(feature)
        return x, feature_list


if __name__ == "__main__":
    inputs = torch.rand((2, 3, 224, 224))
    model = CDCN_v1(2)
    cls, outputs = model(inputs)
    print(cls.size())
    for o in outputs:
        print(o.size())
