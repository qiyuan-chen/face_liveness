#!usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2021/10/19 17:00
# @Author: Chen Jizhi
# @E-mail: chenjizhi@ruijie.com
# @File: binocular.py
# @Software: PyCharm

import torch
import torch.nn as nn


def conv_left(inp, expansion=8, stride=2, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp*expansion, 3, stride, 1, groups=1, bias=False),
        nn.BatchNorm2d(inp*expansion),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_right(inp, expansion=8, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp*expansion, 1, stride, 0, groups=inp, bias=False),
        nn.BatchNorm2d(inp*expansion),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class BinocularNet(nn.Module):
    def __init__(self, num_classes):
        super(BinocularNet, self).__init__()
        self.conv_l_1 = conv_left(3, 2)  # 3 -> 12
        self.conv_r_1 = conv_right(3, 2)  # 3 -> 12
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 12

        self.conv_l_2 = conv_left(12, 2)  # 24 -> 24
        self.conv_r_2 = conv_right(12, 2)  # 24 -> 24
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 48

        self.conv_l_3 = conv_left(48, 2)  # 48 -> 96
        self.conv_r_3 = conv_right(48, 2)  # 48 -> 96
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 192

        self.conv_l_4 = conv_left(192, 2)  # 192 -> 384
        self.conv_r_4 = conv_right(192, 2)  # 768 -> 192
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 768

        self.conv_l_5 = conv_left(768, 1)  # 384 -> 768
        self.conv_r_5 = conv_right(768, 1)  # 384 -> 768
        self.max_pool_5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 1536

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc6 = nn.Linear(1536, 640, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=640, out_features=num_classes, bias=True)

    def forward(self, x):
        feature_list = []
        x1_l = self.conv_l_1(x)
        x1_r = self.conv_r_1(x)
        x1_r = self.max_pool_1(x1_r)
        x1 = torch.cat((x1_l, x1_r), dim=1)

        x2_l = self.conv_l_2(x1)
        x2_r = self.conv_r_2(x1)
        x2_r = self.max_pool_2(x2_r)
        x2 = torch.cat((x2_l, x2_r), dim=1)

        x3_l = self.conv_l_3(x2)
        x3_r = self.conv_r_3(x2)
        x3_r = self.max_pool_3(x3_r)
        x3 = torch.cat((x3_l, x3_r), dim=1)
        feature_list.append(x3)

        x4_l = self.conv_l_4(x3)
        x4_r = self.conv_r_4(x3)
        x4_r = self.max_pool_4(x4_r)
        x4 = torch.cat((x4_l, x4_r), dim=1)
        feature_list.append(x4)

        x5_l = self.conv_l_5(x4)
        x5_r = self.conv_r_5(x4)
        x5_r = self.max_pool_5(x5_r)
        x5 = torch.cat((x5_l, x5_r), dim=1)
        feature_list.append(x5)

        x6 = self.avg_pool(x5)
        x6 = self.flatten(x6)

        x6 = self.fc6(x6)
        x6 = self.relu6(x6)
        x6 = self.dropout6(x6)

        x7 = self.fc7(x6)
        return x7, feature_list


if __name__ == "__main__":
    inputs = torch.rand((2, 3, 224, 224))
    model = BinocularNet(2)
    cls, outputs = model(inputs)
    print(cls.size())
    for o in outputs:
        print(o.size())

