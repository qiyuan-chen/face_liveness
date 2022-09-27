from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import namedtuple
import math
import pdb

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# MobileFaceNet
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class GNAP(Module):
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        #self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class MobileFaceNet(Module):
    def __init__(self, input_size, embedding_size=512, output_name="GDC"):
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", 'GDC']
        assert input_size[0] in [112]
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        feature_list = []
        out = self.conv1(x)
        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        feature_list.append(out)
        out = self.conv_34(out)

        out = self.conv_4(out)
        feature_list.append(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        feature_list.append(conv_features)

        out = self.output_layer(conv_features)
        return out, feature_list


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # [n, c, 1, 1]
        out = self.sigmoid(out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 每一行的均值；
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 每一行的最大值；
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        # out = self.tanh(out) + 1  # [n, 1, 7, 7]
        out = self.sigmoid(out)
        return x * out


class Head(Module):
    def __init__(self, input_dim, embedding_size=512, classes=68, linear=True):
        super(Head, self).__init__()
        if linear:
            self.reduce_1x1 = Linear_block(input_dim, input_dim, groups=input_dim, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        else:
            self.reduce_1x1 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.flatten = Flatten()
        self.linear = Linear(input_dim, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=0.5)
        self.output = Linear(embedding_size, classes)

    def forward(self, x):
        x = self.reduce_1x1(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.output(x)
        return x


class Attention_Head(nn.Module):
    def __init__(self, in_planes, embedding_size=512, classes=68, residual=False):
        super(Attention_Head, self).__init__()
        self.c_attention = ChannelAttention(in_planes)
        self.s_attention = SpatialAttention(kernel_size=3)
        self.head = Head(in_planes, embedding_size, classes)
        self.bn = BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        c_out = self.c_attention(x)
        c_out = self.bn(c_out)
        c_out = self.relu(c_out)
        s_out = self.s_attention(c_out)
        if self.residual:
            s_out = x + s_out
        s_out = self.bn(s_out)
        s_out = self.relu(s_out)
        out = self.head(s_out)
        return out


class FPN_Head(nn.Module):
    def __init__(self, in_channels_list, out_channels, embedding_size=512, classes=68):
        # in_channels_list = [128, 128, 512]
        # features_map = [14*14, 14*14, 7*7]
        # out_channels = 256
        super(FPN_Head, self).__init__()
        self.output1 = Conv_block(in_channels_list[0], out_channels, stride=1)
        self.output2 = Conv_block(in_channels_list[1], out_channels, stride=1)
        self.output3 = Conv_block(in_channels_list[2], out_channels, stride=1)

        self.merge1 = Conv_block(out_channels, out_channels, kernel=(3, 3))
        self.merge2 = Conv_block(out_channels, out_channels, kernel=(3, 3))

        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = Linear(out_channels * 3, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=0.5)
        self.output = Linear(embedding_size, classes)

    def forward(self, inputs):
        # names = list(input.keys())
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        output1 = self.aap(output1).view(output1.shape[0], -1)  # n, out_channels
        output2 = self.aap(output2).view(output2.shape[0], -1)  # n, out_channels
        output3 = self.aap(output3).view(output3.shape[0], -1)  # n, out_channels

        out = torch.cat([output1, output2, output3], dim=1)  # out_channels * 3
        out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.output(out)
        return out
