#!usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2021/9/10 15:48
# @Author: Chen Jizhi
# @E-mail: chenjizhi@ruijie.com
# @File: resnet.py
# @Software: PyCharm
import torch.nn as nn
import torchvision.models as models
import torchvision.models._utils as _utils
import torch


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50, self).__init__()
        backbone = models.resnet50(pretrained=pretrained)  # 1000
        # backbone = models.resnet101(pretrained=pretrained)
        return_layers = {'layer2': 'feat1', 'layer3': 'feat2', 'layer4': 'feat3'}
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        self.bn = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(p=0.5)
        self.output = nn.Linear(1000, num_classes)

    def forward(self, x):
        return_features = self.body(x)
        feature_list = [return_features['feat1'], return_features['feat2'], return_features['feat3']]
        conv_features = return_features['feat3']
        out = self.avgpool(conv_features)
        out = torch.flatten(out, 1)
        fc_out = self.fc(out)
        out = self.bn(fc_out)
        out = self.drop(out)
        out = self.output(out)
        return out, feature_list


if __name__ == "__main__":
    inputs = torch.rand((2, 3, 224, 224))
    model = ResNet50(2)
    cls, outputs = model(inputs)
    print(cls.size())
    for o in outputs:
        print(o.size())
