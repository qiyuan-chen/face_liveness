#!usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2021/9/8 13:42
# @Author: Chen Jizhi
# @E-mail: chenjizhi@ruijie.com
# @File: detector.py
# @Software: PyCharm
from .mobilefacenet import MobileFaceNet, Head, FPN_Head, Attention_Head
from .mobilenet import MobileNet, MobileNet_GDConv, MobileNet_GDConv_SE
from .resnet import ResNet50
from .senet50_ft_dag import senet50_ft_dag
from .inception_v3 import inception_v3
from .vgg16 import vgg_16
from .swim_transformer import SwinTransformer
from .inception_v4 import Inceptionv4
from .inception_resnet_v2 import Inception_ResNetv2
from .cdcn import CDCN_v1, CDCN_v2
from .binocular import BinocularNet
from .xception import xception
import torch.nn as nn
import torch.nn.functional as F
import torch


if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'


class Liveness(nn.Module):
    def __init__(self, conf):
        super(Liveness, self).__init__()
        self.conf = conf
        if conf.model_name == 'mobileface':
            self.backbone = MobileFaceNet([112, 112], 2)
            dim = [64, 128, 512]
        elif conf.model_name == 'mobilenet_v1':
            self.backbone = MobileNet(2)
            dim = [64, 128, 256]
        elif conf.model_name == 'mobilenet_v2':
            self.backbone = MobileNet_GDConv_SE(2)
            dim = [32, 96, 1280]
        elif conf.model_name == 'resnet50':
            self.backbone = ResNet50(2, pretrained=False)
            dim = [512, 1024, 2048]
        elif conf.model_name == 'inception_v3':
            self.backbone = inception_v3(2)
            dim = [288, 768, 2048]
        elif conf.model_name == 'vgg16':
            self.backbone = vgg_16(2)
            dim = [256, 512, 512]
        elif conf.model_name == 'swim':
            self.backbone = SwinTransformer(num_classes=2)
            dim = [192, 384, 768]
        elif conf.model_name == 'cdcn_v1':
            self.backbone = CDCN_v1(2)
            dim = [384, 768, 1280]
        elif conf.model_name == 'cdcn_v2':
            self.backbone = CDCN_v2(2)
            dim = [384, 768, 1280]
        elif conf.model_name == 'xception':
            self.backbone = xception(2)
            dim = [256, 768, 1024]
        elif conf.model_name == 'binocular':
            self.backbone = BinocularNet(2)
            dim = [192, 768, 1536]
        elif conf.model_name == 'inception_resnet_v2':
            self.backbone = Inception_ResNetv2(classes=2)
            dim = [320, 1088, 2080]
        else:
            raise ValueError("The model name is unknown!")
        if tuple(conf.depth_size) == (28, 28):
            self.feat_id = 0
        elif tuple(conf.depth_size) == (14, 14):
            self.feat_id = 1
        elif tuple(conf.depth_size) == (7, 7):
            self.feat_id = 2
        else:
            raise ValueError("The depth size should be 28, 14, 7")

        # self.upsample = nn.Upsample(tuple(conf.depth_size), mode='bilinear', align_corners=True)

        self.depth_final = nn.Conv2d(dim[self.feat_id], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cls_out, conv_features = self.backbone(x)
        depth_map = self.depth_final(conv_features[self.feat_id])
        depth_map = self.sigmoid(depth_map)
        # depth_map = self.upsample(depth_map)
        depth_map = F.interpolate(depth_map, size=self.conf.depth_size, mode="nearest")
        depth_map = torch.squeeze(depth_map, dim=1)
        if self.training:
            return cls_out, depth_map
        else:
            return F.softmax(cls_out, dim=-1), depth_map

