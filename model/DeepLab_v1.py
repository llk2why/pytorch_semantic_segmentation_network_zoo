#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


# refer to https://github.com/CoinCheung/Deeplab-Large-FOV/blob/master/lib/model.py
class DeepLab_v1(nn.Module):
    def __init__(self, in_dim, out_dim,pretrained=False, *args, **kwargs):
        super(DeepLab_v1, self).__init__(*args, **kwargs)

        # vgg16 = torchvision.models.vgg16()
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=pretrained)
        """ 
        vgg16_bn stage 1: [0:7]
        vgg16_bn stage 2: [7:14]
        vgg16_bn stage 3: [14:24]
        vgg16_bn stage 4: [24:33] without Pooling Layer
        """
        self.vgg_part = vgg16_bn.features[:33]
        self.vgg_part.add_module("33",nn.MaxPool2d(3, stride=1, padding=1))
        # print(self.vgg_part)
        # exit()

        """  like vgg stage 5 """ 
        self.dialated_part = nn.Sequential(*[
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1)
        ])

        """ upsampling block """
        self.upsample = nn.Sequential(*[
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=4,dilation=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(1024, out_dim, kernel_size=1),
            # nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        ])

        self.init_weights()
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = self.vgg_part(x)
        x = self.dialated_part(x)
        x = self.upsample(x)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x

    """ weights initialization """
    def init_weights(self):
        for layer in self.dialated_part.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


        for layer in self.upsample.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


        

