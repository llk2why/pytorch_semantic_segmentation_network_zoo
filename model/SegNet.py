import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# Refer to https://github.com/bodokaiser/piwise/blob/master/piwise/network.py
def _SegNetDecodeBlock(in_channels, out_channels, filling_layer_num):
    layers = [
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    ]
    layers += [
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    ] * filling_layer_num
    layers += [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)



class SegNet(nn.Module):

    def __init__(self,classes=2,vgg_trainable=True):
        super().__init__()
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=False)
        features = vgg16_bn.features
        
        self.enc1 = features[0:6]    # vgg16_bn stage 1 without pooling layer 3   => 64
        self.enc2 = features[7:13]   # vgg16_bn stage 2 without pooling layer 64  => 128
        self.enc3 = features[14:23]  # vgg16_bn stage 3 without pooling layer 128 => 256
        self.enc4 = features[24:33]  # vgg16_bn stage 4 without pooling layer 256 => 512
        self.enc5 = features[34:-1]  # vgg16_bn stage 5 without pooling layer 512 => 512

        # print(self.modules)
        if not vgg_trainable:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.requires_grad = False
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad = False

        self.dec5 = _SegNetDecodeBlock(512, 512, 1)      #  decoder 512 => 512 => 512
        self.dec4 = _SegNetDecodeBlock(512, 256, 1)      #  decoder 512 => 512 => 256
        self.dec3 = _SegNetDecodeBlock(256, 128, 1)      #  decoder 256 => 256 => 128
        self.dec2 = _SegNetDecodeBlock(128, 64, 0)       #  decoder 128 => 64
        self.dec1 = _SegNetDecodeBlock(64, classes, 0)   #  decoder 64  => class_num

    def forward(self, x):
        # encoding stage
        x1 = self.enc1(x)
        down1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.enc2(down1)
        down2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.enc3(down2)
        down3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.enc4(down3)
        down4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.enc5(down4)
        down5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        # decoding stage
        up4    = self.dec5(F.max_unpool2d(down5, m5, kernel_size=2, stride=2, output_size=x5.size()))
        up3    = self.dec4(F.max_unpool2d(up4,   m4, kernel_size=2, stride=2, output_size=x4.size()))
        up2    = self.dec3(F.max_unpool2d(up3,   m3, kernel_size=2, stride=2, output_size=x3.size()))
        up1    = self.dec2(F.max_unpool2d(up2,   m2, kernel_size=2, stride=2, output_size=x2.size()))
        output = self.dec1(F.max_unpool2d(up1,   m1, kernel_size=2, stride=2, output_size=x1.size()))

        return output