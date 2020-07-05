from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # https://hangzhang.org/PyTorch-Encoding/notes/compile.html
    # pip install torch-encoding --pre   
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4


def _ConvBnReLU(in_ch, out_ch, kernel_size, stride, padding, dilation=1, relu=True):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
        _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999)
    ]

    if relu:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, kernel_size=1, stride=stride, padding=0, dilation=1, relu=True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, relu=False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Sequential()  # identity
            # else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResStage(nn.Module):

    def __init__(self,n_blocks, in_ch, out_ch, stride, dilation, multi_grids=None):
        """
        Residual stage with multi grids
        """
        super(_ResStage,self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_blocks)]
        else:
            assert n_blocks == len(multi_grids)

        # Downsampling is only in the first block
        layers = []
        for i in range(n_blocks):
            layers.append((
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False))
                ))
        self.layers = nn.Sequential(OrderedDict(layers))

     
    def forward(self,input):
        return self.layers(input)


class _Stem(nn.Module):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.layers = nn.Sequential(
            _ConvBnReLU(in_ch=3, out_ch=out_ch, kernel_size=7, stride=2, padding=3, dilation=1),
            nn.MaxPool2d(3, 2, 1, ceil_mode=False)
        )
    
    def forward(self,input):
        return self.layers(input)

class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNet(nn.Module):
    def __init__(self, n_classes, n_blocks):
        super(ResNet, self).__init__()
        ch = [64 * 2 ** p for p in range(6)] # channel numbers: 64 128 256 512 1024 2048
        print(ch)
        self.stages = nn.Sequential(OrderedDict([
            ("conv1", _Stem(ch[0])),
            ("conv2_x", _ResStage(n_blocks=n_blocks[0], in_ch=ch[0], out_ch=ch[2], stride=1, dilation=1)),
            ("conv3_x", _ResStage(n_blocks=n_blocks[1], in_ch=ch[2], out_ch=ch[3], stride=2, dilation=1)),
            ("conv4_x", _ResStage(n_blocks=n_blocks[2], in_ch=ch[3], out_ch=ch[4], stride=2, dilation=1)),
            ("conv5_x", _ResStage(n_blocks=n_blocks[3], in_ch=ch[4], out_ch=ch[5], stride=2, dilation=1)),
            ("pool5", nn.AdaptiveAvgPool2d(1)),
            ("flatten", _Flatten()),
            ("fc", nn.Linear(ch[5], n_classes)),
        ]))

    def forward(self,input):
        return self.stages(input)


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Improved Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        
        stages = [("c0", _ConvBnReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1),)]
        for i, rate in enumerate(rates):
            stages.append(
                ("c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, kernel_size=3, stride=1, padding=rate, dilation=rate),)
            )
        stages.append(("image_pool", _ImagePool(in_ch, out_ch),))
        self.stages = nn.Sequential(OrderedDict(stages))

    """
    NOTICE: here applies concatenation, different from the ASPP of DeepLab v2
    """
    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class DeepLab_v3(nn.Module):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLab_v3, self).__init__()

        # output_stride describes the reduced size of output 
        if output_stride == 8:
            s = [1, 2, 1, 1] # strides
            d = [1, 1, 2, 4] # dilations
        elif output_stride == 16:
            s = [1, 2, 2, 1] # strides
            d = [1, 1, 1, 2] # dilations

        ch = [64 * 2 ** p for p in range(6)]
        output_channel = 256
        concat_ch = output_channel * (len(atrous_rates) + 2)
        self.stages = nn.Sequential(OrderedDict([
            ("stage1",_Stem(ch[0])),
            ("stage2",_ResStage(n_blocks=n_blocks[0], in_ch=ch[0], out_ch=ch[2], stride=s[0], dilation=d[0])),
            ("stage3",_ResStage(n_blocks=n_blocks[1], in_ch=ch[2], out_ch=ch[3], stride=s[1], dilation=d[1])),
            ("stage4",_ResStage(n_blocks=n_blocks[2], in_ch=ch[3], out_ch=ch[4], stride=s[2], dilation=d[2])),
            ("stage5",_ResStage(n_blocks=n_blocks[3], in_ch=ch[4], out_ch=ch[5], stride=s[3], dilation=d[3], multi_grids=multi_grids)),
            ("aspp", _ASPP(in_ch=ch[5], out_ch=output_channel, rates=atrous_rates)),
            ("fc1", _ConvBnReLU(in_ch=concat_ch, out_ch=output_channel, kernel_size=1, stride=1, padding=0)),
            ("fc2", nn.Conv2d(in_channels=output_channel, out_channels=n_classes, kernel_size=1))
        ]))

    def forward(self,input):
        N,C,H,W = input.shape
        x = self.stages(input)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    model = DeepLab_v3(
        n_classes=2,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    model.eval()
    image = torch.randn(1, 3, 512, 512)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)