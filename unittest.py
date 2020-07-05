import torch
from model import *


def test_unet():
    unet = UNet(3,2,True)
    input = torch.randn((1,3,512,512))
    unet.forward(input)

def test_segnet():
    net = SegNet(2,vgg_trainable=False)
    im = torch.rand((1,3,512,512))
    print(im.shape)
    output = net.forward(im)
    print(output.shape)

def test_deeplab_v1():
    net = DeepLab_v1(3, 2)
    net.eval()
    in_ten = torch.randn(1, 3, 224, 224)
    print(in_ten.size())
    out = net(in_ten)
    print(out.size())

def test_deeplab_v2():
    # model = DeepLab_v2(
    #     n_classes=2, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    # )
    model = Res_Deeplab(2)
    model.eval()
    image = torch.randn(1, 3, 512, 512)

    # print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)


def test_deeplab_v3_plus():
    net = DeepLab_v3_plus()
    net.eval()
    im = torch.rand((1,3,513,513))
    # print(im.shape)
    net.eval()
    output = net.forward(im)
    print(output.shape)

if __name__ == "__main__":
    # test_deeplab_v1()
    test_deeplab_v2()