import torch
from torch import nn

from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.blocks import Residual_Block
from ..initialize import weight_initialize


class ResNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.conv(input)
        output = self.max_pool(output)
        return output


class Make_Layers(nn.Module):
    def __init__(self, layers_configs, pre_layer_ch):
        super(Make_Layers, self).__init__()
        self.pre_ch = pre_layer_ch
        self.layers_configs = layers_configs
        self.layer = self.residual_layer(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def residual_layer(self, cfg):
        layers = []
        input_ch = cfg[0]
        for i in range(cfg[-1]):
            if i == 0:
                layer = Residual_Block(in_channels=self.pre_ch, kernel_size=cfg[1], out_channels=cfg[2], stride=cfg[3])
            else:
                layer = Residual_Block(in_channels=input_ch, kernel_size=cfg[1], out_channels=cfg[2])
            layers.append(layer)
            input_ch = layer.get_channel()
        return nn.Sequential(*layers)


class _ResNet101(nn.Module):
    def __init__(self, in_channels, classes):
        super(_ResNet101, self).__init__()
        self.resnetStem = ResNetStem(in_channels=in_channels, out_channels=64)
        # config = in_channels, kernel_size, out_channels, stride, inter_cnt
        conv2_x = [64, 3, 256, 1, 3]
        conv3_x = [128, 3, 512, 2, 4]
        conv4_x = [256, 3, 1024, 2, 23]
        conv5_x = [512, 3, 2048, 2, 3]

        self.layer1 = Make_Layers(conv2_x, 64)
        self.layer2 = Make_Layers(conv3_x, 256)
        self.layer3 = Make_Layers(conv4_x, 512)
        self.layer4 = Make_Layers(conv5_x, 1024)
        
        self.classifier = nn.Sequential(
            Conv2dBnAct(in_channels=2048, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.resnetStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classifier(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred': pred}


def ResNet(in_channels, classes=1000, varient=101):
    if varient == 101:
        model = _ResNet101(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = ResNet(in_channels=3, classes=1000, varient=50)
    model(torch.rand(1, 3, 256, 256))