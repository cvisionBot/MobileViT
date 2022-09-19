# Lib
import torch
from torch import nn

from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import MV2Block, MobileViTBlock
from ..initialize import weight_initialize


class MobileViT(nn.Module):
    def __init__(self, dims, in_channels, channels, classes, expansion=4, patch_size=(2, 2)):
        super(MobileViT, self).__init__()
        L = [2, 4, 3]
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=2, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())
        self.conv2 = Conv2dBnAct(in_channels=channels[-2], out_channels=channels[-1], kernel_size=1, stride=1, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())
        
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], 3, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], 3, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], 3, patch_size, int(dims[2]*4)))

        self.pool = nn.AvgPool2d(256//32, 1)
        self.fc = nn.Linear(channels[-1], classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.mv2[0](output)

        output = self.mv2[1](output)
        output = self.mv2[2](output)
        output = self.mv2[3](output)

        output = self.mv2[4](output)
        output = self.mvit[0](output)

        output = self.mv2[5](output)
        output = self.mvit[1](output)

        output = self.mv2[6](output)
        output = self.mvit[2](output)

        output = self.conv2(output)

        output = self.pool(output).view(-1, output.shape[1])
        output = self.fc(output)
        return {'pred' : output}


def MobileViT_classification(in_channels, classes=1000, varient='s'):
    if varient == 's':
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        model = MobileViT(dims=dims, in_channels=in_channels, channels=channels, classes=classes)
    elif varient == 'xs':
        dims = [96, 120, 144]
        channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        model = MobileViT(dims=dims, in_channels=in_channels, channels=channels, classes=classes)
    else:
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        model = MobileViT(dims=dims, in_channels=in_channels, channels=channels, classes=classes)

    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = MobileViT_classification(in_channels=3, classes=1000, varient='s')
    model(torch.rand(1, 3, 256, 256))
    print(model)