import torch
from torch import nn
from einops import rearrange

from ..layers.convolution import Conv2dBn, Conv2dBnAct
from ..layers.transformer import Transformer

class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, exp=4):
        super(MV2Block, self).__init__()
        self.stride = stride
        self.expansion = int(in_channels * exp)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=self.expansion, kernel_size=1, stride=1, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())
        self.conv2 = Conv2dBnAct(in_channels=self.expansion, out_channels=self.expansion, kernel_size=3, stride=stride, dilation=1,
                        groups=self.expansion, padding_mode='zeros', act=nn.SiLU())
        self.conv3 = Conv2dBn(in_channels=self.expansion, out_channels=out_channels, kernel_size=1, stride=1, dilation=1,
                        groups=1, padding_mode='zeros')
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.use_res_connect:
            input = self.identity(input)
            output = output + input
        return output


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super(MobileViTBlock, self).__init__()
        self.ph, self.pw = patch_size

        self.conv1 = Conv2dBnAct(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=1, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())
        self.conv2 = Conv2dBnAct(in_channels=channel, out_channels=dim, kernel_size=1, stride=1, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())

        self.conv3 = Conv2dBnAct(in_channels=dim, out_channels=channel, kernel_size=1, stride=1, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())
        self.conv4 = Conv2dBnAct(in_channels=(2 * channel), out_channels=channel, kernel_size=kernel_size, stride=1, dilation=1,
                        groups=1, padding_mode='zeros', act=nn.SiLU())

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

    def forward(self, input):
        
        # Local Representations
        output = self.conv1(input)
        output = self.conv2(output)

        # Global Representations
        _, _, h, w = output.shape
        output = rearrange(output, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        output = self.transformer(output)
        output = rearrange(output, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h = h // self.ph, w = w // self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        output = self.conv3(output)
        output = torch.cat((output, input), 1)
        output = self.conv4(output)
        return output