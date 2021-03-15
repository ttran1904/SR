
import torch.nn as nn
import torch

from typing import Optional
from typing import Callable
from torchvision.models.resnet import BasicBlock

# 5x5 resnet block


class NoNormRes5(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity()
    ) -> None:
        super(NoNormRes5, self).__init__(inplanes, planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                               padding=2, groups=groups, bias=False, dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride,
                               padding=2, groups=groups, bias=False, dilation=dilation)
        # self.bn1 = nn.Identity()
        # self.bn2 = nn.Identity()


class MDSR(nn.Module):

    def __init__(self, f_channels=64, hr_res=(1024, 1024), multiplier=2):
        # self.hr_res = hr_res
        # self.lr_res = tuple(dim / multiplier for dim in hr_res)

        super(MDSR, self).__init__()

        # 3x3 convolution at beginning
        self.prior_conv_0 = nn.Conv2d(3, f_channels, 3, padding=1)

        # 2 5x5 convolutions specific to the multiplier
        self.prior_conv_1 = NoNormRes5(f_channels, f_channels)
        self.prior_conv_2 = NoNormRes5(f_channels, f_channels)

        # 16 or 80 3x3 resnet-relu blocks
        self.shared_blocks = []
        for _ in range(80):
            self.shared_blocks.append(BasicBlock(
                f_channels, f_channels, norm_layer=nn.Identity).cuda())

        # sub-pixel convolution for upscaling
        self.upscale_pre = nn.Conv2d(
            f_channels, 3 * multiplier * multiplier, 3, padding=1)
        self.upscale = nn.PixelShuffle(multiplier)

        # 2 3x3 resnet-relu blocks for result
        self.out_pre_0 = BasicBlock(3, 3, norm_layer=nn.Identity, dilation=1)
        self.out_pre_1 = BasicBlock(3, 3, norm_layer=nn.Identity, dilation=1)

    def forward(self, x):
        x = self.prior_conv_0(x)
        x = self.prior_conv_1(x)
        x = self.prior_conv_2(x)
        res_x = x

        for shared_block in self.shared_blocks:
            x = shared_block(x)

        x = x + res_x
        x = self.upscale_pre(x)
        x = self.upscale(x)
        x = self.out_pre_0(x)
        x = self.out_pre_1(x)

        return x