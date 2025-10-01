"""
@author   :   andredalwin    
@modified :   chuasharmaine
@DateTime :   2025/10/01
@Version  :   1.0
"""

import torch
import torch.nn as nn

from lib.models.modules.LiSAECABlock import ECABlock

class ConvBlock(torch.nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim="2d",
            use_eca=True
    ):
        super().__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise ValueError(f"Invalid dimension '{dim}' for ConvBlock")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(inplace=True),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False
                ),
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False
                ),
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU(inplace=True))

        self.conv = torch.nn.Sequential(*layers)

        # Implement ECA
        self.use_eca = use_eca
        if use_eca:
            self.eca_block = ECABlock(out_channel, k_size=3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_eca:
            x = self.eca_block(x)
        return x

class SingleConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, dim="2d"):
        super(SingleConvBlock, self).__init__()

        if dim == "3d":
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dim == "2d":
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError(f"Invalid dimension '{dim}' for SingleConvBlock")

        self.conv = nn.Sequential(
            conv(in_channel, out_channel, kernel_size, stride, kernel_size // 2, bias=False),
            bn(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim="2d"
    ):
        super(DepthWiseSeparateConvBlock, self).__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise ValueError(f"Invalid dimension '{dim}' for DepthWiseSeparateConvBlock")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(inplace=True),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU(inplace=True))

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)