"""
@author   :   JCruan519    
@Contact  :   jackchenruan@sjtu.edu.cn
@DateTime :   2023/07/17 
@Version  :   1.0
"""

"""
@Modifiedby :   chuasharmaine
@DateTime   :   2026/02/23
@Note       :   Framework compatibility adaptation. 
                Core EGEUNet architecture remains unchanged.
"""

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        self.tail_conv = nn.Conv2d(dim_xl, dim_xl, 1)

    def forward(self, xh, xl):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        x = self.tail_conv(xh + xl)
        return x

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=1
        
        self.params_xy = nn.Parameter(torch.ones(1, c_dim_in, 8, 8))
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1)
        )

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in)
        self.norm2 = LayerNorm(dim_in)
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        param = F.interpolate(self.params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True)
        x1 = x1 * self.conv_xy(param)
        x4 = self.dw(x4)
        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class EGEUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True):
        super().__init__()
        self.bridge = bridge
        
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, padding=1)
        self.encoder4 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3])
        self.encoder5 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4])
        self.encoder6 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5])

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])

        self.decoder1 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4])
        self.decoder2 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3])
        self.decoder3 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2])
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, padding=1)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        t1 = F.gelu(F.max_pool2d(self.encoder1(x),2))
        t2 = F.gelu(F.max_pool2d(self.encoder2(t1),2))
        t3 = F.gelu(F.max_pool2d(self.encoder3(t2),2))
        t4 = F.gelu(F.max_pool2d(self.encoder4(t3),2))
        t5 = F.gelu(F.max_pool2d(self.encoder5(t4),2))
        t6 = F.gelu(self.encoder6(t5))
        
        out5 = self.decoder1(t6)
        out5 = self.GAB5(t6, t5) if self.bridge else t5

        out4 = F.interpolate(self.decoder2(out5), scale_factor=2, mode='bilinear', align_corners=True)
        out4 = self.GAB4(out5, t4) if self.bridge else t4

        out3 = F.interpolate(self.decoder3(out4), scale_factor=2, mode='bilinear', align_corners=True)
        out3 = self.GAB3(out4, t3) if self.bridge else t3

        out2 = F.interpolate(self.decoder4(out3), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = self.GAB2(out3, t2) if self.bridge else t2

        out1 = F.interpolate(self.decoder5(out2), scale_factor=2, mode='bilinear', align_corners=True)
        out1 = self.GAB1(out2, t1) if self.bridge else t1

        out0 = F.interpolate(self.final(out1), scale_factor=2, mode ='bilinear', align_corners=True)
        return out0