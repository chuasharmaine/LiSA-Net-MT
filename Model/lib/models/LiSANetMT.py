# -*- encoding: utf-8 -*-
"""
@author   :   andredalwin (LiSA-Net) + chuasharmaine (LiSA-Net-MT)
@Contact  :   sharmainechua134@gmail.com
@DateTime :   2026/02/25
@Version  :   2.0
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from lib.models.modules.LiSAConvBlock import ConvBlock
from lib.models.modules.LiSALocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.LiSAGlobalPMFSBlock import GlobalPMFSBlock_AP_Separate
from lib.models.modules.LiSASEBlock import SEBlock

class LiSANetMT(nn.Module):
    def __init__(self, in_channels=1, seg_out_channels=2, cls_out_channels=7, dim="3d", scaling_version="TINY",
                 basic_module=DownSampleWithLocalPMFSBlock,
                 global_module=GlobalPMFSBlock_AP_Separate,
                 segmentation=True, classification=True):
        super(LiSANetMT, self).__init__()
        
        self.segmentation = segmentation
        self.classification = classification
        self.scaling_version = scaling_version
        self.dim = dim

        if scaling_version == "BASIC":
            base_channels = [24, 48, 64]
            skip_channels = [24, 48, 64]
            units = [5, 10, 10]
            pmfs_ch = 64
        elif scaling_version == "SMALL":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            pmfs_ch = 48
        elif scaling_version == "TINY":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [3, 5, 5]
            pmfs_ch = 48
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")

        if dim == "3d":
            upsample_mode = 'trilinear'
            self.classifier_pool = nn.AdaptiveAvgPool3d(1)
        elif dim == "2d":
            upsample_mode = 'bilinear'
            self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise RuntimeError(f"{dim} dimension is error")
        
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]

        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                basic_module(
                    in_channel=(in_channels if i == 0 else downsample_channels[i - 1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    downsample=True,
                    skip=((i < 2) if scaling_version == "BASIC" else True),
                    dim=dim
                )
            )

        self.Global = global_module(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=pmfs_ch,
            ch_k=pmfs_ch,
            ch_v=pmfs_ch,
            br=3,
            dim=dim
        )

        if self.classification:
            cls_in_channels = downsample_channels[-1]
            if self.segmentation and seg_out_channels is not None:
                cls_in_channels += seg_out_channels

            self.cls_se = SEBlock(cls_in_channels, reduction=8, dim=dim)
            self.classifier_fc = nn.Sequential(
                nn.Linear(cls_in_channels, cls_in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(cls_in_channels // 2, cls_out_channels)
            )
        else:
            self.classifier_fc = None

        if scaling_version == "BASIC":
            self.up2 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv2 = basic_module(in_channel=downsample_channels[2] + skip_channels[1],
                                         base_channel=base_channels[1],
                                         kernel_size=3,
                                         unit=units[1],
                                         growth_rate=growth_rates[1],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)

            self.up1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv1 = basic_module(in_channel=downsample_channels[1] + skip_channels[0],
                                         base_channel=base_channels[0],
                                         kernel_size=3,
                                         unit=units[0],
                                         growth_rate=growth_rates[0],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)
        else:
            self.bottle_conv = ConvBlock(
                in_channel=downsample_channels[2] + skip_channels[2],
                out_channel=skip_channels[2],
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim,
                use_se=True
            )

            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode=upsample_mode)

        if self.segmentation and seg_out_channels is not None:
            self.out_conv = ConvBlock(
                in_channel=(downsample_channels[0] if scaling_version == "BASIC" else sum(skip_channels)),
                out_channel=seg_out_channels,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )
            self.upsample_out = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
        else:
            self.out_conv = None
            self.upsample_out = None

    def forward(self, x):
        outputs = {}
        
        if self.scaling_version == "BASIC":
            x1, x1_skip = self.down_convs[0](x)
            x2, x2_skip = self.down_convs[1](x1)
            x3 = self.down_convs[2](x2)

            features = self.Global([x1, x2, x3])

            if self.segmentation:
                d2 = self.up2(features)
                d2 = torch.cat((x2_skip, d2), dim=1)
                d2 = self.up_conv2(d2)

                d1 = self.up1(d2)
                d1 = torch.cat((x1_skip, d1), dim=1)
                d1 = self.up_conv1(d1)

                if self.segmentation and self.out_conv is not None:
                    seg_out = self.out_conv(d1)
                    seg_out = self.upsample_out(seg_out)

                    outputs["segmentation"] = seg_out

        else:
            x1, x1_skip = self.down_convs[0](x)
            x2, x2_skip = self.down_convs[1](x1)
            x3, x3_skip = self.down_convs[2](x2)

            features = self.Global([x1, x2, x3])

            if self.segmentation:            
                x3_skip = self.bottle_conv(torch.cat([features, x3_skip], dim=1))
                x2_skip = self.upsample_1(x2_skip)
                x3_skip = self.upsample_2(x3_skip)

                seg_out = self.out_conv(torch.cat([x1_skip, x2_skip, x3_skip], dim=1))
                seg_out = self.upsample_out(seg_out)

                outputs["segmentation"] = seg_out

        if self.classification and self.classifier_fc is not None:
            cls_features = features

            if self.segmentation and "segmentation" in outputs:
                # Resize segmentation output to match feature map
                seg_out = outputs["segmentation"]
                seg_resized = nn.functional.interpolate(
                    seg_out, 
                    size=features.shape[2:], 
                    mode='trilinear' if self.dim=='3d' else 'bilinear',
                    align_corners=False
                )
                cls_features = torch.cat([features, seg_resized], dim=1)

            cls_features = self.cls_se(cls_features)
            cls_features = self.classifier_pool(cls_features)
            cls_features = torch.flatten(cls_features, 1)
            cls_out = self.classifier_fc(cls_features)
            outputs["classification"] = cls_out

        if len(outputs) == 1:
            return list(outputs.values())[0]
        return outputs

if __name__ == '__main__':
    from thop import profile

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dummy_inputs = {
        "2d": torch.randn((1, 3, 224, 224)).to(device),
        "3d": torch.randn((1, 1, 160, 160, 96)).to(device)
    }

    dims = ["3d", "2d"]
    in_channels = {"2d": 3, "3d": 1}

    for dim in dims:
        x = dummy_inputs[dim]

        # Multitask model
        model_mt = LiSANetMT(
            in_channels=in_channels[dim],
            seg_out_channels=2,
            cls_out_channels=7,
            dim=dim,
            scaling_version="BASIC",
            segmentation=True,
            classification=True
        ).to(device)
        flops_mt, params_mt = profile(model_mt, inputs=(x,), verbose=False)
        print(f"\n{dim.upper()} - MULTITASK (seg + cls)")
        print(f"Params: {count_parameters(model_mt):.4f} M | FLOPs: {flops_mt/1e9:.4f} G")

        # Segmentation only
        model_seg = LiSANetMT(
            in_channels=in_channels[dim],
            seg_out_channels=2,
            cls_out_channels=7,
            dim=dim,
            scaling_version="BASIC",
            segmentation=True,
            classification=False
        ).to(device)
        flops_seg, params_seg = profile(model_seg, inputs=(x,), verbose=False)
        print(f"\n{dim.upper()} - SEGMENTATION ONLY")
        print(f"Params: {count_parameters(model_seg):.4f} M | FLOPs: {flops_seg/1e9:.4f} G")

        # Classification only
        model_cls = LiSANetMT(
            in_channels=in_channels[dim],
            seg_out_channels=2,
            cls_out_channels=7,
            dim=dim,
            scaling_version="BASIC",
            segmentation=False,
            classification=True
        ).to(device)
        flops_cls, params_cls = profile(model_cls, inputs=(x,), verbose=False)
        print(f"\n{dim.upper()} - CLASSIFICATION ONLY")
        print(f"Params: {count_parameters(model_cls):.4f} M | FLOPs: {flops_cls/1e9:.4f} G")