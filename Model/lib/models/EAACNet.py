"""
Note: This is a simplified re-implementation inspired by the paper:

Fan et al., "EAAC-Net: An Efficient Adaptive Attention and Convolution Fusion Network for Skin Lesion Segmentation",
Journal of Digital Imaging, 2025.

This implementation is NOT the official code and does not exactly reproduce the original architecture.
It approximates the key concept of attention-enhanced convolutional feature fusion
for experimental comparison purposes in this study.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class EAACNet(nn.Module):
    def __init__(self, in_channels=3, seg_out_channels=2, cls_out_channels=7):
        super(EAACNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.attention = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.Sigmoid()
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, seg_out_channels, 1)
        )

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(128, cls_out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        attn = self.attention(x)
        feat = x * attn

        seg_out = self.seg_head(feat)

        cls_feat = self.cls_pool(feat)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_out = self.cls_fc(cls_feat)

        return {
            "segmentation": seg_out,
            "classification": cls_out
        }
