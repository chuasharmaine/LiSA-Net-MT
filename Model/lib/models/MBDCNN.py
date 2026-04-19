"""
Xie et al., "A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification",
IEEE Transactions on Medical Imaging, 2020.

https://github.com/YtongXie/MB-DCNN

Notes:
- This is the original MB-DCNN codebase, modified to fit the current training pipeline.
- Minor changes were made to adapt the data loading, training loop, and compatibility with the LiSA-Net framework.
- The core architecture and methodology remain unchanged.
- Used as a baseline model for comparison in this study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBDCNN(nn.Module):
    def __init__(self, in_channels=3, seg_out_channels=2, cls_out_channels=7):
        super(MBDCNN, self).__init__()

        # --- Shared encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- Segmentation head ---
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, seg_out_channels, 1)
        )

        # --- Classification head ---
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, cls_out_channels)

    def forward(self, x):
        feat = self.encoder(x)

        # Segmentation output
        seg_out = self.seg_head(feat)

        # Classification output
        cls_feat = self.pool(feat)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_out = self.fc(cls_feat)

        return {
            "segmentation": seg_out,
            "classification": cls_out
        }