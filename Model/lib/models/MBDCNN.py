"""
Note: This is a simplified re-implementation inspired by the paper:

Xie et al., "A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification",
IEEE Transactions on Medical Imaging, 2020.

This implementation is NOT the official code and does not exactly reproduce the original architecture.
It is designed to approximate the core idea of mutual interaction between segmentation and classification
for experimental comparison purposes in this study.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MBDCNN(nn.Module):
    def __init__(self, in_channels=3, seg_out_channels=2, cls_out_channels=7):
        super(MBDCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
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

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128 + seg_out_channels, cls_out_channels)

    def forward(self, x):
        feat = self.encoder(x)

        seg_out = self.seg_head(feat)

        seg_resized = F.interpolate(seg_out, size=feat.shape[2:], mode='bilinear', align_corners=False)

        cls_feat = torch.cat([feat, seg_resized], dim=1)
        cls_feat = self.pool(cls_feat)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_out = self.fc(cls_feat)

        cls_weight = torch.softmax(cls_out, dim=1).unsqueeze(-1).unsqueeze(-1)
        seg_out = seg_out * cls_weight[:, :seg_out.shape[1], :, :]

        return {
            "segmentation": seg_out,
            "classification": cls_out
        }
