"""
Grad-CAM adapated from https://github.com/leftthomas/GradCAM
Original author: Hao Ren (leftthomas)
Paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017

Notes:
- Adapted for multitask model (segmentation + classification)
- Adjusted for medical imaging (ISIC dataset)
- Added support for batch processing
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)

        raw_cams = []   # for evaluation
        vis_cams = []   # for visualization

        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)

            for name, module in self.model.named_children():
                if name == 'classifier':
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature

            classes = torch.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)

            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            cam = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)

            # Resize
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                size=image_size[::-1],
                                mode='bilinear',
                                align_corners=False).squeeze()

            cam = cam.detach().cpu().numpy()

            # Normalize 
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)

            # RAW CAM for metrics
            raw_cams.append(torch.tensor(cam))

            # VISUAL CAM for images
            heat_map = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = heat_map + np.uint8(img.transpose(1, 2, 0) * 255)
            overlay = overlay - np.min(overlay)
            if np.max(overlay) != 0:
                overlay = overlay / np.max(overlay)

            vis_cams.append(torch.tensor(overlay).permute(2, 0, 1))  # (C,H,W)

        raw_cams = torch.stack(raw_cams)   # (B, H, W)
        vis_cams = torch.stack(vis_cams)   # (B, 3, H, W)

        return raw_cams, vis_cams
