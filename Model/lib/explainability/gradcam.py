"""
Grad-CAM adapated from https://github.com/leftthomas/GradCAM
Paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017

Notes:
- Adapted for multitask model (segmentation + classification)
- Adjusted for medical imaging (ISIC dataset)
- Added support for batch processing
"""

import torch
import torch.nn.functional as F

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, cls_out, class_idx):
        self.model.zero_grad()
        score = cls_out[0, class_idx]
        score.backward(retain_graph=True)

        # GAP on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        # upsample to input size
        cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False).squeeze()
        
        # normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() - cam.min() + 1e-8)
        return cam.detach()