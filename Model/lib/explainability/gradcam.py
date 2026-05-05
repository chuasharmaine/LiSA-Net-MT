"""
Grad-CAM adapated from https://github.com/leftthomas/GradCAM
Original author: Hao Ren (leftthomas)
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

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam