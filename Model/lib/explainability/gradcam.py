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
    def __init__(self, model, target_layer, output_key="Classification"):
        self.model = model
        self.gradients = None
        self.activations = None

        self.handle = target_layer.register_forward_hook(self.forward_hook)

    def close(self):
        self.handle.remove()

    def forward_hook(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):
        self.gradients = gradient

    def __call__(self, model, input_image, class_idx):
        self.gradients = None
        model.zero_grad()
        output = self.model(input_image)
        if isinstance(output, dict):
            output = output["classification"]
        elif isinstance(output, tuple):
            output = output[1]
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError("Gradients empty. Wrong targer layer.")

        # GAP on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        # upsample to input size
        cam = F.interpolate(cam.unsqueeze(1), size=input_image.shape[-2:], mode="bilinear", align_corners=False).squeeze()
        
        # normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() - cam.min() + 1e-8)
        return cam.detach()