"""
Opti-CAM implementation adapted from https://arxiv.org/abs/2301.07002
Paper: Zhang et al., "Opti-CAM: Optimizing Saliency Maps for Interpretability," 2023

Notes:
- Adapted for multitask model (segmentation + classification)
- Adjusted for medical imaging (ISIC dataset)
- Supports batch and single-image inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptiCam:
    def __init__(self, model, target_layer, steps=100, learning_rate=0.1):
        self.model = model
        self.target_layer = target_layer
        self.steps = steps
        self.learning_rate = learning_rate
        self.activations = None

    def _capture(self, module, inputs, output):
        self.activations = output

    @staticmethod
    def _classification_output(output):
        if isinstance(output, dict):
            return output["classification"]
        if isinstance(output, tuple):
            return output[1]
        return output

    @staticmethod
    def _normalize(mask):
        minimum = mask.amin(dim=(-2, -1), keepdim=True)
        maximum = mask.amax(dim=(-2, -1), keepdim=True)
        return (mask - minimum) / (maximum - minimum + 1e-8)

    def __call__(self, image, class_idx, progress_callback=None):
        image = image.detach()
        handle = self.target_layer.register_forward_hook(self._capture)
        try:
            with torch.no_grad():
                self.model(image)
            if self.activations is None or self.activations.ndim != 4:
                raise RuntimeError("Opti-CAM target layer must output a 4-D feature map.")
            features = self.activations.detach()
        finally:
            handle.remove()

        weights = nn.Parameter(torch.zeros(features.shape[1], device=image.device))
        optimizer = torch.optim.Adam([weights], lr=self.learning_rate)
        for iteration in range(self.steps):
            optimizer.zero_grad()
            coefficients = torch.softmax(weights, dim=0).view(1, -1, 1, 1)
            mask = (coefficients * features).sum(dim=1, keepdim=True)
            mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
            mask = self._normalize(mask)
            logits = self._classification_output(self.model(image * mask))
            if progress_callback is not None:
                progress_callback(iteration, mask.squeeze().detach().cpu(), float(logits[0, class_idx].detach()))
            loss = -logits[0, class_idx]
            weights.grad, = torch.autograd.grad(loss, weights)
            optimizer.step()

        with torch.no_grad():
            coefficients = torch.softmax(weights, dim=0).view(1, -1, 1, 1)
            mask = (coefficients * features).sum(dim=1, keepdim=True)
            mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
            mask = self._normalize(mask)
            if progress_callback is not None:
                logits = self._classification_output(self.model(image * mask))
                progress_callback(self.steps, mask.squeeze().detach().cpu(), float(logits[0, class_idx]))
            return mask.squeeze().detach()