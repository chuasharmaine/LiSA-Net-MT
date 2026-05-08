"""
SHAP implementation adapted from https://github.com/slundberg/shap
Paper: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017

Notes:
- Adapted for multitask model (segmentation + classification)
- Adjusted for medical imaging (ISIC dataset)
- Supports batch and single-image inference
"""

import torch
import numpy as np
import shap

class SHAP:
    def __init__(self, model, background_size=10):
        self.model = model
        self.model.eval()
        self.background_size = background_size

    def predict_fn(self, x_numpy):
        # convert numpy to torch tensor
        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = self.model(x_tensor)
            # multitask output
            if isinstance(output, tuple):
                _, cls_out = output
            else:
                cls_out = output

            probs = torch.softmax(cls_out, dim=1)
        return probs.detach().cpu().numpy()

    def __call__(self, image):
        # image shape: [1, C, H, W]
        device = next(self.model.parameters()).device
        image = image.to(device)

        # create simple background
        background = torch.zeros((self.background_size, image.shape[1], image.shape[2], image.shape[3]), device=device)

        background = background.detach()
        image_np = image.detach()

        # SHAP explainer
        explainer = shap.GradientExplainer(self.model, background)
        shap_values = explainer.shap_values(image)
        """
        shap_values format
        classification: list[num_classes]
        each: [B,C,H,W]
        """

        # choose predicted class
        preds = self.predict_fn(image_np)
        pred_class = np.argmax(preds)
        shap_map = shap_values[pred_class][0]

        # average channels if RGB
        if shap_map.ndim == 3:
            shap_map = np.mean(np.abs(shap_map), axis=0)

        # normalize
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)

        return shap_map