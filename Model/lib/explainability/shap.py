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
    def __init__(self, model, background_images=None):
        self.model = model
        self.model.eval()
        self.background_images = background_images

    def disable_inplace(self, model):
        for module in model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False

    def __call__(self, image):
        # image shape: [1, C, H, W]
        device = next(self.model.parameters()).device
        input_tensor = image.to(device).detach().clone()

        # create simple background
        if self.background_images is None:
            background = torch.zeros_like(image).repeat(10,1,1,1) * 0.5
        else:
            background = self.background_images.to(device)

        # SHAP explainer
        try:
            self.disable_inplace(self.model)
            explainer = shap.GradientExplainer(self.model, background)
            shap_values = explainer.shap_values(image)
            """
            shap_values format
            classification: list[num_classes]
            each: [B,C,H,W]
            """

            # choose predicted class
            with torch.no_grad():
                preds = self.model(image)
                if isinstance(preds, tuple):
                    preds = preds[1]
                probs = torch.softmax(preds, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
            if isinstance(shap_values, list):
                shap_map = shap_values[pred_class][0]
            else:
                shap_map = shap_values[0]

            if len(shap_map.shape) == 4:
                shap_map = shap_map[0]

            shap_map = np.sum(shap_map, axis=0)

            # normalize
            v_min, v_max = np.percentile(shap_map, [1,99])
            shap_map = np.clip(shap_map,v_min,v_max)
            shap_map = (shap_map - v_min) / (v_max - v_min + 1e-8)

            return shap_map
        
        except Exception as e:
            print(f"shap actual error: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))


