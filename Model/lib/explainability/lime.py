"""
LIME implementation adapted from https://github.com/marcotcr/lime
Paper: Ribeiro et al., "Why Should I Trust You?": Explaining the Predictions of Any Classifier, KDD 2016

Notes:
- Adapted for multitask model (segmentation + classification)
- Adjusted for medical imaging (ISIC dataset)
- Supports batch and single-image inference
"""

import torch
import numpy as np

from lime import lime_image
from skimage.segmentation import mark_boundaries


class LIME:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_fn(self, images):
        # convert numpy to torch tensor
        device = next(self.model.parameters()).device
        # convert to torch
        images = torch.tensor(images, dtype=torch.float32, device=device)
        # NHWC -> NCHW
        images = images.permute(0, 3, 1, 2)
        images = images.to(device)

        with torch.no_grad():
            output = self.model(images)
            # multitask support
            if isinstance(output, tuple):
                _, cls_out = output
            else:
                cls_out = output
            probs = torch.softmax(cls_out, dim=1)
        return probs.detach().cpu().numpy()

    def __call__(self, image):
        # image: torch tensor [1, C, H, W]
        device = next(self.model.parameters()).device
        image = image.to(device)

        # convert tensor to numpy
        img = image.detach().cpu().squeeze().numpy()
        # C,H,W -> H,W,C
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        # grayscale safety
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        # normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img.astype(np.double),
            self.predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        # predicted class
        pred_class = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        lime_map = mark_boundaries(temp, mask)
        # convert to grayscale heatmap
        if lime_map.ndim == 3:
            lime_map = np.mean(lime_map, axis=-1)
        # normalize
        lime_map = (lime_map - lime_map.min()) / (lime_map.max() - lime_map.min() + 1e-8)

        return lime_map