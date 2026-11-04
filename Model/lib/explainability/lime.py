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
from skimage.segmentation import find_boundaries, slic


class LIME:
    def __init__(self, model, normalize_means=None, normalize_stds=None, style="original", seed=1777777):
        self.model = model
        self.model.eval()
        self.normalize_means = np.asarray(normalize_means if normalize_means is not None else (0.0, 0.0, 0.0), dtype=np.float32,).reshape(1, 1, 1, -1)
        self.normalize_stds = np.asarray(normalize_stds if normalize_stds is not None else (1.0, 1.0, 1.0), dtype=np.float32,).reshape(1, 1, 1, -1)
        self.style = style
        self.seed = seed

    def predict_fn(self, images):
        # convert numpy to torch tensor
        device = next(self.model.parameters()).device
        # convert to torch
        images = np.asarray(images, dtype=np.float32)
        images = (images - self.normalize_means) / self.normalize_stds
        images = torch.tensor(images, dtype=torch.float32, device=device)
        # NHWC -> NCHW
        images = images.permute(0, 3, 1, 2)
        images = images.to(device)

        with torch.no_grad():
            output = self.model(images)
            # multitask support
            if isinstance(output, dict):
                cls_out = output["classification"]
            elif isinstance(output, tuple):
                _, cls_out = output
            else:
                cls_out = output
            probs = torch.softmax(cls_out, dim=1)
        return probs.detach().cpu().numpy()

    def __call__(self, image, target_class=None):
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
        # normalization
        means = self.normalize_means.reshape(-1)
        stds = self.normalize_stds.reshape(-1)
        img = np.clip(img * stds + means, 0.0, 1.0)
        if target_class is None:
            target_class = int(self.predict_fn(img[None])[0].argmax())
        explainer = lime_image.LimeImageExplainer(random_state=self.seed)
        explanation = explainer.explain_instance(
            img.astype(np.double),
            self.predict_fn,
            labels=(target_class,),
            top_labels=None,
            hide_color=None,
            num_samples=1000,
            random_seed=self.seed,
            segmentation_fn=lambda rgb: slic(rgb, n_segments=120, compactness=10,
                                              sigma=1, start_label=0, channel_axis=-1),
        )

        self.explanation = explanation
        self.target_class = target_class
        self.local_score = float(explanation.score)
        if self.style == "original":
            # Each superpixel receives its fitted LIME coefficient, not image brightness.
            weights = np.zeros(explanation.segments.shape, dtype=np.float32)
            for region, weight in explanation.local_exp[target_class]:
                weights[explanation.segments == region] = weight
            return weights

        _, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )

        overlay = img.copy()
        alpha = 0.15
        positive = mask > 0
        negative = mask < 0
        overlay[positive] = (1 - alpha) * overlay[positive] + alpha * np.array([0.0, 1.0, 0.0])
        overlay[negative] = (1 - alpha) * overlay[negative] + alpha * np.array([1.0, 0.0, 0.0])
        overlay[find_boundaries(positive, mode="outer")] = np.array([0.0, 1.0, 0.0])
        overlay[find_boundaries(negative, mode="outer")] = np.array([1.0, 0.0, 0.0])
        return np.clip(overlay, 0.0, 1.0)