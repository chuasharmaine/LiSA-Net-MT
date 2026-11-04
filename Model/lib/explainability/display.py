"""
Display functions for explainability outputs.
Includes normal and zoomed views of Grad-CAM, Opti-CAM, and LIME explanations, with optional ground-truth mask overlay.
The zoomed view crops to the predicted lesion region, with a small margin of context. If no lesion is predicted, the full image is shown in both rows.
The figures are saved to the inference_outputs/explainability directory.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def lesion_crop(mask, padding=0.5, minimum_size=48):
    """Include all foreground components, with half a lesion-width of context."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return (0, 0, width, height), False
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    side = min(max(width, height), max(minimum_size, int(np.ceil(max(right-left, bottom-top) * (1+2*padding)))))
    cx, cy = (left+right)/2, (top+bottom)/2
    crop_width, crop_height = min(width, side), min(height, side)
    x0 = max(0, min(width-crop_width, int(np.floor(cx-crop_width/2))))
    y0 = max(0, min(height-crop_height, int(np.floor(cy-crop_height/2))))
    return (x0, y0, x0+crop_width, y0+crop_height), True


def save_zoom_explanations(image, mask, gradcam, opticam, lime_overlay, prediction, path,
                           ground_truth=None, reference_label=None, model_name="LiSA-Net-MT", lime_regions=5):
    image = np.asarray(image)
    mask = np.asarray(mask, dtype=bool)
    box, has_lesion = lesion_crop(mask)
    x0, y0, x1, y1 = box
    cmap = plt.get_cmap("jet")

    def cam_overlay(scores):
        scores = np.clip(np.asarray(scores), 0, 1)
        return .4*image + .6*cmap(scores)[..., :3]

    panels = [image, cam_overlay(gradcam), cam_overlay(opticam), lime_overlay]
    titles = ["Input", "Grad-CAM", "Opti-CAM", "LIME"]
    fig, all_axes = plt.subplots(3, 4, figsize=(16, 13))
    axes = all_axes[1:]
    header = all_axes[0]
    header[0].imshow(image)
    header[0].set_title("Input image", fontsize=14)
    header[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    header[1].set_title("Predicted segmentation mask", fontsize=14)
    if ground_truth is not None:
        truth = np.asarray(ground_truth).squeeze()
        header[2].imshow(truth, cmap="gray", vmin=0, vmax=1)
        header[2].set_title("Ground-truth mask", fontsize=14)
    else:
        overlay = image.copy()
        overlay[mask] = .65 * overlay[mask] + .35 * np.array([0, 1, 0])
        header[2].imshow(overlay)
        header[2].set_title("Predicted lesion overlay", fontsize=14)
    header[3].text(0, .9, "PREDICTION", transform=header[3].transAxes, fontsize=13, weight="bold", va="top")
    header[3].text(0, .75, prediction.replace(" | ", "\n"), transform=header[3].transAxes, fontsize=12, va="top", linespacing=1.7)
    if reference_label is not None:
        header[3].text(0, .35, f"GROUND-TRUTH CLASS\n{reference_label}", transform=header[3].transAxes, fontsize=12, va="top", linespacing=1.7)
    header[3].text(0, .09, "Model score is not clinical certainty.", transform=header[3].transAxes, fontsize=9, color="#555")
    for ax in header:
        ax.axis("off")
    for col, (panel, title) in enumerate(zip(panels, titles)):
        axes[0, col].imshow(panel, interpolation="nearest")
        axes[0, col].set_title(title, fontsize=14, pad=12)
        if col == 0:
            axes[0, col].add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False, edgecolor="#00e5ff", linewidth=1.7))
        axes[1, col].imshow(panel[y0:y1, x0:x1], interpolation="nearest")
        full_crop = (x0, y0, x1, y1) == (0, 0, image.shape[1], image.shape[0])
        axes[1, col].set_title("Full extent retained (large lesion)" if full_crop else "Same region enlarged", fontsize=12, pad=10)
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(f"{model_name} | Prediction, ground truth and explanations", fontsize=19, weight="bold", y=.975)
    fig.subplots_adjust(left=.025, right=.975, top=.92, bottom=.15, wspace=.15, hspace=.28)
    for col in (1, 2):
        position = axes[1, col].get_position()
        color_ax = fig.add_axes([position.x0+.025, .115, position.width-.05, .009])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=color_ax, orientation="horizontal", ticks=[0, 1])
        cb.ax.set_xticklabels(["Lower saliency", "Higher saliency"])
        cb.ax.tick_params(labelsize=9, length=0)
    region_note = f"Strongest {lime_regions} regions shown." if lime_regions > 0 else "All scored regions shown."
    fig.text(.5, .06, "Cyan box on input: zoom area\nLIME: blue supports the predicted class; red opposes it. " + region_note, ha="center", fontsize=11, linespacing=1.5)
    note = "Maps computed on the full image; zoom only enlarges their display. Highlights do not establish clinical border irregularity."
    if not has_lesion:
        note = "No predicted lesion found: full image shown in both rows. " + note
    fig.text(.5, .025, note, ha="center", fontsize=9, color="#555")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)