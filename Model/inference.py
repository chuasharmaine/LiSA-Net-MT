"""
@author   :   andredalwin + chuasharmaine 
@Contact  :   sharmainechua134@gmail.com
@DateTime :   2026/05/08
@Version  :   2.0
"""

import os
import argparse
from glob import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd
import cv2
import lib.transforms.two as my_transforms
import matplotlib.gridspec as gridspec
import datetime
import csv
import math

from lib import utils, dataloaders, models, metrics, testers
from lib.explainability.gradcam import GradCam
from lib.explainability.lime import LIME
from lib.explainability.opticam import OptiCam

params_ISIC_2018 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.1,
    "color_jitter": 0.37,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "ISIC-2018",
    "dataset_path": r"./datasets/ISIC-2018",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "seg_classes": 2,
    "cls_classes": 7,
    "segmentation": True, 
    "classification": True, 
    "index_to_class_dict":
    {
        0: "MEL", # Melanoma
        1: "NV", # Melanocytic nevus
        2: "BCC", # Basal cell carcinoma
        3: "AKIEC", # Actinic keratosis
        4: "BKL", # Benign keratosis
        5: "DF", # Dermatofibroma
        6: "VASC"  # Vascular lesion
    },
    "index_to_class_name_dict":
    {
        0: "Melanoma",
        1: "Melanocytic Nevus",
        2: "Basal Cell Carcinoma",
        3: "Actinic Keratosis",
        4: "Benign Keratosis",
        5: "Dermatofibroma",
        6: "Vascular Lesion"

    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9582311026945434,
    "step_size": 20,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 100,
    "T_0": 5,
    "T_mult": 5,
    "mode": "max",
    "patience": 20,
    "factor": 0.3,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["ACC_SEG", "DSC", "IoU", "JI", "ACC_CLS", "AUC_ROC", "F1_MACRO"],
    "loss_function_name": {
        "segmentation": "DiceLoss",
        "classification": "CrossEntropyLoss"
    },
    "class_weight": [0.029, 1 - 0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 150,
    "best_metric": 0,
    "terminal_show_freq": 20,
    "save_epoch_freq": 50,
    "seg_weight": 0.5,
    "cls_weight": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ISIC-2018", help="dataset name")
    parser.add_argument("--model", type=str, default="LiSANetMT", help="model name")
    parser.add_argument("--pretrain_weight", type=str, required=True, help="complete model checkpoint (.pth or .state)")
    parser.add_argument("--dimension", type=str, default="2d", help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="BASIC", help="scaling version of PMFSNet")
    parser.add_argument("--task", type=str, default="multitask", choices=["segmentation", "classification", "multitask"], help="which task to perform")
    parser.add_argument("--cls_head_variant", type=str, default="baseline", choices=["baseline", "larger_mlp", "avgmax", "multiscale", "multiscale_larger", "projected_fusion", "projected_fusion_se", "lesion_fusion"], help="classification head used by the LiSANetMT checkpoint")
    parser.add_argument("--seg_guided_cls", action="store_true", help="enable only for a checkpoint trained with segmentation-guided classification")
    parser.add_argument("--opticam_steps", type=int, default=100, help="Opti-CAM optimization iterations")
    parser.add_argument("--lime_style", choices=["original", "signed"], default="original", help="LIME visualization style")
    parser.add_argument("--lime_regions", type=int, default=0, help="number of strongest LIME regions displayed; 0 adaptively selects 5-25 regions covering 90%% of explanation strength")
    parser.add_argument(
        "--show_zoom", "--show_zoom_row",
        dest="show_zoom_row",
        action="store_true",
        help="add an optional third row zoomed using the predicted segmentation region",
    )
    parser.add_argument("--output_tag", type=str, default=None, help="name used in saved output files (prevents variant outputs from overwriting each other)")
    parser.add_argument("--qualitative_only", action="store_true", help="save prediction masks, overlays, and labels without running explainability methods")
    parser.add_argument("--image_path", type=str, default=None, help="path of single inferred image")
    parser.add_argument("--images_dir", type=str, default=None, help="directory containing images for batch inference")
    return parser.parse_args()

def load_image(image_path, params):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # match dataset transform
    tfm = my_transforms.Compose([my_transforms.Resize(params["resize_shape"]), my_transforms.ToTensor(), my_transforms.Normalize(mean=params["normalize_means"], std=params["normalize_stds"])])
    image = tfm(image, np.zeros_like(image[:, :, 0]))[0]  # ignore mask
    return image

def load_mask(mask_path, resize_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = cv2.resize(mask, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((mask > 0).astype(np.uint8)).unsqueeze(0)

def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint

def get_gradcam_target_layer(model):
    """Return the last shared spatial feature layer used for classification."""
    # LiSANetMT classifies the output of cls_se after spatial feature fusion.
    if hasattr(model, "cls_se") and model.cls_se is not None:
        return model.cls_se, "cls_se"
    if hasattr(model, "down_convs") and len(model.down_convs) > 0:
        return model.down_convs[-1], "down_convs[-1]"
    if hasattr(model, "bottleneck"):
        return model.bottleneck, "bottleneck"
    if hasattr(model, "encoder"):
        for index in range(len(model.encoder) - 1, -1, -1):
            layer = model.encoder[index]
            if isinstance(layer, nn.Conv2d):
                return layer, f"encoder[{index}]"

    candidates = [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Conv2d) and "seg" not in name.lower()
    ]
    if candidates:
        name, layer = candidates[-1]
        return layer, name
    raise RuntimeError(
        f"No spatial classification layer found for Grad-CAM on {type(model).__name__}."
    )

def tensor_to_image(tensor):
    image = tensor.detach().cpu().squeeze()
    if image.ndim == 3:
        image = image.permute(1, 2, 0)

    image = image.numpy()
    means = np.asarray(params_ISIC_2018["normalize_means"], dtype=np.float32)
    stds = np.asarray(params_ISIC_2018["normalize_stds"], dtype=np.float32)
    return np.clip(image * stds + means, 0.0, 1.0)

# This is a working implementation of the ABCD rule of dermoscopy.
# this is not a clinical tool and should not be used for diagnosis
def lesion_morphology_summary(image, mask):
    """Calculate deterministic, non-clinical morphology indicators.

    Measurements use only the predicted lesion mask and displayed RGB image.
    Thresholds provide readable heuristic labels; they are not validated ABCD
    thresholds and must not be interpreted as diagnostic criteria.
    """
    image = np.asarray(image, dtype=np.float32)
    binary = (np.asarray(mask).squeeze() > 0).astype(np.uint8)
    height, width = binary.shape

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return {
            "available": False,
            "asymmetry_score": np.nan,
            "border_irregularity": np.nan,
            "color_spread": np.nan,
            "relative_area_percent": 0.0,
            "equivalent_diameter_pixels": 0.0,
            "relative_diameter_percent": 0.0,
            "asymmetry_label": "Not available",
            "border_label": "Not available",
            "color_label": "Not available",
        }

    # Ignore isolated segmentation noise and measure the largest lesion region.
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    lesion = (labels == component).astype(np.uint8)
    x, y, box_width, box_height, area = stats[component]
    crop = lesion[y:y + box_height, x:x + box_width].astype(bool)

    horizontal_union = np.logical_or(crop, np.fliplr(crop)).sum()
    vertical_union = np.logical_or(crop, np.flipud(crop)).sum()
    horizontal_difference = np.logical_xor(crop, np.fliplr(crop)).sum()
    vertical_difference = np.logical_xor(crop, np.flipud(crop)).sum()
    asymmetry = 0.5 * (
        horizontal_difference / max(horizontal_union, 1)
        + vertical_difference / max(vertical_union, 1)
    )

    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = max((cv2.arcLength(contour, True) for contour in contours), default=0.0)
    irregularity = (perimeter ** 2) / max(4.0 * math.pi * float(area), 1e-8)

    rgb_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    lesion_colors = lab[lesion.astype(bool)].astype(np.float32)
    if len(lesion_colors):
        q10, q90 = np.percentile(lesion_colors, [10, 90], axis=0)
        color_spread = float(np.mean((q90 - q10) / 255.0))
    else:
        color_spread = 0.0

    relative_area = 100.0 * float(area) / float(height * width)
    equivalent_diameter = math.sqrt(4.0 * float(area) / math.pi)
    relative_diameter = 100.0 * equivalent_diameter / math.hypot(height, width)

    def level(value, low, high):
        if value < low:
            return "Low"
        if value < high:
            return "Moderate"
        return "High"

    return {
        "available": True,
        "asymmetry_score": float(asymmetry),
        "border_irregularity": float(irregularity),
        "color_spread": color_spread,
        "relative_area_percent": relative_area,
        "equivalent_diameter_pixels": equivalent_diameter,
        "relative_diameter_percent": relative_diameter,
        "asymmetry_label": level(asymmetry, 0.15, 0.30),
        "border_label": level(irregularity, 1.20, 1.50),
        "color_label": level(color_spread, 0.06, 0.12),
    }

# saves the summary to a CSV file in the inference_outputs/morphology_summaries directory
def save_morphology_summary(summary, image_stem, display_name):
    save_dir = os.path.join("inference_outputs", "morphology_summaries")
    os.makedirs(save_dir, exist_ok=True)
    safe_name = display_name.replace(os.sep, "_").replace(" ", "_")
    path = os.path.join(save_dir, f"{safe_name}_{image_stem}_morphology.csv")
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["measurement", "value"])
        writer.writerows(summary.items())
    return path

# overlays the LIME scores on the original image, highlighting the strongest regions
def overlay_lime_scores(image, scores, alpha=0.7, num_regions=0, return_count=False):
    magnitude = np.abs(scores)
    scale = float(magnitude.max())
    if scale == 0:
        result = image.copy()
        return (result, 0) if return_count else result
    values = np.unique(scores[scores != 0])
    ranked = sorted(values, key=lambda value: abs(float(value)), reverse=True)
    if num_regions > 0:
        strongest = ranked[:num_regions]
    else:
        strengths = np.asarray([abs(float(value)) for value in ranked])
        cumulative = np.cumsum(strengths)
        target = 0.9 * cumulative[-1]
        adaptive_count = int(np.searchsorted(cumulative, target) + 1)
        adaptive_count = min(max(adaptive_count, 5), 25, len(ranked))
        strongest = ranked[:adaptive_count]
    selected = np.isin(scores, strongest)
    opacity = (alpha * selected)[..., None]
    colors = np.zeros_like(image)
    colors[scores > 0] = [0.0, 0.25, 1.0]
    colors[scores < 0] = [1.0, 0.1, 0.0]
    overlay = (1 - opacity) * image + opacity * colors
    result = np.clip(overlay, 0.0, 1.0)
    return (result, len(strongest)) if return_count else result

def overlay_heatmap(image, heatmap, alpha=0.6, cmap="jet"):
    """
    image: HxWx3 (0–1 or 0–255)
    heatmap: HxW (0–1)
    """
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def overlay_selected_heatmap(image, heatmap, alpha=0.55):
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    heatmap = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colour = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    result = img.copy()
    selected = heatmap > 0
    result[selected] = cv2.addWeighted(img, 1 - alpha, colour, alpha, 0)[selected]
    return result

def segmentation_inference(model, image, image_path, gt_mask=None):
    model.eval()

    with torch.no_grad():
        seg_out = model(image)
        if isinstance(seg_out, dict):
            seg_out = seg_out["segmentation"]
        if isinstance(seg_out, tuple):
            seg_out = seg_out[0]
        seg_mask = torch.argmax(seg_out, dim=1)[0].detach().cpu()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(tensor_to_image(image))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    if gt_mask is not None:
        ax[1].imshow(gt_mask.squeeze().cpu(), cmap="gray")
        ax[1].set_title("Ground Truth")
    else:
        ax[1].imshow(seg_mask, cmap="gray")
        ax[1].set_title("Ground Truth Not Available")

    ax[1].axis("off")
    ax[2].imshow(seg_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")
    plt.tight_layout()
    save_dir = os.path.join("inference_outputs", "segmentation_masks")
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_save_path = os.path.join(save_dir, f"{img_name}_mask.png")
    mask_np = seg_mask.numpy().astype(np.uint8) * 255
    cv2.imwrite(mask_save_path, mask_np)
    print(f"Saved mask: {mask_save_path}")
    plt.show()

def classification_inference(model, image, image_path, gt_label=None):
    model.eval()

    with torch.no_grad():
        cls_out = model(image)
        if isinstance(cls_out, dict):
            cls_out = cls_out["classification"]
        if isinstance(cls_out, tuple):
            cls_out = cls_out[1]
        probs = F.softmax(cls_out, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    print("\nPrediction:")
    print(f"Predicted: {params_ISIC_2018['index_to_class_name_dict'][pred_class]} -> {probs[pred_class].item()*100:.2f}%")
    if gt_label is not None:
        print(f"Ground Truth: {params_ISIC_2018['index_to_class_name_dict'][gt_label]}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(tensor_to_image(image))
    ax.set_title("Original Image")
    ax.axis("off")
    plt.tight_layout()
    log_dir = os.path.join("inference_outputs", "classification_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "results.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    log_line = (
        f"[{timestamp}] "
        f"{img_name} | "
        f"GT: {gt_label if gt_label is not None else 'N/A'} | "
        f"PRED: {pred_class} | "
        f"CONF: {probs[pred_class].item()*100:.2f}%\n"
    )
    with open(log_path, "a") as f:
        f.write(log_line)
    print(f"Saved classification log: {log_path}")
    plt.show()

def save_prediction_panel(image, mask, probabilities, image_path, display_name, gt_label=None, gt_mask=None):
    class_names = params_ISIC_2018["index_to_class_name_dict"]
    predicted = int(probabilities.argmax())
    overlay = image.copy()
    lesion = np.asarray(mask, dtype=bool)
    overlay[lesion] = .65 * overlay[lesion] + .35 * np.array([0.0, 1.0, 0.0])
    panels = [(image, "Input image"), (overlay, "Predicted lesion area"), (mask, "Predicted mask")]
    if gt_mask is not None:
        panels.append((gt_mask.squeeze().cpu().numpy(), "Reference mask"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel, cmap="gray" if np.asarray(panel).ndim == 2 else None)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.suptitle(f"{display_name} · Skin lesion analysis", fontsize=16, fontweight="bold")
    caption = f"Model prediction: {class_names[predicted]}   |   Model score: {float(probabilities[predicted]):.1%}"
    if gt_label is not None:
        caption += f"\nReference label: {class_names[gt_label]}"
    fig.text(.5, .1, caption, ha="center", fontsize=12)
    fig.text(.5, .035, "Research model output · Prediction unverified; model score is not a clinical certainty", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, .16, 1, .93))
    destination = os.path.join("inference_outputs", "prediction_panels")
    os.makedirs(destination, exist_ok=True)
    filename = f"{display_name}_{os.path.splitext(os.path.basename(image_path))[0]}_result.png"
    path = os.path.join(destination, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction panel: {path}")
    return path


def save_focus_distribution(cam, opticam_map, reference_mask, probabilities,
                            image_stem, display_name, ground_truth_used):
    """Save raw maps and quantify saliency mass in lesion, border, and background."""
    mask = np.asarray(reference_mask, dtype=np.float32).squeeze() > 0.5
    height, width = mask.shape
    radius = max(3, int(round(min(height, width) * 0.03)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel).astype(bool)
    dilated = cv2.dilate(mask_u8, kernel).astype(bool)
    regions = {
        "Lesion interior": eroded,
        "Lesion border": dilated & ~eroded,
        "Background": ~dilated,
    }

    def normalized_map(values):
        values = cv2.resize(np.asarray(values, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        values = np.clip(values, 0, None)
        return values / (values.sum() + 1e-8)

    maps = {"Grad-CAM": normalized_map(cam), "Opti-CAM": normalized_map(opticam_map)}
    shares = {method: [float(values[region].sum()) for region in regions.values()]
              for method, values in maps.items()}

    save_dir = os.path.join("inference_outputs", "focus_distribution")
    os.makedirs(save_dir, exist_ok=True)
    base = f"{display_name}_{image_stem}"
    np.savez_compressed(
        os.path.join(save_dir, f"{base}_raw_maps.npz"),
        gradcam=maps["Grad-CAM"], opticam=maps["Opti-CAM"],
        reference_mask=mask.astype(np.uint8), probabilities=np.asarray(probabilities),
    )
    with open(os.path.join(save_dir, f"{base}_focus.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *regions.keys(), "reference"])
        for method, values in shares.items():
            writer.writerow([method, *values, ground_truth_used])

    colors = ["#2ca25f", "#fdae61", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    left = np.zeros(len(maps))
    for region_index, (region_name, color) in enumerate(zip(regions, colors)):
        values = np.array([shares[method][region_index] for method in maps])
        bars = ax.barh(list(maps), values, left=left, color=color, label=region_name)
        for bar, value, offset in zip(bars, values, left):
            if value >= .045:
                ax.text(offset + value / 2, bar.get_y() + bar.get_height() / 2,
                        f"{value:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if region_index == 0 else "#222")
        left += values
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("Share of total saliency")
    ax.set_title(f"Explanation localization — {display_name}\nReference: {ground_truth_used}", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -0.48), ncol=3, frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{base}_focus.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def explanation_localization_metrics(values, lesion_mask, top_percent=20.0):
    """Return localization DSC and Pointing Game against a reference lesion mask."""
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent must be greater than 0 and at most 100")
    lesion = np.asarray(lesion_mask).squeeze() > 0
    if lesion.ndim != 2 or not lesion.any():
        return {"localization_dsc": np.nan, "pointing_game": np.nan}
    explanation = cv2.resize(np.asarray(values, dtype=np.float32).squeeze(),
                             (lesion.shape[1], lesion.shape[0]), interpolation=cv2.INTER_LINEAR)
    explanation = np.nan_to_num(explanation, nan=0.0, posinf=0.0, neginf=0.0)
    positive = np.flatnonzero(explanation.ravel() > 0)
    if positive.size == 0:
        return {"localization_dsc": 0.0, "pointing_game": 0.0}
    keep_count = min(max(1, int(np.ceil(explanation.size * top_percent / 100.0))), positive.size)
    selected = positive[np.argpartition(explanation.ravel()[positive], -keep_count)[-keep_count:]]
    explanation_mask = np.zeros(explanation.size, dtype=bool)
    explanation_mask[selected] = True
    explanation_mask = explanation_mask.reshape(explanation.shape)
    intersection = np.logical_and(explanation_mask, lesion).sum(dtype=np.float64)
    denominator = explanation_mask.sum(dtype=np.float64) + lesion.sum(dtype=np.float64)
    peak_y, peak_x = np.unravel_index(np.argmax(explanation), explanation.shape)
    return {
        "localization_dsc": float(2 * intersection / denominator) if denominator else 0.0,
        "pointing_game": float(lesion[peak_y, peak_x]),
    }


def save_clinical_report(image, predicted_mask, probabilities, image_path, display_name,
                         gradcam_overlay, opticam_overlay, lime_overlay, opticam_frames,
                         opticam_steps, gt_mask=None, gt_label=None, show_zoom_row=False,
                         output_tag=None):
    """Save the standard clinician-facing report and its Opti-CAM animation."""
    class_names = params_ISIC_2018["index_to_class_name_dict"]
    malignant, benign = [0, 2, 3], [1, 4, 5, 6]
    predicted_class = int(np.argmax(probabilities))
    predicted_category = "MALIGNANT" if predicted_class in malignant else "BENIGN"
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    predicted_mask = np.asarray(predicted_mask).squeeze() > 0
    reference_mask = None if gt_mask is None else gt_mask.squeeze().cpu().numpy() > 0

    # Red = lesion missed by the prediction; green = excess predicted lesion.
    if reference_mask is not None:
        error_overlay = image.copy()
        true_positive = predicted_mask & reference_mask
        false_negative = reference_mask & ~predicted_mask
        false_positive = predicted_mask & ~reference_mask
        error_overlay[true_positive] = .55 * error_overlay[true_positive] + .45 * np.array([1., 1., 1.])
        error_overlay[false_negative] = .35 * error_overlay[false_negative] + .65 * np.array([1., 0., 0.])
        error_overlay[false_positive] = .35 * error_overlay[false_positive] + .65 * np.array([0., 1., 0.])
        error_title = "Segmentation error overlay"
    else:
        error_overlay = image.copy()
        error_overlay[predicted_mask] = .55 * error_overlay[predicted_mask] + .45 * np.array([0.10, .45, .85])
        error_title = "Predicted lesion overlay"

    fig = plt.figure(figsize=(18, 15 if show_zoom_row else 10), facecolor="white")
    if show_zoom_row:
        grid = gridspec.GridSpec(4, 3, figure=fig, width_ratios=[1.55, 1, 1],
                                 height_ratios=[1, .62, 1, 1], left=.035, right=.975,
                                 top=.90, bottom=.055, wspace=.10, hspace=.19)
    else:
        grid = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.85, 1, 1, 1],
                                 height_ratios=[1, 1.15], left=.035, right=.975,
                                 top=.86, bottom=.075, wspace=.10, hspace=.22)
    fig.suptitle(display_name, fontsize=20, fontweight="bold", color="#17365d", y=.985)
    fig.text(.5, .952, f"Image {image_stem}", ha="center", fontsize=9.5, color="#666")

    # Left side: input/ground truth, then concise diagnostic information and probabilities.
    source_grid = grid[0, 0].subgridspec(1, 2, wspace=.10)
    source_panels = [(image, "Input image", None)]
    if reference_mask is not None:
        source_panels.append((reference_mask, "Ground-truth mask", "gray"))
    else:
        source_panels.append((error_overlay, "Predicted lesion overlay", None))
    for slot, (panel, title, cmap) in enumerate(source_panels):
        axis = fig.add_subplot(source_grid[0, slot])
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title, fontsize=10, fontweight="semibold", color="#17365d")
        axis.axis("off")

    details_axis = fig.add_subplot(grid[1, 0] if show_zoom_row else grid[0, 0])
    details_axis.axis("off")
    reference_label = class_names[gt_label].upper() if gt_label is not None else "NOT AVAILABLE"
    details_text = (
        "Predicted class:\n"
        f"{class_names[predicted_class].upper()} ({predicted_category})\n"
        f"{probabilities[predicted_class] * 100:.2f}%\n\n"
        "Ground-truth label:\n"
        f"{reference_label}"
    )

    details_axis.text(
        .0, .98, details_text,
        ha="left",
        va="top",
        fontsize=11,
        color="#17365d",
        linespacing=1.35,
        fontweight="semibold"
    )

    chart_axis = fig.add_subplot(grid[2:, 0] if show_zoom_row else grid[1, 0])
    chart_axis.set_title("Class probabilities", fontsize=11, fontweight="semibold", color="#17365d", loc="left", pad=8)
    order = malignant + benign
    values = [float(probabilities[index]) for index in order]
    labels = [class_names[index] for index in order]
    colors = ["#b71c1c", "#d84343", "#ef6c5b", "#1565a8", "#2f80bd", "#57a0ce", "#79b8d8"]
    positions = np.arange(len(order))
    chart_axis.axhspan(-.5, 2.5, color="#fff1f1", zorder=0)
    chart_axis.axhspan(2.5, 6.5, color="#edf6fc", zorder=0)
    bars = chart_axis.barh(positions, values, color=colors, height=.62)
    chart_axis.set_yticks(positions, labels=[])
    chart_axis.invert_yaxis()
    chart_axis.set_xlim(-.72, 1.16)
    chart_axis.set_xticks([0, .25, .5, .75, 1], labels=["0", "25", "50", "75", "100%"], fontsize=8)
    chart_axis.grid(axis="x", color="#e8e8e8", linewidth=.7)
    chart_axis.set_axisbelow(True)
    chart_axis.tick_params(axis="y", length=0)
    for spine in chart_axis.spines.values():
        spine.set_visible(False)
    for position, bar, index, value, label in zip(positions, bars, order, values, labels):
        chart_axis.text(-.03, position, label, ha="right", va="center", fontsize=9)
        if index == predicted_class:
            bar.set_edgecolor("#111")
            bar.set_linewidth(1.8)
        chart_axis.text(min(value + .018, 1.07), bar.get_y() + bar.get_height() / 2,
                        f"{value:.1%}", va="center", fontsize=8.5)
    chart_axis.text(.98, .94, "MALIGNANT", transform=chart_axis.transAxes, ha="right", va="top",
                    fontsize=7.5, fontweight="bold", color="#a52222")
    chart_axis.text(.98, .49, "BENIGN", transform=chart_axis.transAxes, ha="right", va="top",
                    fontsize=7.5, fontweight="bold", color="#1565a8")

    # Right side: segmentation agreement, followed by original/zoomed explanations.
    top_panels = [(predicted_mask, "Predicted segmentation mask", "gray"), (error_overlay, error_title, None)]
    top_columns = (1, 2) if show_zoom_row else (1, 2)
    for column, (panel, title, cmap) in zip(top_columns, top_panels):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title, fontsize=10, fontweight="semibold", color="#17365d")
        axis.axis("off")
        if reference_mask is not None and column == 2:
            axis.text(.5, -.08, "Red: missed lesion   •   Green: excess prediction",
                      transform=axis.transAxes, ha="center", fontsize=7.5, color="#555")

    from lib.explainability.display import lesion_crop
    (x0, y0, x1, y1), has_lesion = lesion_crop(predicted_mask)
    zoom_title = "Zoomed lesion region" if has_lesion else "Full image (no predicted lesion)"
    explanation_panels = [
        (gradcam_overlay, "Grad-CAM overlay"),
        (opticam_overlay, f"Opti-CAM overlay — iteration {opticam_steps}/{opticam_steps}"),
        (lime_overlay, "LIME overlay"),
    ]
    opticam_artist = opticam_zoom_artist = opticam_axis = None
    if show_zoom_row:
        for row, (panel, title) in enumerate(explanation_panels, start=1):
            original_axis = fig.add_subplot(grid[row, 1])
            original_axis.imshow(panel)
            original_axis.set_box_aspect(1)
            original_axis.set_title(title, fontsize=10, fontweight="semibold", color="#17365d")
            original_axis.axis("off")
            if row == 2:
                opticam_artist, opticam_axis = original_axis.images[0], original_axis
            zoom_axis = fig.add_subplot(grid[row, 2])
            zoom_artist = zoom_axis.imshow(panel[y0:y1, x0:x1])
            zoom_axis.set_box_aspect(1)
            zoom_axis.set_title(f"{title.split(' — ')[0]} — {zoom_title}", fontsize=10,
                                fontweight="semibold", color="#17365d")
            if row == 2:
                opticam_zoom_artist = zoom_artist
            zoom_axis.axis("off")
    else:
        for column, (panel, title) in enumerate(explanation_panels, start=1):
            axis = fig.add_subplot(grid[1, column])
            artist = axis.imshow(panel)
            axis.set_box_aspect(1)
            axis.set_title(title, fontsize=10, fontweight="semibold", color="#17365d")
            axis.axis("off")
            if column == 2:
                opticam_artist, opticam_axis = artist, axis

    fig.text(.5, .016, "Explanation maps identify influential image regions; they do not establish clinical reasoning or a diagnosis.",
             ha="center", fontsize=7.5, color="#555")
    output_dir = os.path.join("inference_outputs", "clinical_reports", output_tag or display_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{image_stem}_clinical_report.png")
    fig.savefig(output_path, dpi=250, facecolor="white")
    if opticam_frames:
        from PIL import Image
        frames = []
        for iteration, mask in opticam_frames:
            frame_overlay = overlay_heatmap(image, mask, alpha=.52)
            opticam_artist.set_data(frame_overlay)
            opticam_axis.set_title(f"Opti-CAM overlay — iteration {iteration}/{opticam_steps}",
                                   fontsize=10, fontweight="semibold", color="#17365d")
            if opticam_zoom_artist is not None:
                opticam_zoom_artist.set_data(frame_overlay[y0:y1, x0:x1])
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=[500] * (len(frames) - 1) + [2000], loop=0)
        print(f"Saved animated inference display: {gif_path}")
    plt.close(fig)
    print(f"Saved clinical report: {output_path}")
    return output_path

def multitask_inference(model, image, image_path, model_name, gt_mask=None, gt_label=None, opticam_steps=100, lime_style="original", output_tag=None, qualitative_only=False, lime_regions=0, show_zoom_row=False):
    model.eval()

    image.requires_grad = True
    output = model(image)
    if isinstance(output, dict):
        seg_out = output["segmentation"]
        cls_out = output["classification"]
    elif isinstance(output, tuple) and len(output) >= 2:
        seg_out, cls_out = output[:2]
    else:
        raise RuntimeError("Multitask model must return segmentation and classification outputs.")

    probs = F.softmax(cls_out, dim=1)[0]
    pred_class = torch.argmax(cls_out).item()
    seg_mask = torch.argmax(seg_out, dim=1)[0].detach().cpu()

    class_names = params_ISIC_2018["index_to_class_name_dict"]
    # malignant / benign grouping
    malignant_classes = [0, 2, 3]
    benign_classes = [1, 4, 5, 6]

    # print prediction
    print("\nPrediction Class:")
    print(f"{class_names[pred_class]} — {probs[pred_class].item()*100:.2f}%")

    if gt_label is not None:
        print(f"Actual Class: {class_names[gt_label]}")

    # sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    print("\nBreakdown of Probabilities")

    print("\nMalignant")
    for idx in sorted_indices:
        idx = idx.item()
        if idx in malignant_classes:
            print(f"{idx+1}. {class_names[idx]} — {probs[idx].item()*100:.2f}%")

    print("\nBenign")
    for idx in sorted_indices:
        idx = idx.item()
        if idx in benign_classes:
            print(f"{idx+1}. {class_names[idx]} — {probs[idx].item()*100:.2f}%")

    img_np = tensor_to_image(image)
    display_name = output_tag or model_name
    figure_name = model_name
    if model_name == "BreastCancerMT":
        figure_name = "Aumente-Maestro et al. Framework"
    elif output_tag and "lisanetmt-lr" in output_tag.lower():
        figure_name = "LiSANetMT — LR Variant"
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    morphology = lesion_morphology_summary(img_np, seg_mask.numpy())
    morphology_path = save_morphology_summary(morphology, image_stem, display_name)
    print("\nAutomated lesion morphology summary (heuristic; not diagnostic)")
    if morphology["available"]:
        print(f"Asymmetry indicator: {morphology['asymmetry_label']} "
              f"(score {morphology['asymmetry_score']:.3f})")
        print(f"Border irregularity indicator: {morphology['border_label']} "
              f"(compactness index {morphology['border_irregularity']:.3f})")
        print(f"Color variation indicator: {morphology['color_label']} "
              f"(robust Lab spread {morphology['color_spread']:.3f})")
        print(f"Relative lesion area: {morphology['relative_area_percent']:.1f}%")
        print(f"Equivalent diameter: {morphology['equivalent_diameter_pixels']:.1f} px "
              f"({morphology['relative_diameter_percent']:.1f}% of image diagonal)")
    else:
        print("No predicted lesion region was available for measurement.")
    print(f"Saved morphology measurements: {morphology_path}")
    if qualitative_only:
        save_prediction_panel(img_np, seg_mask.numpy(), probs.detach().cpu().numpy(),
                              image_path, display_name, gt_label, gt_mask)
        save_dir = os.path.join("inference_outputs", "qualitative", display_name)
        os.makedirs(save_dir, exist_ok=True)
        mask_np = (seg_mask.numpy().astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(save_dir, f"{image_stem}_mask.png"), mask_np)
        overlay = img_np.copy()
        lesion = seg_mask.numpy().astype(bool)
        overlay[lesion] = 0.55 * overlay[lesion] + 0.45 * np.array([0.0, 1.0, 0.0])
        cv2.imwrite(
            os.path.join(save_dir, f"{image_stem}_overlay.png"),
            cv2.cvtColor((np.clip(overlay, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        csv_path = os.path.join(save_dir, "predictions.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["image", "predicted_class", "confidence", "ground_truth_class", *class_names.values()])
            writer.writerow([
                image_stem,
                class_names[pred_class],
                probs[pred_class].item(),
                class_names[gt_label] if gt_label is not None else "N/A",
                *[prob.item() for prob in probs],
            ])
        print(f"Saved qualitative outputs to: {save_dir}")
        return

    # GRAD-CAM
    target_layer, target_layer_name = get_gradcam_target_layer(model)
    print(f"Grad-CAM target layer: {target_layer_name}")
    gradcam = GradCam(model, target_layer)
    cam = gradcam(model, image, pred_class)
    cam = cam.detach().cpu().squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam_overlay = overlay_heatmap(img_np, cam, alpha=0.6)
    gradcam.close()

    # Opti-CAM
    opticam = OptiCam(model, target_layer, steps=opticam_steps)
    opticam_frames = []
    def record_opticam(iteration, mask, logit):
        if iteration % 10 == 0 or iteration == opticam_steps:
            opticam_frames.append((iteration, mask.numpy().copy()))
    opticam_map = opticam(image.detach(), pred_class, progress_callback=record_opticam).cpu().numpy()
    opticam_overlay = overlay_heatmap(img_np, opticam_map, alpha=0.6)

    seg_overlay = img_np.copy()
    lesion = seg_mask.numpy().astype(bool)
    seg_overlay[lesion] = 0.55 * seg_overlay[lesion] + 0.45 * np.array([0.0, 1.0, 0.0])

    class ForwardEx(torch.nn.Module):
        def __init__(self, model, device):
            super().__init__()
            self.model = model
            self.device = device
            self.model.eval()

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            x = x.to(self.device)
            out = self.model(x)
            if isinstance(out, tuple):
                out = out[1]
            if isinstance(out, dict):
                out = out["classification"]
            return out
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    forward_fn = ForwardEx(model, device)
    forward_fn.eval()
    input_tensor = image.detach()
    
    # LIME is a required explainability output. Let failures surface instead
    # of silently displaying an all-zero image as if it were an explanation.
    lime = LIME(
        forward_fn,
        normalize_means=params_ISIC_2018["normalize_means"],
        normalize_stds=params_ISIC_2018["normalize_stds"],
        style=lime_style,
    )
    lime_map = lime(input_tensor, target_class=pred_class)
    print(f"LIME target class: {pred_class}; local surrogate R²: {lime.local_score:.4f}")
    if lime_style == "original":
        lime_overlay, displayed_lime_regions = overlay_lime_scores(
            img_np, lime_map, num_regions=lime_regions, return_count=True
        )
    else:
        lime_overlay = lime_map
        displayed_lime_regions = 0

    # Quantify where saliency falls. Ground truth is used only for retrospective
    # explanation assessment; it is never supplied to the model.
    if gt_mask is not None:
        focus_mask = gt_mask.squeeze().cpu().numpy()
        focus_reference = "ground-truth mask (evaluation only)"
    else:
        focus_mask = seg_mask.numpy()
        focus_reference = "predicted mask"
    save_focus_distribution(
        cam, opticam_map, focus_mask, probs.detach().cpu().numpy(),
        image_stem, display_name, focus_reference,
    )

    save_clinical_report(
        img_np, seg_mask.numpy(), probs.detach().cpu().numpy(), image_path,
        figure_name, cam_overlay, opticam_overlay, lime_overlay, opticam_frames,
        opticam_steps, gt_mask, gt_label, show_zoom_row, display_name,
    )
    return

def main():
    # analyse console arguments
    args = parse_args()

    # select the dictionary of hyperparameters used for training
    if args.dataset == "ISIC-2018":
        params = params_ISIC_2018
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    # update the dictionary of hyperparameters used for training
    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", ("NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["model_name"] = args.model
    params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
    params["cls_head_variant"] = args.cls_head_variant
    params["seg_guided_cls"] = args.seg_guided_cls
    params["task"] = args.task
    params["segmentation"] = args.task in ["segmentation", "multitask"]
    params["classification"] = args.task in ["classification", "multitask"]
    params["seg_classes"] = 2 if params["segmentation"] else None
    params["cls_classes"] = 7 if params["classification"] else None
    if args.image_path is None and args.images_dir is None:
        raise RuntimeError("Either image_path or images_dir must be provided")

    # launch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # get the cuda device
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("Complete the initialization of configuration")

    # initialize the model
    model = models.get_model(params)
    checkpoint = torch.load(args.pretrain_weight, map_location=params["device"], weights_only=False)
    model_state = extract_model_state(checkpoint)
    if not isinstance(model_state, dict):
        raise TypeError("Checkpoint must be a model state_dict or contain one under 'model' or 'state_dict'.")
    model.load_state_dict(model_state, strict=True)
    print("Complete the initialization of model:{}".format(params["model_name"]))

    # Detect model task 
    is_seg = args.task in ["segmentation", "multitask"]
    is_cls = args.task in ["classification", "multitask"]
 
    print(f"Segmentation: {is_seg}, Classification: {is_cls}")
 
    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # load training weights
    model.to(params["device"]).eval()
    print("Complete loading training weights")

    # prepare images for inference
    image_paths = []

    if args.image_path is not None:
        image_paths.append(args.image_path)
    elif args.images_dir is not None:
        image_paths = glob(os.path.join(args.images_dir, "*"))
    else:
        raise RuntimeError("Provide image_path or images_dir")


    # perform inference
    for image_path in image_paths:
        print(f"\nRunning inference on: {image_path}")
        image = load_image(image_path, params)
        image = image.unsqueeze(0).to(params["device"])
        gt_mask = None
        gt_label = None

        # load mask for segmentation / multitask
        if args.task in ["segmentation", "multitask"]:
            image_name = os.path.basename(image_path)
            image_stem = os.path.splitext(image_name)[0]
            mask_name = f"{image_stem}_segmentation.png"
            mask_candidates = [
                os.path.join(params["dataset_path"], "multitask", "test", "masks", mask_name),
                os.path.join(params["dataset_path"], "segmentation", "test", "masks", mask_name),
            ]
            mask_path = next((path for path in mask_candidates if os.path.exists(path)), None)

            if mask_path is not None:
                gt_mask = load_mask(mask_path, params["resize_shape"])
            else:
                print(f"Ground truth mask not found in: {mask_candidates}")
        
        if args.task in ["classification", "multitask"]:
            label_candidates = [
                os.path.join(params["dataset_path"], "multitask", "test", "labels.csv"),
                os.path.join(params["dataset_path"], "classification", "test", "labels.csv"),
            ]
            for labels_path in label_candidates:
                if not os.path.exists(labels_path):
                    continue
                labels_df = pd.read_csv(labels_path)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                row = labels_df[labels_df["image"] == image_name]

                if len(row) > 0:
                    class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
                    gt_vector = row.iloc[0][class_cols].values.astype(float)
                    gt_label = int(np.argmax(gt_vector))
                    break

        # task selection
        if args.task == "segmentation":
            segmentation_inference(model, image, image_path, gt_mask)
        elif args.task == "classification":
            classification_inference(model, image, image_path, gt_label)
        else:
            multitask_inference(
                model, image, image_path, args.model, gt_mask, gt_label,
                args.opticam_steps, args.lime_style, args.output_tag,
                args.qualitative_only, args.lime_regions, args.show_zoom_row,
            )

if __name__ == '__main__':
    main()
