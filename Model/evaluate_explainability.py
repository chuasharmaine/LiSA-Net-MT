"""
Batch evaluation of Grad-CAM and Opti-CAM lesion-localization metrics.

This script writes numeric results only.  It intentionally does not create
clinical reports, PNGs, GIFs, or LIME explanations, making it suitable for
evaluating a complete masked image folder.
"""

import argparse
import csv
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from inference import (
    explanation_localization_metrics,
    extract_model_state,
    get_gradcam_target_layer,
    load_image,
    load_mask,
    params_ISIC_2018,
)
from lib import models, utils
from lib.explainability.gradcam import GradCam
from lib.explainability.opticam import OptiCam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM and Opti-CAM alignment with available lesion masks."
    )
    parser.add_argument("--dataset", default="ISIC-2018")
    parser.add_argument("--model", default="LiSANetMT")
    parser.add_argument("--pretrain_weight", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_tag", default=None)
    parser.add_argument("--dimension", default="2d")
    parser.add_argument("--scaling_version", default="BASIC")
    parser.add_argument(
        "--cls_head_variant",
        default="baseline",
        choices=[
            "baseline", "larger_mlp", "avgmax", "multiscale",
            "multiscale_larger", "projected_fusion",
            "projected_fusion_se", "lesion_fusion",
        ],
    )
    parser.add_argument("--seg_guided_cls", action="store_true")
    parser.add_argument(
        "--top_percent", type=float, default=20.0,
        help="top positive saliency percentage used for localization DSC (default: 20)",
    )
    parser.add_argument(
        "--opticam_steps", type=int, default=100,
        help="fixed Opti-CAM optimization iterations used for this evaluation (default: 100)",
    )
    return parser.parse_args()


def classification_logits(output):
    if isinstance(output, dict):
        return output["classification"]
    if isinstance(output, tuple):
        return output[1]
    raise RuntimeError("Multitask model must return classification logits.")


def image_files(images_dir):
    allowed = {".jpg", ".jpeg", ".png"}
    return sorted(
        path for path in glob(os.path.join(images_dir, "*"))
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in allowed
    )


def find_ground_truth_mask(image_path, params):
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    mask_name = f"{image_stem}_segmentation.png"
    candidates = [
        os.path.join(params["dataset_path"], "multitask", "test", "masks", mask_name),
        os.path.join(params["dataset_path"], "segmentation", "test", "masks", mask_name),
    ]
    mask_path = next((path for path in candidates if os.path.exists(path)), None)
    return load_mask(mask_path, params["resize_shape"]) if mask_path else None


def main():
    args = parse_args()
    if args.dataset != "ISIC-2018":
        raise RuntimeError("Only ISIC-2018 is configured for this evaluator.")
    if not 0 < args.top_percent <= 100:
        raise ValueError("--top_percent must be greater than 0 and at most 100")

    params = params_ISIC_2018.copy()
    params.update({
        "dataset_name": args.dataset,
        "dataset_path": os.path.join("./datasets", args.dataset),
        "model_name": args.model,
        "pretrain": args.pretrain_weight,
        "dimension": args.dimension,
        "scaling_version": args.scaling_version,
        "cls_head_variant": args.cls_head_variant,
        "seg_guided_cls": args.seg_guided_cls,
        "segmentation": True,
        "classification": True,
        "seg_classes": 2,
        "cls_classes": 7,
    })
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])
    device = torch.device("cuda:0" if params["cuda"] and torch.cuda.is_available() else "cpu")
    params["device"] = device

    model = models.get_model(params)
    checkpoint = torch.load(args.pretrain_weight, map_location=device, weights_only=False)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.to(device).eval()

    paths = image_files(args.images_dir)
    if not paths:
        raise RuntimeError(f"No image files found in: {args.images_dir}")

    target_layer, target_layer_name = get_gradcam_target_layer(model)
    gradcam = GradCam(model, target_layer)
    opticam = OptiCam(model, target_layer, steps=args.opticam_steps)
    rows = []
    skipped = 0

    try:
        for number, image_path in enumerate(paths, start=1):
            gt_mask = find_ground_truth_mask(image_path, params)
            gt_label = None
            if gt_mask is None:
                skipped += 1
                continue

            image = load_image(image_path, params).unsqueeze(0).to(device)
            image.requires_grad_(True)
            with torch.no_grad():
                probabilities = F.softmax(classification_logits(model(image)), dim=1)[0]
            predicted_class = int(probabilities.argmax())

            gradcam_map = gradcam(model, image, predicted_class).cpu().numpy()
            opticam_map = opticam(image.detach(), predicted_class).cpu().numpy()
            image_stem = os.path.splitext(os.path.basename(image_path))[0]

            for method, explanation in (("Grad-CAM", gradcam_map), ("Opti-CAM", opticam_map)):
                metrics = explanation_localization_metrics(
                    explanation, gt_mask.squeeze().numpy(), args.top_percent
                )
                rows.append({
                    "image": image_stem,
                    "method": method,
                    "top_percent": args.top_percent,
                    "predicted_class": predicted_class,
                    "ground_truth_class": gt_label,
                    "prediction_correct": None if gt_label is None else int(predicted_class == gt_label),
                    "localization_dsc": metrics["localization_dsc"],
                    "pointing_game": metrics["pointing_game"],
                    "reference": "ground_truth_mask",
                })
            print(f"[{number}/{len(paths)}] {image_stem}")
    finally:
        gradcam.close()

    if not rows:
        raise RuntimeError("No matching ground-truth masks were found. Localization metrics require masks.")

    tag = args.output_tag or args.model
    output_dir = os.path.join("inference_outputs", "explainability_evaluation", tag)
    os.makedirs(output_dir, exist_ok=True)
    per_image_path = os.path.join(output_dir, "per_image_metrics.csv")
    pd.DataFrame(rows).to_csv(per_image_path, index=False)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method")[["localization_dsc", "pointing_game"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary.to_csv(summary_path)

    print(f"\nTarget layer: {target_layer_name}")
    print(f"Images evaluated: {results['image'].nunique()} | skipped without masks: {skipped}")
    print("\nExplainability localization summary")
    print(summary.to_string())
    print(f"\nSaved per-image metrics: {per_image_path}")
    print(f"Saved averaged metrics: {summary_path}")


if __name__ == "__main__":
    main()
