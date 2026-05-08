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

from lib import utils, dataloaders, models, metrics, testers
from lib.explainability.gradcam import GradCam
from lib.explainability.shap import SHAP
from lib.explainability.lime import LIME

params_3D_CBCT_Tooth = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [0.5, 0.5, 0.5],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 2048,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 1,
    "classes": 2,
    "index_to_class_dict":
        {
            0: "background",
            1: "foreground"
        },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "ReduceLROnPlateau",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["HD", "ASSD", "IoU", "SO", "DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0.60,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_MMOTU = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.12097393901893663,
    "color_jitter": 0.4203933474361258,
    "random_rotation_angle": 30,
    "normalize_means": (0.22250386, 0.21844882, 0.21521868),
    "normalize_stds": (0.21923075, 0.21622984, 0.21370508),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "MMOTU",
    "dataset_path": r"./datasets/MMOTU",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict":
        {
            0: "background",
            1: "foreground"
        },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.01,
    "weight_decay": 0.00001,
    "momentum": 0.7725414416309884,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8689275449032848,
    "step_size": 5,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 200,
    "T_0": 10,
    "T_mult": 5,
    "mode": "max",
    "patience": 1,
    "factor": 0.97,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.2350689696563569, 1 - 0.2350689696563569],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 2000,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 500,
}

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
    parser.add_argument("--pretrain_weight", type=str, default="pretrain/LiSANet-1000-K1.pth", help="pre-trained weight file path")
    # loading two pretrain for multitask, todo: combine path next time
    parser.add_argument("--pretrain_weight_seg", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--pretrain_weight_cls", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--dimension", type=str, default="2d", help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="BASIC", help="scaling version of PMFSNet")
    parser.add_argument("--task", type=str, default="multitask", choices=["segmentation", "classification", "multitask"], help="which task to perform")
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

def tensor_to_image(tensor):
    image = tensor.detach().cpu().squeeze()
    if image.ndim == 3:
        image = image.permute(1, 2, 0)

    image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def segmentation_inference(model, image, gt_mask=None):
    model.eval()

    with torch.no_grad():
        seg_out = model(image)
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
    plt.show()

def classification_inference(model, image, gt_label=None):
    model.eval()

    with torch.no_grad():
        cls_out = model(image)
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
    plt.show()

def multitask_inference(model_seg, model_cls, image, gt_mask=None, gt_label=None):
    model_seg.eval()
    model_cls.eval()

    image.requires_grad = True
    seg_out = model_seg(image)
    cls_out = model_cls(image)
    if isinstance(seg_out, dict):
        seg_out = seg_out["segmentation"]
    if isinstance(cls_out, dict):
        cls_out = cls_out["classification"]

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

    # GRAD-CAM
    target_layer = model_cls.down_convs[-1]
    gradcam = GradCam(model_cls, target_layer)
    cam = gradcam(model_cls, image, pred_class)
    cam = cam.detach().cpu().squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

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
    model_cls.to(device)
    forward_fn = ForwardEx(model_cls, device)
    forward_fn.eval()
    input_tensor = image.detach()
    
    # SHAP
    try:
        shap = SHAP(forward_fn)
        shap_map = shap(input_tensor)
        if len(shap_map.shape) == 3:
            shap_map = np.mean(shap_map, axis=0)

    except Exception as e:
        print(f"SHAP failed: {e}")
        shap_map = np.zeros((224, 224))

    # LIME
    try:
        lime = LIME(forward_fn)
        lime_map = lime(input_tensor)
        if len(lime_map.shape) == 3:
            lime_map = np.mean(lime_map, axis=0)

    except Exception as e:
        print(f"LIME failed: {e}")
        lime_map = np.zeros((224, 224))

    # output results

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    # title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, f"Prediction: {class_names[pred_class]} — {probs[pred_class].item()*100:.2f}%", ha="center", va="center", fontsize=20, fontweight="bold")

    # GT Class (text)
    ax_gt_text = fig.add_subplot(gs[1, 0])
    ax_gt_text.axis("off")
    if gt_label is not None:
        gt_text = f"GT Class:\n{class_names[gt_label]}"
    else:
        gt_text = "GT Class:\nN/A"
    ax_gt_text.text(0, 0.5, gt_text, fontsize=12, va="center")

    # Input Image
    ax_img = fig.add_subplot(gs[1, 1])
    ax_img.imshow(tensor_to_image(image))
    ax_img.set_title("Input")
    ax_img.axis("off")

    # GT Mask
    ax_gt = fig.add_subplot(gs[1, 2])
    if gt_mask is not None:
        ax_gt.imshow(gt_mask.squeeze().cpu(), cmap="gray")
        ax_gt.set_title("GT Mask")
    else:
        ax_gt.imshow(seg_mask, cmap="gray")
        ax_gt.set_title("GT Mask (N/A)")
    ax_gt.axis("off")

    # Pred Mask
    ax_pred = fig.add_subplot(gs[1, 3])
    ax_pred.imshow(seg_mask, cmap="gray")
    ax_pred.set_title("Pred Mask")
    ax_pred.axis("off")

    # Probabilities
    ax_prob = fig.add_subplot(gs[2, 0])
    ax_prob.axis("off")
    text = "Probabilities\n\n"
    for i in sorted_indices:
        i = i.item()
        text += f"{class_names[i]}: {probs[i].item()*100:.2f}%\n"
    ax_prob.text(0, 1, text, va="top", fontsize=10)

    # GradCAM
    ax_cam = fig.add_subplot(gs[2, 1])
    ax_cam.imshow(cam, cmap="jet")
    ax_cam.set_title("Grad-CAM")
    ax_cam.axis("off")

    # SHAP
    ax_shap = fig.add_subplot(gs[2, 2])
    ax_shap.imshow(shap_map, cmap="jet")
    ax_shap.set_title("SHAP")
    ax_shap.axis("off")

    # LIME
    ax_lime = fig.add_subplot(gs[2, 3])
    ax_lime.imshow(lime_map, cmap="jet")
    ax_lime.set_title("LIME")
    ax_lime.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    # analyse console arguments
    args = parse_args()

    # select the dictionary of hyperparameters used for training
    if args.dataset == "3D-CBCT-Tooth":
        params = params_3D_CBCT_Tooth
    elif args.dataset == "MMOTU":
        params = params_MMOTU
    elif args.dataset == "ISIC-2018":
        params = params_ISIC_2018
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    # update the dictionary of hyperparameters used for training
    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", ("NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["model_name"] = args.model
    if args.pretrain_weight is None and (args.pretrain_weight_seg is None and args.pretrain_weight_cls is None):
        raise RuntimeError("model weights cannot be None")
    if args.task == "multitask":
        params["pretrain_seg"] = args.pretrain_weight_seg
        params["pretrain_cls"] = args.pretrain_weight_cls
    else:
        params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
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
    if args.task == "multitask":
        model_seg = models.get_model(params)
        model_cls = models.get_model(params)
        model_seg.load_state_dict(torch.load(args.pretrain_weight_seg, map_location=params["device"]))
        model_cls.load_state_dict(torch.load(args.pretrain_weight_cls, map_location=params["device"]))
    else:
        model = models.get_model(params)
        model.load_state_dict(torch.load(args.pretrain_weight, map_location=params["device"]))
    print("Complete the initialization of model:{}".format(params["model_name"]))

    # Detect model task 
    is_seg = args.task in ["segmentation", "multitask"]
    is_cls = args.task in ["classification", "multitask"]
 
    print(f"Segmentation: {is_seg}, Classification: {is_cls}")
 
    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # load training weights
    if args.task == "multitask":
        model_seg.to(params["device"]).eval()
        model_cls.to(params["device"]).eval()
    else:
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
            mask_path = os.path.join(params["dataset_path"], args.task, "test", "masks", mask_name)

            if os.path.exists(mask_path):
                gt_mask = load_image(mask_path, params)
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask[0]
                gt_mask = gt_mask.unsqueeze(0)
            else:
                print(f"Ground truth mask not found: {mask_path}")
        
        if args.task in ["classification", "multitask"]:
            labels_path = os.path.join(params["dataset_path"], args.task, "test", "labels.csv")

            if os.path.exists(labels_path):
                labels_df = pd.read_csv(labels_path)
                image_name = os.path.basename(image_path)
                row = labels_df[labels_df["image"] == image_name]
                if len(row) > 0:
                    gt_label = int(row.iloc[0]["label"])

        # task selection
        if args.task == "segmentation":
            segmentation_inference(model, image, gt_mask)
        elif args.task == "classification":
            classification_inference(model, image, gt_label)
        else:
            multitask_inference(model_seg, model_cls, image, gt_mask, gt_label)

if __name__ == '__main__':
    main()