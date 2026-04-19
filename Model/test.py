# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 17:05
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import argparse

try:
    import nni
except ImportError:
    nni = None
    
import torch

from lib import utils, dataloaders, models, metrics, testers

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
            0: "MEL",
            1: "NV",
            2: "BCC",
            3: "AKIEC",
            4: "BKL",
            5: "DF",
            6: "VASC"
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
    "metric_names": { 
        "segmentation": ["DSC", "IoU", "JI", "ACC_SEG"],
        "classification": ["ACC_CLS", "AUC_ROC", "F1_MACRO"]
    },
    "loss_function_name": "DiceLoss",
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
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ISIC-2018", help="dataset name")
    parser.add_argument("--model", type=str, default="PMFSNet", help="model name")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--dimension", type=str, default="3d", help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="TINY", help="scaling version of PMFSNet")
    parser.add_argument("--task", type=str, default="multitask", choices=["segmentation", "classification", "multitask"], help="which task to perform"
        )
    args = parser.parse_args()

    if args.dataset == "ISIC-2018":
        params = params_ISIC_2018
        # detect model tasks
        # set task mode from CLI
        if args.task == "segmentation":
            params["segmentation"] = True
            params["classification"] = False
            params["metric_names"] = ["ACC_SEG", "DSC", "IoU", "JI"]
            params["seg_classes"] = 2 
            params["cls_classes"] = None 

        elif args.task == "classification":
            params["segmentation"] = False
            params["classification"] = True
            params["metric_names"] = ["ACC_CLS", "AUC_ROC", "F1_MACRO"]
            params["seg_classes"] = None 
            params["cls_classes"] = 7 

        elif args.task == "multitask":
            params["segmentation"] = True
            params["classification"] = True
            params["metric_names"] = ["ACC_SEG", "DSC", "IoU", "JI", "ACC_CLS", "AUC_ROC", "F1_MACRO"]
            params["seg_classes"] = 2 
            params["cls_classes"] = 7 
            params["seg_guided_cls"] = True
    return args


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

    if args.dataset == "ISIC-2018":
        if args.task == "segmentation":
            params["metric_names"] = ["DSC", "IoU", "JI", "ACC_SEG"]
        elif args.task == "classification":
            params["metric_names"] = ["ACC_CLS", "AUC_ROC", "F1_MACRO"]
        else:  # multitask
            params["metric_names"] = ["DSC", "IoU", "JI", "ACC_SEG", "ACC_CLS", "AUC_ROC", "F1_MACRO"]

    # update the dictionary of hyperparameters used for training
    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", ("NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["model_name"] = args.model
    if args.pretrain_weight is None:
        raise RuntimeError("model weights cannot be None")
    params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version

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

    # initialize the dataloader
    valid_loader = dataloaders.get_test_dataloader(params)
    print("Complete the initialization of dataloader")

    # initialize the model
    model = models.get_model(params)
    print("Complete the initialization of model:{}".format(params["model_name"]))

    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # initialize the tester
    tester = testers.get_tester(params, model, metric)
    print("Complete the initialization of tester")

    # load training weights
    tester.load()
    print("Complete loading training weights")

    # evaluate valid set
    tester.evaluation(valid_loader)


if __name__ == '__main__':
    main()
