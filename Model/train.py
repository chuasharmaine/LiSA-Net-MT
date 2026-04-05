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

import torch

from lib import utils, dataloaders, models, losses, metrics, trainers


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
    # for testing on CPU
    # "batch_size": 2,
    # "num_workers": 0,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "seg_classes": 2,
    "cls_classes": 7,
    "segmentation": True, 
    "classification": True, 
    "index_to_class_dict":
    {
        0: "Melanoma",
        1: "Melanocytic nevus",
        2: "Basal cell carcinoma",
        3: "Actinic keratosis",
        4: "Benign keratosis",
        5: "Dermatofibroma",
        6: "Vascular lesion"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": None,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9582311026945434,
    "step_size": 20,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_0": 20,
    "T_mult": 1,
    "mode": "max",
    "patience": 20,
    "factor": 0.3,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI"],
    "loss_function_name": {
        "segmentation": "DiceLoss",
        "classification": "CrossEntropyLoss"
    },
    "class_weight": [0.029, 1-0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 150,
    # for testing on CPU
    # "end_epoch": 1,
    "best_metric": 0,
    "terminal_show_freq": 20,
    "save_epoch_freq": 50,
}
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ISIC-2018", help="dataset name")
    parser.add_argument("--model", type=str, default="AttU_Net", help="model name")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--dimension", type=str, default="2d", help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="BASIC", help="scaling version of PMFSNet")
    parser.add_argument("--epoch", type=int, default=150, help="training epoch")
    parser.add_argument("--task", type=str, default="multitask", choices=["segmentation", "classification", "multitask"], help="which task to perform")
    args = parser.parse_args()
    return args



def main():
    # analyse console arguments
    args = parse_args()

    params = params_ISIC_2018

    # update the dictionary of hyperparameters used for training
    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", ("NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["model_name"] = args.model
    if args.pretrain_weight is not None:
        params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
    if args.epoch is not None:
        params["end_epoch"] = args.epoch
        params["save_epoch_freq"] = max(1, args.epoch // 4)

    params["seg_guided_cls"] = True   # False if normal multitask

    # launch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])
    
    # for testing on CPU
    # params["device"] = torch.device("cpu")

    # get the cuda device
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("Complete the initialization of configuration")

    # detect model tasks
    # set task mode from CLI
    if args.task == "segmentation":
        params["segmentation"] = True
        params["classification"] = False
        params["metric_names"] = ["ACC_SEG", "DSC", "IoU", "JI"]
        params["seg_classes"] = 2 
        params["cls_classes"] = None 
        params["learning_rate"] = 0.0001

    elif args.task == "classification":
        params["segmentation"] = False
        params["classification"] = True
        params["metric_names"] = ["ACC_CLS", "AUC_ROC", "F1_MACRO"]
        params["seg_classes"] = None 
        params["cls_classes"] = 7 
        params["learning_rate"] = 0.0001

    elif args.task == "multitask":
        params["segmentation"] = True
        params["classification"] = True
        params["metric_names"] = ["ACC_SEG", "DSC", "IoU", "JI", "ACC_CLS", "AUC_ROC", "F1_MACRO"]
        params["seg_classes"] = 2 
        params["cls_classes"] = 7 
        params["seg_guided_cls"] = True
        params["lr_cls"] = 0.00003
        params["lr_seg"] = 0.00005
        params["learning_rate"] = 0.00005

    if args.model == "EGEUNet" and params["segmentation"]:
        params["learning_rate"] = 0.001

    print(f"Segmentation training: {params['segmentation']}, Classification training: {params['classification']}")

    # initialize the dataloader
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("Complete the initialization of dataloader")

    # initialize the model, optimizer, and lr_scheduler
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    # initialize the loss function
    loss_functions = losses.get_loss_function(params)
    print("Complete the initialization of loss function")

    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # initialize the trainer
    trainer = trainers.get_trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_functions, metric)

    # resume or load pretrained weights
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()
    print("Complete the initialization of trainer")

    # start training
    trainer.training()



if __name__ == '__main__':
    main()


