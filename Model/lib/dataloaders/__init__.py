# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .ToothDataset import ToothDataset
from .MMOTUDataset import MMOTUDataset
from .ISIC2018Dataset import ISIC2018Dataset


def get_dataloader(opt):
    """
    get dataloader
    Args:
        opt: params dict
    Returns:
    """
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        train_set = ToothDataset(opt, mode="train")
        valid_set = ToothDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "MMOTU":
        train_set = MMOTUDataset(opt, mode="train")
        valid_set = MMOTUDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    elif opt["dataset_name"] == "ISIC-2018":
        train_set = ISIC2018Dataset(opt, mode="train")
        valid_set = ISIC2018Dataset(opt, mode="valid")

        if opt.get("classification", False):
            train_counts = [779, 4693, 360, 229, 769, 81, 99]  # MEL, NV, BCC, AKIEC, BKL, DF, VASC
            class_weights = [1.0 / c for c in train_counts]

            # Assign a weight to every sample based on its class label
            sample_weights = []
            for i in range(len(train_set)):
                item = train_set[i]
                label = item[-1]
                if hasattr(label, 'item'):
                    label = label.item()
                sample_weights.append(class_weights[int(label)])

            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float),
                num_samples=len(sample_weights),
                replacement=True
            )

            train_loader = DataLoader(train_set, batch_size=opt["batch_size"], sampler=sampler, num_workers=opt["num_workers"], pin_memory=True)

        else:
            train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader


def get_test_dataloader(opt):
    """
    get test dataloader
    :param opt: params dict
    :return:
    """
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        valid_set = ToothDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "MMOTU":
        valid_set = MMOTUDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "ISIC-2018":
        valid_set = ISIC2018Dataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader