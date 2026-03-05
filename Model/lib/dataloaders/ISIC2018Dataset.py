# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/29 01:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms.two as my_transforms


class ISIC2018Dataset(Dataset):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(ISIC2018Dataset, self).__init__()
        self.opt = opt
        self.mode = mode

        # follow parse_args flags
        self.segmentation = opt["segmentation"]
        self.classification = opt["classification"]

        self.root = opt["dataset_path"]

        self.transforms_dict = {
            "train": my_transforms.Compose([
                my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ]),
            "valid": my_transforms.Compose([
                my_transforms.Resize(self.opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ])
        }

        # Segmentation
        if self.segmentation:
            folder = os.path.join(self.root, "segmentation", mode)
            self.images_list = sorted(glob.glob(os.path.join(folder, "images", "*.jpg")))
            self.labels_list = sorted(glob.glob(os.path.join(folder, "masks", "*_segmentation.png")))

        # Classification
        if self.classification:
            folder = os.path.join(self.root, "classification", mode)
            self.images_list = sorted(glob.glob(os.path.join(folder, "images", "*.jpg")))
            label_csv = os.path.join(folder, "labels.csv")
            if not os.path.exists(label_csv):
                raise FileNotFoundError(f"Classification labels CSV not found: {label_csv}")
            self.labels_dict = {}
            with open(label_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels_dict[row["image"]] = [float(row[c]) for c in reader.fieldnames if c != "image"]

            # keep only images that exist in CSV
            self.images_list = [
                img for img in self.images_list
                if os.path.splitext(os.path.basename(img))[0] in self.labels_dict
            ]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        mask = None
        label = None

        if self.segmentation:
            mask_path = self.labels_list[index]
            mask = cv2.imread(mask_path, -1)
            mask[mask == 255] = 1

        if self.classification:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            label = torch.tensor(self.labels_dict[filename], dtype=torch.float32)

        # Apply transforms
        if self.segmentation:
            image, mask = self.transforms_dict[self.mode](image, mask)
        else:
            dummy_mask = np.zeros_like(image[:, :, 0])
            image, _ = self.transforms_dict[self.mode](image, dummy_mask)

        # multitask
        if self.segmentation and self.classification:
            return image, mask, label
        elif self.segmentation:
            return image, mask
        elif self.classification:
            return image, label