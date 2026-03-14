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
        self.segmentation = opt["segmentation"]
        self.classification = opt["classification"]

        self.root = opt["dataset_path"]

        self.seg_images_list = []
        self.seg_labels_list = []
        self.cls_images_list = []
        self.cls_labels_dict = {}

        # transforms
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
            self.seg_images_list = sorted(glob.glob(os.path.join(folder, "images", "*.jpg")))
            self.seg_labels_list = sorted(glob.glob(os.path.join(folder, "masks", "*_segmentation.png")))

        # Classification
        if self.classification:
            folder = os.path.join(self.root, "classification", mode)
            self.cls_images_list = sorted(glob.glob(os.path.join(folder, "images", "*.jpg")))
            label_csv = os.path.join(folder, "labels.csv")
            self.cls_labels_dict = {}
            with open(label_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.cls_labels_dict[row["image"]] = [float(row[c]) for c in reader.fieldnames if c != "image"]
            self.cls_images_list = [
                img for img in self.cls_images_list
                if os.path.splitext(os.path.basename(img))[0] in self.cls_labels_dict
            ]

    def __len__(self):
        if self.segmentation and self.classification:
            return max(len(self.seg_images_list), len(self.cls_images_list))
        elif self.segmentation:
            return len(self.seg_images_list)
        elif self.classification:
            return len(self.cls_images_list)
        else:
            return 0

    def __getitem__(self, index):
        mask = None
        label = None

        if self.segmentation:
            seg_index = index % len(self.seg_images_list)
            image_path = self.seg_images_list[seg_index]
            mask_path = self.seg_labels_list[seg_index]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, -1)
            mask[mask == 255] = 1
            mask = mask.astype(np.uint8)
        else:
            cls_index = index % len(self.cls_images_list)
            image_path = self.cls_images_list[cls_index]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

        # Classification
        label = None
        if self.classification:
            cls_index = index % len(self.cls_images_list)
            cls_image_path = self.cls_images_list[cls_index]
            filename = os.path.splitext(os.path.basename(cls_image_path))[0]
            label = torch.tensor(self.cls_labels_dict[filename], dtype=torch.float32)
            if not self.segmentation:
                image = cv2.imread(cls_image_path, cv2.IMREAD_COLOR)

        # Apply transforms
        image, mask = self.transforms_dict[self.mode](image, mask)

        # multitask
        if self.segmentation and self.classification:
            return image, mask, label
        elif self.segmentation:
            return image, mask
        elif self.classification:
            return image, label