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

        if self.segmentation and self.classification:
            self.root = os.path.join(opt["dataset_path"], "multitask", mode)
        elif self.segmentation:
            self.root = os.path.join(opt["dataset_path"], "segmentation", mode)
        else:
            self.root = os.path.join(opt["dataset_path"], "classification", mode)

        self.image_paths = sorted(glob.glob(os.path.join(self.root, "images", "*.jpg")))
        self.image_names = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]

        self.cls_labels_dict = {}
        # Classification
        if self.classification:
            label_csv = os.path.join(self.root, "labels.csv")
            with open(label_csv, "r") as f:
                reader = csv.reader(f)
                header = next(reader)  
                for row in reader:
                    image_id = row[0]
                    labels = list(map(float, row[1:]))
                    self.cls_labels_dict[image_id] = labels

            self.image_names = [n for n in self.image_names if n in self.cls_labels_dict]

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

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        
        img_path = os.path.join(self.root, "images", image_name + ".jpg")
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"cannot find: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.segmentation:
            mask_path = os.path.join(self.root, "masks", image_name + "_segmentation.png")
            mask = cv2.imread(mask_path, 0)
            mask[mask == 255] = 1 
        
        # apply transforms
        if self.segmentation and self.classification:
            image, mask = self.transforms_dict[self.mode](image, mask)
        elif self.segmentation:
            image, mask = self.transforms_dict[self.mode](image, mask)
        elif self.classification:
            image, _ = self.transforms_dict[self.mode](image, np.zeros_like(image[:,:,0]))

        label = None
        if self.classification:
            label_list = self.cls_labels_dict[image_name]
            label = torch.tensor(label_list, dtype=torch.float32)
            label = torch.argmax(label).long()

        if self.segmentation and self.classification:
            return image, mask, label
        elif self.segmentation:
            return image, mask
        elif self.classification:
            return image, label