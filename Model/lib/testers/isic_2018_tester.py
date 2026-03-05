# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/01/01 00:33
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from torchvision import transforms


class ISIC2018Tester:
    """
    Tester class
    """

    def __init__(self, opt, model, metrics=None):
        self.opt = opt
        self.model = model
        self.metrics = metrics
        self.device = self.opt["device"]

        self.statistics_dict = self.init_statistics_dict()

    def inference(self, image_path, output_path):
        test_transforms = transforms.Compose([
            transforms.Resize(self.opt["resize_shape"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
        ])

        image_pil = Image.open(image_path).convert("RGB")
        w, h = image_pil.size
        image = test_transforms(image_pil)

        self.model.eval()
        with torch.no_grad():
            image = torch.unsqueeze(image, dim=0)
            image = image.to(self.device)
            output = self.model(image)

        seg_out, cls_out = None, None
        if self.opt["segmentation"] and self.opt["classification"]:
            seg_out, cls_out = output
        elif self.opt["segmentation"]:
            seg_out = output
        elif self.opt["classification"]:
            cls_out = output

        # Segmentation output
        if seg_out is not None and output_path is not None:
            segmented_image = torch.argmax(seg_out, dim=1).squeeze(0).to(dtype=torch.uint8).cpu().numpy()
            segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_AREA)
            segmented_image[segmented_image == 1] = 255
            cv2.imwrite(output_path, segmented_image)
            print("Save segmented image to {}".format(output_path))

        # Classification output
        if cls_out is not None:
            pred_class = torch.argmax(cls_out, dim=1).item()
            print("Predicted class:", self.opt["index_to_class_dict"][pred_class])

    def evaluation(self, dataloader):
        self.reset_statistics_dict()
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, leave=True):

                if self.opt["segmentation"] and self.opt["classification"]:
                    input_tensor, seg_target, cls_target = batch
                elif self.opt["segmentation"]:
                    input_tensor, seg_target = batch
                    cls_target = None
                elif self.opt["classification"]:
                    input_tensor, cls_target = batch
                    seg_target = None

                input_tensor = input_tensor.to(self.device)

                if seg_target is not None:
                    seg_target = seg_target.to(self.device)

                if cls_target is not None:
                    cls_target = cls_target.to(self.device)

                output = self.model(input_tensor)

                seg_out, cls_out = None, None
                if self.opt["segmentation"] and self.opt["classification"]:
                    seg_out, cls_out = output
                elif self.opt["segmentation"]:
                    seg_out = output
                elif self.opt["classification"]:
                    cls_out = output

                if seg_out is not None:
                    self.calculate_metric_and_update_statistcs(
                        seg_out.cpu(),
                        seg_target.cpu(),
                        len(seg_target),
                        task="segmentation"
                    )

                if cls_out is not None:
                    self.calculate_metric_and_update_statistcs(
                        cls_out.cpu(),
                        cls_target.cpu(),
                        len(cls_target),
                        task="classification"
                    )

        if self.opt["segmentation"]:
            class_IoU = self.statistics_dict["total_area_intersect"] / self.statistics_dict["total_area_union"]
            class_IoU = np.nan_to_num(class_IoU)
            dsc = self.statistics_dict["DSC_sum"] / self.statistics_dict["count"]
            JI = self.statistics_dict["JI_sum"] / self.statistics_dict["count"]
            ACC_seg = self.statistics_dict["ACC_seg_sum"] / self.statistics_dict["count"] if "ACC_SEG" in self.statistics_dict else 0
            print("valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}".format(dsc, class_IoU[1], ACC_seg, JI))
        if self.opt["classification"]:
            ACC_cls = self.statistics_dict.get("ACC_cls_sum", 0) / self.statistics_dict["count"]
            AUC_ROC = self.statistics_dict.get("AUC_ROC_sum", 0) / self.statistics_dict["count"]
            F1_MACRO = self.statistics_dict.get("F1_MACRO_sum", 0) / self.statistics_dict["count"]
            print("valid_ACC_cls:{:.6f}  valid_AUC_ROC:{:.6f}  valid_F1_MACRO:{:.6f}".format(ACC_cls, AUC_ROC, F1_MACRO))

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, task="segmentation"):

        if task == "segmentation":
            mask = torch.zeros(self.opt["classes"])
            unique_index = torch.unique(target).int()
            for index in unique_index:
                mask[index] = 1
        else:
            mask = torch.ones(self.opt["classes"])

        self.statistics_dict["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size

        for metric_name, metric_func in self.metrics.items():
            if task == "segmentation":
                if metric_name == "IoU":
                    area_intersect, area_union, _, _ = metric_func(output, target)
                    self.statistics_dict["total_area_intersect"] += area_intersect.numpy()
                    self.statistics_dict["total_area_union"] += area_union.numpy()
                elif metric_name == "ACC_SEG":
                    self.statistics_dict["ACC_seg_sum"] += metric_func(output, target) * cur_batch_size
                elif metric_name == "JI":
                    batch_mean_JI = metric_func(output, target)
                    self.statistics_dict["JI_sum"] += batch_mean_JI * cur_batch_size
                elif metric_name == "DSC":
                    batch_mean_DSC = metric_func(output, target)
                    self.statistics_dict["DSC_sum"] += batch_mean_DSC * cur_batch_size
                else:
                    per_class_metric = metric_func(output, target)
                    per_class_metric = per_class_metric * mask
                    self.statistics_dict[metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
                    for j, class_name in self.opt["index_to_class_dict"].items():
                        self.statistics_dict[metric_name][class_name] += per_class_metric[j].item() * cur_batch_size
            elif task == "classification":
                if metric_name == "ACC_CLS":
                    self.statistics_dict["ACC_cls_sum"] += metric_func(output, target) * cur_batch_size
                elif metric_name == "AUC_ROC":
                    self.statistics_dict.setdefault("AUC_ROC_sum", 0)
                    self.statistics_dict["AUC_ROC_sum"] += metric_func(output, target) * cur_batch_size
                elif metric_name == "F1_MACRO":
                    self.statistics_dict.setdefault("F1_MACRO_sum", 0)
                    self.statistics_dict["F1_MACRO_sum"] += metric_func(output, target) * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
            for metric_name in self.opt["metric_names"]
        }
        statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["JI_sum"] = 0.0
        statistics_dict["ACC_seg_sum"] = 0.0
        statistics_dict["ACC_cls_sum"] = 0.0
        statistics_dict["DSC_sum"] = 0.0
        statistics_dict["AUC_ROC_sum"] = 0.0
        statistics_dict["F1_MACRO_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        self.statistics_dict["count"] = 0
        self.statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["JI_sum"] = 0.0
        self.statistics_dict["ACC_seg_sum"] = 0.0
        self.statistics_dict["ACC_cls_sum"] = 0.0
        self.statistics_dict["DSC_sum"] = 0.0
        self.statistics_dict["AUC_ROC_sum"] = 0.0
        self.statistics_dict["F1_MACRO_sum"] = 0.0
        for _, class_name in self.opt["index_to_class_dict"].items():
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[metric_name][class_name] = 0.0

    def load(self):
        pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
        model_state_dict = self.model.state_dict()
        load_count = 0
        for param_name in model_state_dict.keys():
            if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                load_count += 1
        self.model.load_state_dict(model_state_dict, strict=True)
        print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))