# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:57
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from lib.metrics import Tooth
from lib.metrics import MMOTU
from lib.metrics import ISIC2018


def get_metric(opt):
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        metrics = []
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics.append(Tooth.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"]))

            elif metric_name == "ASSD":
                metrics.append(Tooth.AverageSymmetricSurfaceDistance(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "HD":
                metrics.append(Tooth.HausdorffDistance(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "SO":
                metrics.append(Tooth.SurfaceOverlappingValues(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], theta=1.0))

            elif metric_name == "SD":
                metrics.append(Tooth.SurfaceDice(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], theta=1.0))

            elif metric_name == "IoU":
                metrics.append(Tooth.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "MMOTU":
        metrics = {}
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics[metric_name] = MMOTU.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"])

            elif metric_name == "IoU":
                metrics[metric_name] = MMOTU.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "ISIC-2018":
        metrics = {}
        seg_classes = opt.get("seg_classes", 1)
        cls_classes = opt.get("cls_classes", 1)
        for metric_name in opt["metric_names"]:
            if metric_name in ["DSC", "IoU", "JI"]:
                # segmentation metrics
                num_classes = seg_classes
            elif metric_name in ["ACC", "AUC_ROC", "F1_MACRO"]:
                # classification metrics
                num_classes = cls_classes
            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

            if metric_name == "DSC":
                metrics[metric_name] = ISIC2018.DICE(num_classes=num_classes, sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"])

            elif metric_name == "IoU":
                metrics[metric_name] = ISIC2018.IoU(num_classes=num_classes, sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "JI":
                metrics[metric_name] = ISIC2018.JI(num_classes=num_classes, sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "ACC":
                # segmentation or classification ACC
                if opt.get("segmentation", True):
                    metrics[metric_name] = ISIC2018.ACCSEG(num_classes=num_classes, sigmoid_normalization=opt["sigmoid_normalization"])
                elif opt.get("classification", False):
                    metrics[metric_name] = ISIC2018.ACCCLS()
            
            elif metric_name == "AUC_ROC":
                metrics[metric_name] = ISIC2018.AUC_ROC(num_classes=num_classes, sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "F1_MACRO":
                metrics[metric_name] = ISIC2018.F1_MACRO(num_classes=num_classes, sigmoid_normalization=opt["sigmoid_normalization"])

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize metrics")

    return metrics