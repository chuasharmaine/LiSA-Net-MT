# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch

from .DiceLoss import DiceLoss
from .CrossEntropyLoss import CrossEntropyLoss


def get_loss_function(opt):
    loss_functions = {}
    device = opt.get("device", "cpu")
    
    if opt.get("segmentation", False):
        seg_classes = opt.get("seg_classes", 1)
        seg_class_weight = opt.get("class_weight", [1.0] * seg_classes)
        seg_weight_tensor = torch.FloatTensor(seg_class_weight).to(device)

        loss_functions["segmentation"] = DiceLoss(
            classes=seg_classes,
            weight=seg_weight_tensor,
            sigmoid_normalization=opt.get("sigmoid_normalization", False),
            mode=opt.get("dice_loss_mode", "standard")
        )
    
    if opt.get("classification", False):
        # compute weights from dataset counts
        loss_functions["classification"] = CrossEntropyLoss()

    if not loss_functions:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")
    
    return loss_functions
