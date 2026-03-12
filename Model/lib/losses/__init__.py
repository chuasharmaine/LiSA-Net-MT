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


def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        # Ensure num_classes is always int
        num_classes = opt.get("seg_classes", 1)
        if num_classes is None:
            num_classes = 1

        # Safely get class_weight
        class_weight = opt.get("class_weight")
        if class_weight is None:
            class_weight = [1.0] * num_classes

        weight_tensor = torch.FloatTensor(class_weight).to(opt.get("device", "cpu"))

        loss_function = DiceLoss(
            classes=num_classes,
            weight=weight_tensor,
            sigmoid_normalization=opt.get("sigmoid_normalization", False),
            mode=opt.get("dice_loss_mode", "standard")
        )

    else:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")

    return loss_function
