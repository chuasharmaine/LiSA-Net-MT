# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/6/13 2:54
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import math
import torch
import torch.nn as nn
import numpy as np


def cal_accuracy(pred, target):
    """Simple ACC calculation"""
    correct = (pred == target).sum()
    total = np.prod(target.shape)
    return correct / total


class ACCSEG(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):
        """
        定义ACC评价指标计算器

        :param num_classes: 类别数
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def __call__(self, input, target):
        """
        ACC

        :param input: 网络模型输出的预测图,(B, C, H, W)
        :param target: 标注图像,(B, H, W)
        :return:
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input.cpu(), dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"
        # 转换seg和target数据类型为numpy.ndarray
        # 改良变成float tensor
        seg = seg.float()
        target = target.float()
        
        correct = (seg == target).sum()
        total = target.numel()

        return correct / total
    
class ACCCLS:
    """
    Classification Accuracy (for image-level classification)
    input: (B, C) logits
    target: (B,) labels
    """
    def __init__(self):
        pass

    def __call__(self, input, target):
        if input.ndim > 1:
            pred = torch.argmax(input, dim=1)
        else:
            pred = input
          
        if target.ndim == 2:
            target = torch.argmax(target, dim=1)

        assert pred.shape == target.shape, "pred and target shapes must match"
        correct = (pred == target).sum()
        total = target.size(0)
        return correct.float() / total 