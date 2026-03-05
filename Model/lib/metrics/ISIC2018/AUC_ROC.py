import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

from lib.utils import *


class AUC_ROC(object):
    def __init__(self, num_classes=2, sigmoid_normalization=False):
        """
        Multi-class or binary AUC-ROC metric calculator.
        Args:
            num_classes: number of classes
            sigmoid_normalization: whether to apply sigmoid (binary) or softmax (multi-class)
        """
        super(AUC_ROC, self).__init__()
        self.num_classes = num_classes
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
    

    def __call__(self, input, target):
        """
        Args:
            input: model output, shape (B, C, H, W) or (B, C)
            target: ground truth, shape (B, H, W) or (B)
        Returns:
            AUC-ROC score
        """

        input = self.normalization(input)

        # If segmentation output (B, C, H, W)
        if input.dim() > 2:
            # reshape to (N_pixels, C)
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(-1, self.num_classes)
            target = target.view(-1)

        else:
            # classification output (B, C)
            target = target.view(-1)

        input_np = input.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        try:
            return roc_auc_score(target_np, input_np, multi_class='ovr')
        except ValueError:
            return 0.0