import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np

from lib.utils import *


class F1_MACRO(object):
    def __init__(self, num_classes=2, sigmoid_normalization=False):
        """
        Macro-F1 Score for multi-class classification
        Args:
            num_classes: number of classes
            sigmoid_normalization: whether to apply sigmoid (binary) or softmax (multi-class)
        """
        super(F1_MACRO, self).__init__()
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
            Macro F1 score
        """
        input = self.normalization(input)
        if input.dim() > 2:  # segmentation maps
            input = torch.argmax(input, dim=1)
        else:
            input = torch.argmax(input, dim=1)
        
        input_np = input.detach().cpu().numpy().reshape(-1)
        target_np = target.detach().cpu().numpy().reshape(-1)

        return f1_score(target_np, input_np, average='macro')