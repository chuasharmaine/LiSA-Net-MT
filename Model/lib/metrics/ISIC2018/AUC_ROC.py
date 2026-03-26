import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

from lib.utils import *


class AUC_ROC(object):
   def __init__(self, num_classes=7):
       """
       Multi-class AUC-ROC metric calculator.
       Args:
           num_classes: number of classes
       """
       self.num_classes = num_classes

   def __call__(self, input, target):
       """
       Args:
           input: model output, shape (B, C, H, W) or (B, C)
           target: ground truth, shape (B, H, W) or (B) or one-hot (B, C)
       Returns:
           AUC-ROC score
       """

       # If segmentation output (B, C, H, W)
       if input.dim() > 2:
           # reshape to (N_pixels, C)
           input = input.permute(0, 2, 3, 1).contiguous()
           input = input.view(-1, self.num_classes)
           target = target.view(-1)

       else:
           # classification output (B, C)
           # convert one-hot to class indices if needed
           if target.ndim > 1 and target.shape[1] > 1:
               target = target.argmax(dim=1)
           else:
               target = target.view(-1)

       # apply softmax for multi-class probabilities
       probs = torch.softmax(input, dim=1)
       input_np = probs.detach().cpu().numpy()
       target_np = target.detach().cpu().numpy()

       try:
           return roc_auc_score(target_np, input_np, multi_class='ovr')
       except ValueError:
           return 0.5