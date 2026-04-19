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
       self.reset()

    def reset(self):
        """
        Clears stored predictions and targets.

        Call this at the START of each epoch.
        """
        self.all_probs = []
        self.all_targets = []

    def update(self, input, target):
        """
        Stores predictions and targets for the current batch.
        """
        
        # Ensure classification output
        if input.ndim != 2:
            return

        # Convert one-hot labels to class indices
        if target.ndim > 1 and target.shape[1] > 1:
            target = target.argmax(dim=1)

        self.all_probs.append(input.detach().cpu())
        self.all_targets.append(target.detach().cpu())

    def compute(self):
        """
        Computes final AUC using ALL accumulated data.

        Returns:
            Multi-class AUC-ROC score (one-vs-rest)
        """

        # Concatenate all batches into full dataset tensors
        probs = torch.cat(self.all_probs).numpy()
        targets = torch.cat(self.all_targets).numpy()

        try:
            return roc_auc_score(targets, probs, multi_class='ovr')

        except ValueError:
            return 0.5