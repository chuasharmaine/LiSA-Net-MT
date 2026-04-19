import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        return self.ce(pred, target)