from lib.utils import *


class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=1, weight=None, sigmoid_normalization=True, mode="extension"):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.weight = weight
        self.mode = mode
        
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target):
        target = target.unsqueeze(1).float()

        assert input.size() == target.size(), "Inconsistency of dimensions between predicted and labeled images after one-hot processing in dice loss"

        input = self.normalization(input)

        return compute_per_channel_dice(input, target, epsilon=1e-6, mode=self.mode)


    def forward(self, input, target):
        per_channel_dice = self.dice(input, target)

        if self.weight is None:
            real_weight = torch.ones_like(per_channel_dice)
        else:
            real_weight = self.weight.clone()

        for i, dice in enumerate(per_channel_dice):
            if dice.item() < 1e-6:
                real_weight[i] = 0

        denom = torch.sum(real_weight)
        # to avoid NaN when all channels are empty
        if denom == 0:
            return torch.tensor(1.0, device=input.device)
        weighted_dsc = torch.sum(per_channel_dice * real_weight) / denom

        loss = 1. - weighted_dsc

        return loss
