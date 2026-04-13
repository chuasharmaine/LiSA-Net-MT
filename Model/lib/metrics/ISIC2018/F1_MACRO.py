from sklearn.metrics import f1_score
import torch

class F1_MACRO(object):
    """
    Macro F1 Score for multi-class classification.

    This implementation accumulates predictions across the entire
    dataset and computes the F1 score once per epoch.

    Macro F1 = unweighted mean of F1 scores across all classes.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []

    def update(self, input, target):
        """
        Store predictions and targets from one batch.

        Args:
            input: model logits [B, C]
            target: ground truth labels [B]
        """
        if input.ndim == 1:
            raise ValueError("F1_MACRO expects logits of shape [B, C], got [B]")

        preds = torch.argmax(input, dim=1)

        preds = preds.detach().cpu().numpy()
        targets = target.detach().cpu().numpy()

        self.all_preds.extend(preds.tolist())
        self.all_targets.extend(targets.tolist())

    def compute(self):
        """
        Compute macro F1 over the entire dataset.

        Returns:
            float: macro F1 score
        """
        if len(self.all_targets) == 0:
            return 0.0
        return f1_score(self.all_targets, self.all_preds, average='macro')