import torch
import torch.nn as nn
from typing import Dict


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        loss = 0.0
        details = {}
        for k, w in self.weights.items():
            if k in logits and k in targets and targets[k] is not None:
                l = self.bce(logits[k], targets[k].float())
                details[k] = float(l.item())
                loss = loss + w * l
        return loss, details
