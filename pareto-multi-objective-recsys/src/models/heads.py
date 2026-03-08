import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    def __init__(self, input_dim: int, use_creator: bool = True):
        super().__init__()
        self.use_creator = use_creator
        # logits returned from towers in custom_mmoe; heads keep identity but
        # allow for calibration layers if desired.
        self.cal_like = nn.Identity()
        self.cal_long = nn.Identity()
        self.cal_creator = nn.Identity()

    def forward(self, preds: dict):
        out = {}
        out['like_score'] = self.cal_like(preds['like'])
        out['longview_score'] = self.cal_long(preds['longview'])
        if self.use_creator:
            out['creator_score'] = self.cal_creator(preds['creator'])
        return out
