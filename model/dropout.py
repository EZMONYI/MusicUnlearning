import torch
import torch.nn as nn
import torch.nn.functional as F

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

