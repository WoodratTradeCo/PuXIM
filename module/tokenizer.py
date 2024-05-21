import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F


class Scale_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 768, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        return self.seq(x)
