
import torch
from torch import nn
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def getModel(mul=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(6 * 6 * 3 * 3, round(mul * 4096)),
        nn.ReLU(inplace=True),
        nn.Linear(round(mul * 4096), round(mul * 2048)),
        nn.ReLU(inplace=True),
        nn.Linear(round(mul * 2048), round(mul * 512)),
        nn.ReLU(inplace=True),
        nn.Linear(round(mul * 512), 6 * 2),
    )