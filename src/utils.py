
import torch
from torch import nn
from torch.nn import functional as F

class CubeLoss(nn.Module):
    def __init__(self):
        super(CubeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        output = output.view(output.shape[0], -1) # NSC -> N(SC)
        target = 3 * target[:, 0] + target[:, 1]
        loss = self.criterion(output, target)
        return loss