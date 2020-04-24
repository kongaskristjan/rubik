
import torch
from torch import nn
from torch.nn import functional as F

class CubeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CubeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, output, target):
        output = output.view(output.shape[0], -1) # NSC -> N(SC)
        target = 3 * target[:, 0] + target[:, 1]

        loss = self.criterion(output, target)

        pred = torch.argmax(output, dim=1)
        if self.reduction == 'mean': acc = torch.mean(pred == target)
        if self.reduction == 'sum': acc = torch.sum(pred == target)
        return loss, acc


class Stats:
    def __init__(self):
        self.n = 0
        self.loss = 0
        self.acc = 0

    def accumulate(self, _n, _loss, _acc):
        self.n += _n
        self.loss += _loss.item()
        self.acc += _acc.item()

    def getLoss(self):
        return self.loss / max(1, self.n)

    def getAcc(self):
        return self.acc / max(1, self.n)
