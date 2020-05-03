
import torch
from torch import nn
from torch.nn import functional as F

class CubeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean', 'sum')
        super(CubeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, output, target):
        output = output.view(output.shape[0], -1) # NSC -> N(SC)
        target = 2 * target[:, 0] + target[:, 1]

        loss = self.criterion(output, target)

        pred = torch.argmax(output, dim=1)
        acc = (pred == target).float()
        if self.reduction == 'mean': acc = torch.mean(acc)
        if self.reduction == 'sum': acc = torch.sum(acc)
        return loss, acc


class Stats:
    def __init__(self):
        self.n = 0
        self.loss = 0

    def accumulate(self, _n, _loss):
        self.n += _n
        self.loss += torch.sum(_loss).item()

    def getLoss(self):
        return self.loss / max(1, self.n)


class PerClassStats:
    def __init__(self, maxScrambles):
        self.stats = [Stats() for i in range(maxScrambles + 1)]

    def accumulate(self, scrambles, loss):
        for scr, l in zip(scrambles, loss):
            self.stats[scr].accumulate(1, l)

    def distStr(self):
        return '  '.join([f'{i}: {s.getLoss()**0.5:.3f}' for i, s in enumerate(self.stats[1:], start=1)])
