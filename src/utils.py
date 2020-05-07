
import torch
from torch import nn
from torch.nn import functional as F

defaultColors = 'YGORWB'
ops = 'LRUDFB'
invOps = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U', 'F': 'B', 'B': 'F'}

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


def opToIndices(op):
    opIdx = ops.index(op[0])
    if len(op) == 1: amountIdx = 0
    if len(op) == 2 and op[1] == 'i': amountIdx = 1
    y = torch.tensor([opIdx, amountIdx], dtype=torch.long)
    return y


def indicesToOp(indices):
    op, amount = indices
    op, amount = op.item(), amount.item()
    op = ops[op]

    assert amount in (0, 1)
    if amount == 0: return op
    if amount == 1: return f'{op}i'


def strToIndices(cubeStr, colors=defaultColors):
    lines = cubeStr.split('\n')
    indices = torch.zeros((6, 3, 3), dtype=torch.long) # SHW

    # x coordinates have stride 4, y coordinates have 3. See Cube.__str__() method for details
    _setIndices(indices[0], lines, 0, 3, colors) # L
    _setIndices(indices[1], lines, 8, 3, colors) # R
    _setIndices(indices[2], lines, 4, 0, colors) # U
    _setIndices(indices[3], lines, 4, 6, colors) # D
    _setIndices(indices[4], lines, 4, 3, colors) # F
    _setIndices(indices[5], lines, 12, 3, colors) # B
    return indices


def indicesToStr(indices, colors=defaultColors):
    lines = [' ' * 15] * 9
    _setString(indices[0], lines, 0, 3, colors) # L
    _setString(indices[1], lines, 8, 3, colors) # R
    _setString(indices[2], lines, 4, 0, colors) # U
    _setString(indices[3], lines, 4, 6, colors) # D
    _setString(indices[4], lines, 4, 3, colors) # F
    _setString(indices[5], lines, 12, 3, colors) # B

    cubeStr = '\n'.join(lines)
    return cubeStr


def indicesToOneHot(indices):
    eye = torch.eye(6, dtype=torch.float32)
    oneHot = eye[indices] # SHWC
    oneHot = oneHot.permute(0, 3, 1, 2) # SCHW
    return oneHot


def _setIndices(indices, lines, x0, y0, colors=defaultColors):
    for y in range(3):
        for x in range(3):
            indices[y][x] = colors.index(lines[y0 + y][x0 + x])


def _setString(indices, lines, x0, y0, colors=defaultColors):
    for y in range(3):
        for x in range(3):
            char = colors[indices[y, x].item()]
            lines[y0 + y] = lines[y0 + y][:x0 + x] + char + lines[y0 + y][x0 + x + 1:]
