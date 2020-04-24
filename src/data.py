
import random
from simulator.cube import Cube
import torch
from torch.utils.data import Dataset, DataLoader

ops = 'LRUDFB'

class RubikDataset(Dataset):
    def __init__(self, size, maxIters):
        super(RubikDataset, self).__init__()
        self.size = size
        self.maxIters = maxIters

    def __getitem__(self, _):
        x, y, scrambles = self._getCube()
        x = _cubeToIndices(x)
        x = _indicesToOneHot(x)
        y = _opToIndices(y)
        return x, y, scrambles

    def __len__(self):
        return self.size

    def _getCube(self):
        cube = Cube()
        seq = ''
        scrambles = random.randint(1, self.maxIters)
        for _ in range(scrambles):
            op = random.choice(ops)
            amount = random.choice(['', '2', 'i']) # normal op, 2x op, inverse op
            if amount == '2':
                seq += f' {op} {op}'
            else:
                seq += f' {op}{amount}'
        cube.sequence(seq)
        reverseAmount = {'': 'i', '2': '2', 'i': ''}[amount]
        return str(cube), f'{op}{reverseAmount}', scrambles


def _opToIndices(op):
    opIdx = ops.index(op[0])
    if len(op) == 1: amountIdx = 0
    if len(op) == 2 and op[1] == '2': amountIdx = 1
    if len(op) == 2 and op[1] == 'i': amountIdx = 2
    y = torch.tensor([opIdx, amountIdx], dtype=torch.long)
    return y


def _cubeToIndices(cube):
    cube = cube.split('\n')
    indices = torch.zeros((6, 3, 3), dtype=torch.long) # SHW

    # x coordinates have stride 4, y coordinates have 3. See Cube.__str__() method for details
    _setIndices(indices[0], cube, 0, 3) # L
    _setIndices(indices[1], cube, 8, 3) # R
    _setIndices(indices[2], cube, 4, 0) # U
    _setIndices(indices[3], cube, 4, 6) # D
    _setIndices(indices[4], cube, 4, 3) # F
    _setIndices(indices[5], cube, 12, 3) # B
    return indices


def _setIndices(indices, cube, x0, y0):
    colors = "OYWGBR"
    c2Idx = {colors[i]: i for i in range(len(colors))}
    for y in range(3):
        for x in range(3):
            indices[y][x] = c2Idx[cube[y0 + y][x0 + x]]


def _indicesToOneHot(indices):
    eye = torch.eye(6, dtype=torch.float32)
    oneHot = eye[indices] # SHWC
    oneHot = oneHot.permute(0, 3, 1, 2) # SCHW
    return oneHot
