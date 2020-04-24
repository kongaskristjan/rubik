
import random
from simulator.cube import Cube
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

ops = 'LRUDFB'

class RubikDataset(Dataset):
    def __init__(self, size, maxIters, tfms=Compose([])):
        super(RubikDataset, self).__init__()
        self.size = size
        self.maxIters = maxIters
        self.tfms = tfms

    def __getitem__(self, _):
        x, y, scrambles = self._getCube()
        x, y = self.tfms((x, y))
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


class CubeToIndices:
    def __call__(self, xy):
        cube, op = xy
        cube = cube.split('\n')
        indices = torch.zeros((6, 3, 3), dtype=torch.long) # SHW

        # x coordinates have stride 4, y coordinates have 3. See Cube.__str__() method for details
        self._setIndices(indices[0], cube, 0, 3) # L
        self._setIndices(indices[1], cube, 8, 3) # R
        self._setIndices(indices[2], cube, 4, 0) # U
        self._setIndices(indices[3], cube, 4, 6) # D
        self._setIndices(indices[4], cube, 4, 3) # F
        self._setIndices(indices[5], cube, 12, 3) # B

        opIdx = ops.index(op[0])
        if len(op) == 1: amountIdx = 0
        if len(op) == 2 and op[1] == '2': amountIdx = 1
        if len(op) == 2 and op[1] == 'i': amountIdx = 2
        y = torch.tensor([opIdx, amountIdx], dtype=torch.long)

        return indices, y

    def _setIndices(self, indices, cube, x0, y0):
        colors = "OYWGBR"
        c2Idx = {colors[i]: i for i in range(len(colors))}
        for y in range(3):
            for x in range(3):
                indices[y][x] = c2Idx[cube[y0 + y][x0 + x]]


class IndicesToOneHot:
    def __call__(self, xy):
        indices, y = xy
        eye = torch.eye(6, dtype=torch.float32)
        oneHot = eye[indices] # SHWC
        oneHot = oneHot.permute(0, 3, 1, 2) # SCHW
        return oneHot, y
