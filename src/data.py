
import random, utils
from simulator.fastCube import Cube
import torch
from torch.utils.data import Dataset, DataLoader

class RubikEnv:
    def __init__(self, scrambles=1000):
        self.cube, _ = _getCube(scrambles)

    def step(self, action):
        op = utils.indicesToOp(action)
        self.cube.sequence(op)

    def getState(self):
        obs = str(self.cube)
        hsh = hash(obs)
        obs = utils.strToIndices(str(obs))
        obs = utils.indicesToOneHot(obs)

        done = self.cube.is_solved()
        return obs, done, hsh


class RubikDataset(Dataset):
    def __init__(self, size, maxIters):
        super(RubikDataset, self).__init__()
        self.size = size
        self.maxIters = maxIters

    def __getitem__(self, _):
        scrambles = random.randint(1, self.maxIters)
        x, y = _getCube(scrambles)
        x = utils.strToIndices(str(x))
        x = utils.indicesToOneHot(x)
        y = utils.opToIndices(y)
        return x, y, scrambles

    def __len__(self):
        return self.size


def _getCube(scrambles):
    cube = Cube()
    seq = ''
    lastOps = []
    for _ in range(scrambles):
        while True:
            op = random.choice(utils.ops)
            if op in lastOps: continue
            elif utils.invOps[op] in lastOps: lastOps.append(op)
            else: lastOps = [op]
            break
        amount = random.choice(['', 'i']) # normal op, inverse op
        seq += f' {op}{amount}'
    cube.sequence(seq)
    reverseAmount = {'': 'i', 'i': ''}[amount]
    return cube, f'{op}{reverseAmount}'
