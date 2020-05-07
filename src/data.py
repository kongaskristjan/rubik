
import random, utils
from simulator.fastCube import Cube
import torch
from torch.utils.data import Dataset, DataLoader
import copy

class RubikEnv:
    def __init__(self, scrambles=1000):
        self.cube = _getCube(scrambles)

    def generateCombinations(self):
        keys, cubes = [], []
        for op in utils.ops:
            for amount in ('', 'i'):
                mutation = copy.deepcopy(self)
                step = f'{op}{amount}'
                keys.append(step)
                mutation.step(step)
                cubes.append(mutation)
        return keys, cubes

    def step(self, action):
        self.cube.sequence(action)

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
        x = _getCube(scrambles)
        x = utils.strToIndices(str(x))
        x = utils.indicesToOneHot(x)
        return x, scrambles

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
    return cube
