#!/usr/bin/python3

import torch, os, fire, random, operator
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data, models, utils

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
maxScrambles = 20
modelDir = '../models'

def main(start_epoch=0, end_epoch=1000, save_frequency=50, mode='train'):
    print(f'Using device {device}')
    assert mode in ('train', 'test')
    assert save_frequency > 0

    ds = data.RubikDataset(60000, maxScrambles)
    dl = DataLoader(ds, batch_size=64, num_workers=16)

    if start_epoch == 0:
        net = models.DeepCube(mul=1).to(device)
    else:
        net = torch.load(getModelPath(start_epoch), map_location=device)

    if mode == 'train':
        train(net, dl, start_epoch, end_epoch, save_frequency)
    if mode in ('train', 'test'):
        sum, numSolves = 0, 50
        for i in range(numSolves):
            sum += testSolve(net, scrambles=random.randint(1000, 1099))
        print(f'Average solution steps: {sum / numSolves}')

def train(net, dl, start_epoch, end_epoch, save_frequency):
    optim = torch.optim.Adam(net.parameters())
    criterion = nn.MSELoss(reduction='none')

    for e in range(start_epoch + 1, end_epoch + 1):
        stats = utils.Stats()
        perClass = utils.PerClassStats(maxScrambles)

        for input, scrambles in dl:
            optim.zero_grad()
            input, scrambles = input.to(device), scrambles.to(device)
            output = net(input)
            loss = criterion(output, scrambles.float())
            torch.mean(loss).backward()
            optim.step()
            stats.accumulate(len(scrambles), loss)
            perClass.accumulate(scrambles, loss)

        print(f'Epoch {e}/{end_epoch}:')
        print(f'dist error = {stats.getLoss()**0.5:.3f}')
        print(f'dist error = ({perClass.distStr()})')
        print()
        if e % save_frequency == 0:
            os.makedirs(modelDir, exist_ok=True)
            filePath = getModelPath(e)
            print(f'Saving to {filePath}')
            torch.save(net, filePath)


def getModelPath(epoch):
    return f'{modelDir}/net.{epoch:04}.pt'


def testSolve(net, scrambles):
    cube = data.RubikEnv(scrambles=scrambles)

    numSteps = 0
    for i in range(500): # Randomize if solving fails
        pastStates = set()
        for i in range(100): # Try solving for a short amount of time
            cubes, dists, dones, hashes = getCubesDistsDonesHashes(net, cube)
            if dones['']:
                print(f'Test with {scrambles} scrambles solved in {numSteps} steps')
                return numSteps

            pastStates.add(hashes[''])
            for key in cubes:
                if hashes[key] in pastStates:
                    dists[key] = 1e6 # Already been here
                if dones[key]:
                    dists[key] = -1e6 # Solved

            minKey = min(dists.items(), key=lambda x: untorch(x[1]))[0]
            cube = cubes[minKey]
            numSteps += 1

        for j in range(20): # Randomize
            indices = torch.LongTensor([random.randint(0, 5), random.randint(0, 1)])
            action = data._indicesToOp(indices)
            cube.step(action)
            numSteps += 1

    print(f'did not solve {scrambles} scrambles')
    return numSteps


def getCubesDistsDonesHashes(net, cube):
    keys, cubes = cube.generateCombinations()
    keys, cubes = [''] + keys, [cube] + cubes
    obs, dones, hashes = zip(*[c.getState() for c in cubes])
    obs = torch.stack(obs, dim=0)
    obs = obs.to(device)  # batch and GPU
    dists = net(obs)
    dists = dists.to('cpu').view(-1)  # CPU and debatch
    cubes, dists, dones, hashes = (dict(zip(keys, i)) for i in (cubes, dists, dones, hashes))

    return cubes, dists, dones, hashes


def untorch(x):
    if type(x) == torch.Tensor:
        return x.item()
    return x


if __name__ == '__main__':
    fire.Fire(main)
