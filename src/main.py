#!/usr/bin/python3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data, models, utils
from torchvision.transforms import Compose

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
maxScrambles = 20

def main():
    ds = data.RubikDataset(60000, maxScrambles)
    dl = DataLoader(ds, batch_size=64, num_workers=16)

    net = models.DeepCube().to(device)
    train(net, dl)

def train(net, dl):
    optim = torch.optim.Adam(net.parameters())
    criterion = utils.CubeLoss('none').to(device)

    epochs = 10
    for e in range(1, epochs + 1):
        stats = utils.Stats()
        perClass = utils.PerClassStats(maxScrambles)

        for input, target, scrambles in dl:
            optim.zero_grad()
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss, acc = criterion(output, target)
            torch.mean(loss).backward()
            optim.step()
            stats.accumulate(len(target), loss, acc)
            perClass.accumulate(scrambles, loss, acc)

        print()
        print()
        print()
        print(f'Epoch {e}/{epochs}:')
        print(f'acc={100*stats.getAcc():.2f}%, loss={stats.getLoss():.3f}')
        print(f'acc= {perClass.accStr()}')
        print()
        for scrambles in range(1, 11):
            testSolve(net, scrambles=scrambles)


def testSolve(net, scrambles=100):
    env = data.RubikEnv(scrambles=scrambles)
    for i in range(101):
        obs, done = env.getState()
        if done:
            print(f'Test with {scrambles} scrambles solved in {i} steps')
            return

        obs = obs.view(1, *obs.shape).to(device) # batch and GPU
        action = net(obs)
        action = action.to('cpu').view(*action.shape[1:]) # CPU and debatch

        action = torch.argmax(action.view(-1))
        action = torch.tensor([action.item() // 3, action.item() % 3], dtype=torch.long)
        env.step(action)
    print(f'did not solve {scrambles} scrambles')


if __name__ == '__main__':
    main()