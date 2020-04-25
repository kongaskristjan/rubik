#!/usr/bin/python3

import torch, os, fire
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data, models, utils

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
maxScrambles = 20
modelDir = '../models'

def main(epochs=10, load_path=''):
    ds = data.RubikDataset(60000, maxScrambles)
    dl = DataLoader(ds, batch_size=64, num_workers=16)

    if load_path == '':
        net = models.DeepCube().to(device)
        train(net, dl, epochs)
    else:
        net = torch.load(load_path).to(device)
        testMulti(net, 20)

def train(net, dl, epochs):
    optim = torch.optim.Adam(net.parameters())
    criterion = utils.CubeLoss('none').to(device)

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
        testMulti(net, 20)
        if e % 50 == 0:
            os.makedirs(modelDir, exist_ok=True)
            filePath = f'{modelDir}/net.{e:04}.pt'
            print(f'Saving to {filePath}')
            torch.save(net, filePath)


def testMulti(net, maxScrambles):
    for scrambles in range(1, maxScrambles + 1):
        testSolve(net, scrambles=scrambles)


def testSolve(net, scrambles=100):
    env = data.RubikEnv(scrambles=scrambles)

    pastObsActions = {}
    for i in range(101):
        obs, done, hsh = env.getState()
        if done:
            print(f'Test with {scrambles} scrambles solved in {i} steps')
            return

        obs = obs.view(1, *obs.shape).to(device) # batch and GPU
        logits = net(obs)
        logits = logits.to('cpu').view(-1) # CPU and debatch

        while True: # Compute action
            action = torch.argmax(logits).item()
            pastActions = pastObsActions.setdefault(hsh, [])
            if action in pastActions: # If observation/action pair already done, avoid redoing
                logits[action] = -1000
            else: # Execute action
                pastActions.append(action)
                envAction = torch.tensor([action // 3, action % 3], dtype=torch.long)
                env.step(envAction)
                break


    print(f'did not solve {scrambles} scrambles')


if __name__ == '__main__':
    fire.Fire(main)
