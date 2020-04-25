#!/usr/bin/python3

import torch, os, fire
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data, models, utils

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
maxScrambles = 20
modelDir = '../models'

def main(start_epoch=0, end_epoch=1000, mode='train'):
    assert mode in ('train', 'validate', 'test')
    ds = data.RubikDataset(60000, maxScrambles)
    dl = DataLoader(ds, batch_size=64, num_workers=16)

    if start_epoch == 0:
        net = models.DeepCube().to(device)
    else:
        net = torch.load(getModelPath(start_epoch)).to(device)

    if mode == 'train':
        train(net, dl, start_epoch, end_epoch)
    if mode == 'validate':
        validate(net, maxScrambles=20)
    if mode == 'test':
        for i in range(1000):
            testSolve(net, scrambles=1000)

def train(net, dl, start_epoch, end_epoch):
    optim = torch.optim.Adam(net.parameters())
    criterion = utils.CubeLoss('none').to(device)

    for e in range(start_epoch + 1, end_epoch + 1):
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
        print(f'Epoch {e}/{end_epoch}:')
        print(f'acc={100*stats.getAcc():.2f}%, loss={stats.getLoss():.3f}')
        print(f'acc= {perClass.accStr()}')
        print()
        validate(net, 20)
        if e % 50 == 0:
            os.makedirs(modelDir, exist_ok=True)
            filePath = getModelPath(e)
            print(f'Saving to {filePath}')
            torch.save(net, filePath)


def validate(net, maxScrambles):
    for scrambles in range(1, maxScrambles + 1):
        testSolve(net, scrambles=scrambles)


def getModelPath(epoch):
    return f'{modelDir}/net.{epoch:04}.pt'


def testSolve(net, scrambles):
    env = data.RubikEnv(scrambles=scrambles)

    pastObsActions = {}
    for i in range(501):
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
                if logits[action] < -999:
                    break
                logits[action] = -1000
            else: # Execute action
                pastActions.append(action)
                envAction = torch.tensor([action // 3, action % 3], dtype=torch.long)
                env.step(envAction)
                break


    print(f'did not solve {scrambles} scrambles')


if __name__ == '__main__':
    fire.Fire(main)
