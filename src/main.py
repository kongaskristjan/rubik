#!/usr/bin/python3

import torch, os, fire, random
from torch import nn
from torch.cuda import amp
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
        net = models.DeepCube(mul=2).to(device)
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
    criterion = utils.CubeLoss('none').to(device)

    for e in range(start_epoch + 1, end_epoch + 1):
        stats = utils.Stats()
        perClass = utils.PerClassStats(maxScrambles)

        for input, target, scrambles in dl:
            optim.zero_grad()
            input, target = input.to(device), target.to(device)

            with amp.autocast():
                output = net(input)
                loss, acc = criterion(output, target)
                loss = torch.mean(loss)
            loss.backward()
            optim.step()
            stats.accumulate(len(target), loss, acc)
            perClass.accumulate(scrambles, loss, acc)

        print(f'Epoch {e}/{end_epoch}:')
        print(f'acc={100*stats.getAcc():.2f}%, loss={stats.getLoss():.3f}')
        print(f'acc= {perClass.accStr()}')
        print()
        if e % save_frequency == 0:
            os.makedirs(modelDir, exist_ok=True)
            filePath = getModelPath(e)
            print(f'Saving to {filePath}')
            torch.save(net, filePath)


def getModelPath(epoch):
    return f'{modelDir}/net.{epoch:04}.pt'


def testSolve(net, scrambles):
    env = data.RubikEnv(scrambles=scrambles)

    numSteps = 0
    for i in range(500): # Randomize if solving fails
        pastStates = set()
        for i in range(100): # Try solving for a short amount of time
            obs, done, hsh = env.getState()
            pastStates.add(hsh)
            if done:
                print(f'Test with {scrambles} scrambles solved in {numSteps} steps')
                return numSteps

            obs = obs.view(1, *obs.shape).to(device) # batch and GPU
            logits = net(obs)
            logits = logits.to('cpu').view(-1) # CPU and debatch

            while True: # Compute action
                action = torch.argmax(logits).item()
                envAction = torch.tensor([action // 2, action % 2], dtype=torch.long)
                env.step(envAction)
                hsh = env.getState()[2]
                if hsh in pastStates:
                    if logits[action] < -999:
                        break
                    logits[action] = -1000

                    envAction[1] = {0: 1, 1: 0}[envAction[1].item()] # invert action
                    env.step(envAction)
                else:
                    break
            numSteps += 1

        for j in range(20): # Randomize
            action = torch.LongTensor([random.randint(0, 5), random.randint(0, 1)])
            env.step(action)
            numSteps += 1

    print(f'did not solve {scrambles} scrambles')
    return scrambles


if __name__ == '__main__':
    fire.Fire(main)
