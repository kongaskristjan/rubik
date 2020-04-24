#!/usr/bin/python3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data, models, utils
from torchvision.transforms import Compose

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    tfms = Compose([data.CubeToIndices(), data.IndicesToOneHot()])
    ds = data.RubikDataset(600000, 20, tfms=tfms)
    dl = DataLoader(ds, batch_size=64, num_workers=16)

    net = models.DeepCube().to(device)
    train(net, dl)

def train(net, dl):
    optim = torch.optim.Adam(net.parameters())
    criterion = utils.CubeLoss().to(device)
    for input, target in dl:
        optim.zero_grad()
        input, target = input.to(device), target.to(device)
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        print(loss.item())


if __name__ == '__main__':
    main()
