#!/usr/bin/python3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data
from torchvision.transforms import Compose

def main():
    tfms = Compose([data.CubeToIndices(), data.IndicesToOneHot()])
    trainDS = data.RubikDataset(60000, 20, tfms=tfms)
    testDS = data.RubikDataset(10000, 20, tfms=tfms)
    trainDL = DataLoader(trainDS, batch_size=64, num_workers=8)
    testDL = DataLoader(testDS, batch_size=64, num_workers=8)
    x, y = iter(trainDL).__next__()
    print(x.shape, y.shape)


if __name__ == '__main__':
    main()
