
import torch
from torch import nn
from torch.nn import functional as F

# Closely mimics DeepCube from SOLVING THE RUBIK’S CUBE WITH APPROXIMATE POLICY ITERATION
class DeepCube(nn.Module):
    def __init__(self, mul=1):
        super(DeepCube, self).__init__()
        self.linear1 = nn.Linear(6 * 6 * 3 * 3, round(mul * 4096))
        self.linear2 = nn.Linear(round(mul * 4096), round(mul * 2048))
        self.linear3 = nn.Linear(round(mul * 2048), round(mul * 512))
        self.linear4 = nn.Linear(round(mul * 512), 6 * 3)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.linear1(x), inplace=True)
        x = F.relu(self.linear2(x), inplace=True)
        x = F.relu(self.linear3(x), inplace=True)
        x = self.linear4(x)
        x = x.reshape(x.shape[0], 6, 3)
        return x
