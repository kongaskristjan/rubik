
import torch
from torch import nn
from torch.nn import functional as F

# Closely mimics DeepCube from SOLVING THE RUBIK’S CUBE WITH APPROXIMATE POLICY ITERATION
class DeepCube(nn.Module):
    def __init__(self):
        super(DeepCube, self).__init__()
        self.linear1 = nn.Linear(6 * 6 * 3 * 3, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 512)
        self.linear4 = nn.Linear(512, 6 * 3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x), inplace=True)
        x = F.relu(self.linear2(x), inplace=True)
        x = F.relu(self.linear3(x), inplace=True)
        x = self.linear4(x)
        x = x.view(x.shape[0], 6, 3)
        return x
