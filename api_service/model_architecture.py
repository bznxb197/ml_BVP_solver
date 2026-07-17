import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16, hidden=256):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden)
        )
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.res3 = ResidualBlock(hidden)
        self.out_layer = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.out_layer(x)