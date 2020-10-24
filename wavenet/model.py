"""
Wavenet https://arxiv.org/pdf/1609.03499.pdf
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Wavenet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = []
        for i in range(cfg.n_layers):
            block = ResBlock(cfg.n_channels, cfg.kernel_size, 2**i)
            self.layers.append(block)

        // FIXME check this should be same padding
        self.a1x1 = nn.Conv1d(
            cfg.n_channels,
            cfg.n_channels,
            kernel_size=1)

        self.b1x1 = nn.Conv1d(
            cfg.n_channels,
            cfg.n_channels,
            kernel_size=1)

    def forward(self, x):
        skips = []
        for l in self.layers:
            x = l(x)
            skips.append(x)

        x = F.relu(torch.cat(skips, 0).sum())
        x = F.relu(self.a1x1(x))
        x = self.b1x1(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, n_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Causal1d(
            n_channels * 2,
            n_channels * 2,
            kernel_size=kernel_size,
            dilation=dilation)

        self.end1x1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)

    def forward(self, x):
        return self.end1x1(F.glu(self.conv(x))) + x


class Causal1d(nn.Conv1d):

    def __init__(self, n_channels, kernel_size, dilation):
        super().__init__(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            dilation=dilation)

    def forward(self, x):
        super().forward(F.pad(x, (self.kernel_size - 1)))


class HParams:
    n_channels = 256
    n_layers = 10
    kernel_size = 5
