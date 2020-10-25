"""
Wavenet https://arxiv.org/pdf/1609.03499.pdf
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Wavenet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input = CausalNoPresent1d(
            cfg.n_audio_chans,
            cfg.n_chans,
            kernel_size=cfg.kernel_size
        )

        self.layers = []
        for i in range(cfg.n_layers):
            block = ResBlock(cfg.n_chans, cfg.kernel_size, 2 ** i)
            self.layers.append(block)

        self.a1x1 = nn.Conv1d(cfg.n_chans, cfg.n_chans, kernel_size=1)
        self.b1x1 = nn.Conv1d(cfg.n_chans, cfg.n_chans, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.input(x))
        skips = []
        for l in self.layers:
            x = l(x)
            skips.append(x)

        x = F.relu(torch.stack(skips, 0).sum(0))
        x = F.relu(self.a1x1(x))
        x = self.b1x1(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, n_chans, kernel_size, dilation):
        super().__init__()

        self.conv = Causal1d(
            n_chans,
            n_chans * 2,
            kernel_size=kernel_size,
            dilation=dilation
        )

        self.end1x1 = nn.Conv1d(n_chans, n_chans, kernel_size=1)

    def forward(self, x):
        return self.end1x1(F.glu(self.conv(x), dim=1)) + x


class Causal1d(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, self.lpad()))

    def lpad(self):
        return (self.kernel_size[0] - 1) * self.dilation[0], 0


class CausalNoPresent1d(Causal1d):
    def forward(self, x):
        return super().forward(F.pad(x, (1, -1)))


class HParams:
    n_audio_chans = 2
    n_chans = 256
    n_layers = 10
    kernel_size = 2

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
