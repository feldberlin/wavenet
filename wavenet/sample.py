"""
Fast wavenet generation algorithm https://arxiv.org/abs/1611.09482
"""

from collections import deque

import torch
import torch.nn as nn
from torch.nn import functional as F

from wavenet import utils, model


def sample(m: model.Wavenet, cfg: model.HParams,
           n_samples: int, batch_size: int = 1):

    g = Generator(m)
    track = torch.zeros((batch_size, cfg.n_audio_chans, 1))
    sample = track
    for i in range(n_samples):
        logits, _ = g.forward(sample)
        sample = utils.sample_from_logits(logits)
        track = torch.cat([track, sample], -1)

    return track


class Generator(model.Wavenet):

    def __init__(self, m: model.Wavenet):
        super().__init__(m.cfg)
        self.cfg = m.cfg
        self.input = Memo(to_conv1d(m.input), m.cfg.kernel_size - 1)
        self.layers = nn.ModuleList([ResBlock(l) for l in m.layers])
        self.a1x1 = to_conv1d(m.a1x1)
        self.b1x1 = to_conv1d(m.b1x1)


class ResBlock(model.ResBlock):

    def __init__(self, r: model.ResBlock):
        super(model.ResBlock, self).__init__()
        self.conv = Memo(to_conv1d(r.conv), r.conv.dilation[0])
        self.end1x1 = to_conv1d(r.end1x1)


class Memo(nn.Module):

    def __init__(self, c: nn.Conv1d, depth=1):
        super().__init__()
        self.c = c
        self.queue = deque([None] * depth)

    def forward(self, x):
        self.queue.append(x)
        memo = self.queue.popleft()
        memo = torch.zeros_like(x) if memo is None else memo
        x = torch.cat([memo, x], -1)
        return self.c.forward(x)


def to_conv1d(x: nn.Conv1d):
    "Convert to non causal conv1d without padding or dilation"
    y = nn.Conv1d(x.in_channels, x.out_channels, x.kernel_size[0])
    y.weight.detach().copy_(x.weight)
    return y
