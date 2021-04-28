"""
Fast wavenet generation algorithm https://arxiv.org/abs/1611.09482
"""

from collections import deque

import torch
import torch.nn as nn
import torch.cuda as cuda

from wavenet import utils, model, train, audio


def load(run_path):
    "Load config and model from wandb"
    p, ptrain = utils.load_wandb_cfg(run_path)
    p, ptrain = model.HParams(**p), train.HParams(**ptrain)
    return utils.load_chkpt(model.Wavenet(p), run_path), ptrain


def sample(m: model.Wavenet, decoder, n_samples: int, batch_size: int = 1):
    "Sample with the given utils.decode_* decoder function."
    g, device = Generator(m).to_device()
    sample = torch.zeros((batch_size, m.cfg.n_audio_chans, 1), dtype=torch.long)
    sample = sample.to(device)

    # one sample only, at a time, from and into the memoised network
    with torch.set_grad_enabled(False):
        track = None
        for i in range(n_samples):
            logits, _ = g.forward(sample.float())
            idxs = decoder(logits)
            sample = utils.audio_from_class_idxs(idxs, m.cfg.n_classes)
            if track is not None:
                track = torch.cat([track, sample.detach().cpu()], -1)
            else:
                track = sample.detach().cpu()

        dequantised = audio.dequantise(track.numpy(), m.cfg)
        expanded = audio.mu_expand(dequantised, m.cfg)
        return track, expanded


class Generator(model.Wavenet):
    """Memoize the wavenet for fast sampling.

    Observe that a dilated convolution is the same as a non dilated
    convolution if the input is subsampled with the given stride. For any
    given convolution, we process the last known timestep, plus kernel_size
    - 1 older ones. These older inputs are fetched from a rotating queue,
    which is sized such that subsampling with the given kernel_size and
    dilation will be possible. Note also that cnns are not fixed size, so we
    can pass (N, C, 1) batches in without problems.

    This implementation currently left pops off the queue only, which means
    that only kernel size 2 is currently supported.

    Each forward pass expects one (N, C, W=1) time step only, and will emit
    the next (N, K, C, W=1) logits, where K is n_classes.
    """

    def __init__(self, m: model.Wavenet):
        assert m.cfg.kernel_size == 2
        super().__init__(m.cfg)
        self.cfg = m.cfg
        self.input = Memo(to_conv1d(m.input), m.cfg.kernel_size - 1)
        self.layers = nn.ModuleList([ResBlock(block) for block in m.layers])
        self.a1x1 = to_conv1d(m.a1x1)
        self.b1x1 = to_conv1d(m.b1x1)

    def to_device(self):
        if self.cfg.sample_from_gpu and cuda.is_available():
            device = cuda.current_device()
            return self.to(device), device
        return self, 'cpu'


class ResBlock(model.ResBlock):

    def __init__(self, r: model.ResBlock):
        super(model.ResBlock, self).__init__()
        self.conv = Memo(to_conv1d(r.conv), r.conv.dilation[0])
        self.end1x1 = to_conv1d(r.end1x1)
        self.skip1x1 = to_conv1d(r.skip1x1)


class Memo(nn.Module):

    def __init__(self, c: nn.Conv1d, depth=1):
        super().__init__()
        self.c = c
        self.queue = deque([None] * depth)

    def forward(self, x):
        assert x.shape[-1] == 1  # one timestep
        return self.c.forward(self.pushpop(x))

    def pushpop(self, x):
        self.queue.append(x)
        memo = self.queue.popleft()
        memo = torch.zeros_like(x) if memo is None else memo
        return torch.cat([memo, x], -1)


def to_conv1d(x: nn.Conv1d):
    "Convert to non causal conv1d without padding or dilation"
    y = nn.Conv1d(x.in_channels, x.out_channels, x.kernel_size[0])
    with torch.no_grad():
        y.weight.copy_(x.weight)
        y.bias.copy_(x.bias)
        return y
