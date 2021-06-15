"""Sample Generation.

Fast sampling as per https://arxiv.org/abs/1611.09482
"""

from collections import deque

import torch
import torch.cuda.amp as amp
import torch.nn as nn

from wavenet import utils, model, train, datasets


def fast(
    m: model.Wavenet,
    tf: datasets.Transforms,
    decoder,
    n_samples: int,
    batch_size: int = 1,
):
    "Process one sample at a time with a utils.decode_* decoder function."
    device = m.cfg.sampling_device()
    g = Generator(m).to(device)
    shape = (batch_size, m.cfg.n_audio_chans, 1)
    x = torch.zeros(shape, dtype=torch.float)
    x = x.to(device)
    with torch.set_grad_enabled(False):
        with amp.autocast(enabled=m.cfg.mixed_precision):
            track = []
            probabilities = []
            for i in range(n_samples):
                logits, _ = g.forward(x)
                probabilities.append(logits)
                y = decoder(logits)
                x = tf.normalise(y)
                track.append(y.detach().cpu())

            return (torch.cat(track, -1), torch.cat(probabilities, -1), g)


def simple(
    m: model.Wavenet,
    tf: datasets.NormaliseTransforms,
    decoder,
    n_samples: int,
    batch_size: int = 1,
):
    "Naïve sampling loop"
    device = m.cfg.sampling_device()
    m = m.to(device)
    shape = (batch_size, m.cfg.n_audio_chans, n_samples)
    y = torch.zeros(shape, dtype=torch.float)
    y += tf.mean  # this will be normalised back to zero
    y = y.to(device)
    with torch.set_grad_enabled(False):
        with amp.autocast(enabled=m.cfg.mixed_precision):
            probabilities = []
            for t in range(n_samples):
                x = tf.normalise(y)
                logits, _ = m(x)
                logits = logits[:, :, :, t].unsqueeze(-1)  # N, K, C, 1
                probabilities.append(logits)
                yt = decoder(logits)  # N, C, 1
                y[:, :, t] = yt.squeeze(-1)

            return y, torch.cat(probabilities, -1)


class Generator(model.Wavenet):
    """Memoize the wavenet for fast sampling using dynamic programming.

    Each forward pass expects one (N, C, W=1) time step only, and will emit
    the next (N, K, C, W=1) logits, where K is n_classes. Intermediate
    computations are remembered such that they do not have to be recomputed.
    The resulting output is the same as what would be obtained with a naive
    sampling loop.

    See the paper referenced at the top of the file for more details.
    """

    def __init__(self, m: model.Wavenet):
        assert m.cfg.kernel_size == 2, m.cfg.kernel_size
        super().__init__(m.cfg)
        self.cfg = m.cfg
        self.shifted = Memo(m.shifted)
        self.layers = nn.ModuleList([ResBlock(block) for block in m.layers])
        self.a1x1 = to_conv1d(m.a1x1)
        self.b1x1 = to_conv1d(m.b1x1)


class ResBlock(model.ResBlock):
    def __init__(self, r: model.ResBlock):
        super(model.ResBlock, self).__init__()
        self.conv = Memo(r.conv)
        self.res1x1 = to_conv1d(r.res1x1)
        self.skip1x1 = to_conv1d(r.skip1x1)


class Memo(nn.Module):
    """Memoize a dilated model.Causal1d.

    Can memoize a single past value, an arbitrary number of steps back. This
    is enough to implement a conv1d with a kernel size of two. The queue depth
    reflects the dilation factor of the memoized convolution.

    Instead of a single convolutional forward pass on an input x, e.g.
    (N, C, W=10), we split x into e.g. 10 x (N, C, W=1) inputs and call these
    one after the other. When we stack the outputs along W, we obtain the same
    result as with an ordinary forward pass with left padding of
    `queue_depth`, which is (kernel_size - 1) * dilation. Note that we can
    implement this with a non-dilated convolution here.

    Memo is implemented with a circular queue. To support the initial
    ShiftedCausal1d, we also have to delay inputs by one timestep. This is
    implemented with a second queue.

    Arguments
    ---------
    c: nnConv1d the convolution being memoized
    """

    def __init__(self, c: model.Causal1d):
        super().__init__()
        assert c.kernel_size[0] == 2, c.kernel_size
        self.c = to_conv1d(c)
        self.queue_depth = (c.kernel_size[0] - 1) * c.dilation[0]
        self.queue = deque([None] * self.queue_depth)

    def forward(self, x):
        assert x.shape[-1] == 1, x.shape  # one timestep
        x = self.pushpop(x)
        return self.c.forward(x)

    def pushpop(self, x):
        self.queue.append(x)
        memo = self.queue.popleft()
        memo = torch.zeros_like(x) if memo is None else memo
        return torch.cat([memo, x], -1)


def to_conv1d(x: nn.Conv1d):
    "Convert to non causal conv1d without padding or dilation"
    y = nn.Conv1d(x.in_channels, x.out_channels, x.kernel_size[0])
    with torch.no_grad():
        y.weight.copy_(x.weight)  # type: ignore
        y.bias.copy_(x.bias)  # type: ignore
        return y


def load(run_path: str):
    "Load config and model from wandb"
    p, ptrain = utils.load_wandb_cfg(run_path)
    p, ptrain = model.HParams(**p), train.HParams(**ptrain)
    return utils.load_chkpt(model.Wavenet(p), run_path), ptrain
