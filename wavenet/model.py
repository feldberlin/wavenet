"""
Wavenet https://arxiv.org/pdf/1609.03499.pdf
"""

import torch.nn as nn
from torch.nn import functional as F
import torch.cuda.amp as amp

from wavenet import utils


class Wavenet(nn.Module):
    """An implementation of the original Wavenet paper, but no conditioning.

    Cross entropy requires ground truth as 8 bit mu law companded quantized
    audio in e.g. [-128, 127], but shifted e.g. to [0, 255]. This net is
    using a categorical loss, and logits will be argmaxed to obtain
    predictions. Network input is 8 bit mu law companded quantized audio
    in [-128, 127].
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # hide the present time step t in the input
        # go from stereo straight to the network channel depth
        self.input = CausalShifted1d(
            cfg.n_audio_chans,
            cfg.n_chans,
            kernel_size=cfg.kernel_size
        )

        # residual blocks, as described in Figure 4
        # a single context stack with 1, 2, 4... dilations
        self.layers = nn.ModuleList([
            ResBlock(cfg.n_chans, cfg.kernel_size, 2 ** i)
            for i in range(cfg.n_layers)
            for _ in range(cfg.dilation_stacks)
        ])

        # the final network in network dense layers
        self.a1x1 = nn.Conv1d(cfg.n_chans, cfg.n_chans, kernel_size=1)
        self.b1x1 = nn.Conv1d(cfg.n_chans, cfg.n_logits(), kernel_size=1)

    def forward(self, audio):
        """Audio is trained on (N, C, W) batches.

        There are C stereo input channels, W samples in each example. Logits
        are produced in (N, K, C, W) form, where K is the number of classes
        as determined by the audio bit depth.
        """

        with amp.autocast(enabled=self.cfg.mixed_precision):
            N, C, W = audio.shape
            x = utils.quantized_audio_to_unit_loudness(audio, self.cfg)
            x = F.relu(self.input(x))
            skips = 0
            for block in self.layers:
                x = block(x)
                skips += x

            x = F.relu(skips)
            x = F.relu(self.a1x1(x))
            x = self.b1x1(x)
            x = x.view(N, self.cfg.n_classes, C, W)
            y = utils.quantized_audio_to_class_idxs(audio, self.cfg)

            return x, F.cross_entropy(x, y)


class ResBlock(nn.Module):
    """ResBlock, as described in Figure 4 of the paper.

    These blocks use linear gating units instead of tan gating, as
    described in https://arxiv.org/pdf/1612.08083.pdf.
    """

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
    """Causal 1d convolution.

    Left pads s.t. output timesteps depend only on past or present inputs.
    """

    def forward(self, x):
        kernel = self.kernel_size[0]
        dilation = self.dilation[0]
        return super().forward(F.pad(x, ((kernel - 1) * dilation, 0)))


class CausalShifted1d(Causal1d):
    """Shift input one to the right then do a Causal1d convolution.

    Prepends 0 and removes the last element. This convolution can be used
    as an initial layer in a causal stack in order to remove a dependency
    on the current time step t, so we can model P(x_t|x_(t-1),x_(t-2)..., x_t0)
    instead of accidentally modelling P(x_t|x_t, x_(t-1)..., x_t0).
    """

    def forward(self, x):
        return super().forward(F.pad(x, (1, -1)))


class HParams(utils.HParams):

    # use mixed precision
    mixed_precision = True

    # retain stereo in the input dataset. otherwise squashes to mono
    stereo = True

    # resample input dataset to sampling_rate before mu law compansion
    resample = True

    # length of a single track in samples
    sample_length = 16000

    # stereo, mono
    n_audio_chans = 2

    # audio sampling rate
    sampling_rate = 16000

    # sample bit depth
    n_classes = 2**8

    # conv channels used throughout
    n_chans = 256

    # layers per dilation stack in a single context stack
    n_layers = 10

    # convolution kernel size
    kernel_size = 2

    # number of repeated dilation patterns in a single context stack,
    # e.g. 1, 2, 4...128, 1, 2, 4...128 is 2 dilation stacks.
    dilation_stacks = 3

    # random seed
    seed = 5762

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def n_logits(self):
        return self.n_classes * self.n_audio_chans

    def receptive_field_size(self):
        return self.dilation_stacks * 2**self.n_layers

    def receptive_field_size_ms(self):
        return 1000 * self.receptive_field_size() / self.sampling_rate
