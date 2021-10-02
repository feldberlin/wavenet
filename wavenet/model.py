"""
Wavenet https://arxiv.org/pdf/1609.03499.pdf
"""

import hashlib

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.nn import functional as F

from wavenet import utils


class Wavenet(nn.Module):
    """An implementation of the original Wavenet paper, but no conditioning.

    Cross entropy requires ground truth as 8 bit mu law companded quantized
    audio in e.g. [-128, 127], but shifted e.g. to [0, 255]. This net is
    using a categorical loss, and logits will be argmaxed to obtain
    predictions. Network input is 8 bit mu law companded normalised audio.
    """

    def __init__(self, cfg, run_path=None):
        super().__init__()
        self.cfg = cfg

        # embed inputs. not documented in the wavenet paper, see gh issue #2
        if cfg.embed_inputs:
            self.embed = InputEmbedding(
                cfg.n_classes, cfg.n_chans_embed, cfg.n_audio_chans
            )

        # hide the present time step t in the input
        # go from input straight to the network channel depth
        self.shifted = Conv1d(
            cfg.n_input_chans(),
            cfg.n_chans,
            kernel_size=cfg.kernel_size,
            causal=True,
            shifted=True,
            relu=False,
            bn=cfg.batch_norm,
        )

        # residual blocks, as described in Figure 4
        # a single context stack with 1, 2, 4... dilations
        self.layers = nn.ModuleList(
            [
                ResBlock(2 ** i, cfg)
                for _ in range(cfg.dilation_stacks)
                for i in range(cfg.n_layers)
            ]
        )

        # net-in-net 1x1 dense layer
        self.a1x1 = Conv1d(
            cfg.n_chans_skip,
            cfg.n_chans_end,
            kernel_size=1,
            relu=True,
            bn=cfg.batch_norm,
        )

        # net-in-net 1x1 to logits
        self.b1x1 = Conv1d(
            cfg.n_chans_end,
            cfg.n_logits(),
            kernel_size=1,
            relu=False,
            bn=cfg.batch_norm,
        )

        # load a checkpoint
        if run_path:
            utils.restore(self, run_path)

    def forward(self, x, y=None):
        """Audio is trained on (N, C, W) batches.

        There are C stereo input channels, W samples in each example. Logits
        are produced in (N, K, C, W) form, where K is the number of classes
        as determined by the audio bit depth.
        """

        with amp.autocast(enabled=self.cfg.mixed_precision):
            N, C, W = x.shape  # N, C=self.cfg.n_input_chans(), W

            # embed each categorical sample
            if self.cfg.embed_inputs:
                x = self.embed(y)  # N, C, W

            # hide the present
            x = self.shifted(x)  # N, C, W

            # residual
            skips = 0
            for block in self.layers:
                x, s = block(x)
                skips += s
            x = F.relu(skips)

            # dense
            x = self.a1x1(x)
            x = self.b1x1(x)  # N, C, W in and out
            x = x.view(N, self.cfg.n_classes, self.cfg.n_audio_chans, W)

            loss = None
            if y is not None:
                loss = F.cross_entropy(x, y)

            return x, loss


class InputEmbedding(nn.Embedding):
    """Embed each sample. Also works with stereo.

    N, C, W to N, C * cfg.n_audio_chans, W. Expects the target representation
    y as input. The channel dimension in the input represents mono or stereo.
    In the output, the channel is the embedding dimension.
    """

    def __init__(self, n_classes: int, n_dims: int, n_audio_chans: int):
        super().__init__(n_classes, n_dims, padding_idx=0)  # see gh issue #3

    def forward(self, y):
        N, C, W = y.shape
        y = super().forward(y)  # embed into N, C, W, H=embedding_dim
        y = y.permute(0, 1, 3, 2)  # N, C, H, W
        return torch.reshape(y, (N, self.embedding_dim * C, W))  # fold stereo


class Conv1d(nn.Conv1d):
    """Conv1d with causal and shifted modes.

    Causal 1d convolution

    Left pads s.t. output timesteps depend only on past or present inputs. For
    example, with a kernel size 2 and a dilation of 1, we need to left pad by
    1, so that the first cross product at timestep t=0 uses only the left
    padded zero, and timestep t=0 as input. These units are used everywhere
    except for the first layer, and ensure that information flows directly
    upward from timestep t. The reason we don't left-pad by the full kernel
    size is that with many layers, output timestep t would no longer depend on
    the input at t-1. See below for details on masking t.

    Shifted 1d convolution

    Shift input one to the right then do a causal convolution.
    Shifting prepends 0 and removes the last element. This convolution can be
    used as an initial layer in a causal stack in order to remove a dependency
    on the current time step t, so we can model P(x_t|x_(t-1),x_(t-2)...,
    x_t0) instead of accidentally modelling P(x_t|x_t, x_(t-1)..., x_t0).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        causal: bool = False,
        shifted: bool = False,
        relu: bool = False,
        bn: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=not bn,
        )
        self.shifted = shifted
        self.causal = causal
        self.relu = relu
        self.bn = bn
        if bn:
            self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        kernel = self.kernel_size[0]
        dilation = self.dilation[0]

        if self.shifted:
            # remove dependency on current timestep
            x = F.pad(x, (1, -1))

        if self.shifted or self.causal:
            # left pad to depend only on past or current timesteps
            x = F.pad(x, ((kernel - 1) * dilation, 0))

        # run the layer
        x = super().forward(x)
        if self.bn:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x  # N, C, W


class ResBlock(nn.Module):
    """ResBlock, as described in Figure 4 of the paper.

    These blocks use linear gating units instead of tan gating, as
    described in https://arxiv.org/pdf/1612.08083.pdf.
    """

    def __init__(self, dilation, cfg):
        super().__init__()

        # double sized output channels for glu
        self.conv = Conv1d(
            cfg.n_chans,
            cfg.n_chans_res * 2,
            kernel_size=cfg.kernel_size,
            dilation=dilation,
            causal=True,
            relu=False,
            bn=cfg.batch_norm,
        )

        # back up to n_chans, as input to this module
        self.res1x1 = Conv1d(
            cfg.n_chans_res,
            cfg.n_chans,
            kernel_size=1,
            relu=False,
            bn=cfg.batch_norm,
        )

        # not explicitly described in the paper
        self.skip1x1 = Conv1d(
            cfg.n_chans_res,
            cfg.n_chans_skip,
            kernel_size=1,
            relu=False,
            bn=cfg.batch_norm,
        )

    def forward(self, x):
        gated = F.glu(self.conv(x), dim=1)
        return self.res1x1(gated) + x, self.skip1x1(gated)


class HParams(utils.HParams):

    # use mixed precision
    mixed_precision: bool = True

    # resample input dataset to sampling_rate before mu law compansion
    resample: bool = True

    # see `librosa.resample`. trades off speed and quality
    resampling_method: str = "soxr_hq"

    # squashes stereo to mono. otherwise retain stereo in the input dataset
    squash_to_mono: bool = False

    # mu compress the input to n_classes
    compress: bool = True

    # length of a single track in samples
    sample_length: int = 16000

    # length of the overlap between two examples in the audio
    sample_overlap_length = 0

    # set sample_overlap_length to self.receptive_field_size()
    sample_overlap_receptive_field = False

    # stereo, mono
    n_audio_chans: int = 2

    # audio sampling rate
    sampling_rate: int = 16000

    # map each input sample to an embedding in the channel domain
    embed_inputs = False

    # sample bit depth
    n_classes: int = 2 ** 8

    # layers per dilation stack in a single context stack
    n_layers: int = 11

    # convolution kernel size
    kernel_size: int = 2

    # number of repeated dilation patterns in a single context stack,
    # e.g. 1, 2, 4...128, 1, 2, 4...128 is 2 dilation stacks.
    dilation_stacks: int = 3

    # used from the input conv and between layers. affects how much per sample
    # information gets passed between each layer in a stack.
    n_chans: int = 128

    # number of embedding dimensions per stereo channel when embedding inputs
    n_chans_embed: int = 256

    # number of channels collected from each layer via `skip1x1`.
    n_chans_skip: int = 256

    # channel depth used in the residual branch of each res blocks. affects
    # the capacity of the conv and gating part in each resblock.
    n_chans_res: int = 96

    # final 1x1 convs at the very end of the network. use to reduce from
    # `n_chans_skip` capacity, down towards `n_classes`.
    n_chans_end: int = 256

    # random seed
    seed: int = 5763

    # run the generator on gpus
    sample_from_gpu: bool = True

    # used when setting the seed. this is an experimental torch feature
    use_deterministic_algorithms: bool = False

    # use batch norm layers
    batch_norm: bool = False

    # leader compute device
    device: torch.device = torch.device("cpu")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if torch.cuda.is_available():
            self.device = torch.device(torch.cuda.current_device())

        if self.sample_overlap_receptive_field:
            self.sample_overlap_length = self.receptive_field_size()

    def n_logits(self):
        return self.n_classes * self.n_audio_chans

    def n_input_chans(self):
        return self.n_embed_dims() if self.embed_inputs else self.n_audio_chans

    def n_embed_dims(self):
        return self.n_chans_embed * self.n_audio_chans

    def receptive_field_size(self):
        return self.dilation_stacks * 2 ** self.n_layers

    def receptive_field_size_ms(self):
        return 1000 * self.receptive_field_size() / self.sampling_rate

    def sample_size_ms(self):
        return 1000 * self.sample_length / self.sampling_rate

    def sample_hop_length(self):
        return self.sample_length - self.sample_overlap_length

    def sampling_device(self):
        if self.sample_from_gpu and torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        return torch.device("cpu")

    def audio_cache_key(self):
        key = "/".join(
            [
                str(self.resample),
                str(self.resampling_method),
                str(self.squash_to_mono),
                str(self.compress),
                str(self.sampling_rate)
            ]
        )
        return hashlib.md5(key.encode()).hexdigest()

    def with_all_chans(self, n_chans: int):
        "Set all channel parameters to the same value"
        cfg = self.clone()
        cfg.n_chans = n_chans
        cfg.n_chans_embed = n_chans
        cfg.n_chans_res = n_chans
        cfg.n_chans_skip = n_chans
        cfg.n_chans_end = n_chans
        return cfg
