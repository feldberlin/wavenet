"""
Wavenet https://arxiv.org/pdf/1609.03499.pdf
"""

import copy

from torch.nn import functional as F
import torch
import torch.cuda.amp as amp
import torch.nn as nn

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
            self.embed = InputEmbedding(cfg.n_classes, cfg.n_chans_embed)

        # hide the present time step t in the input
        # go from input straight to the network channel depth
        self.shifted = ShiftedCausal1d(
            cfg.n_input_chans(), cfg.n_chans, kernel_size=cfg.kernel_size
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

        # the final network in network dense layers
        self.a1x1 = nn.Conv1d(cfg.n_chans_skip, cfg.n_chans_end, kernel_size=1)
        self.b1x1 = nn.Conv1d(cfg.n_chans_end, cfg.n_logits(), kernel_size=1)

        # load a checkpoint
        if run_path:
            utils.load_chkpt(self, run_path)

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

            x = F.relu(self.shifted(x))  # N, C, W
            skips = 0
            for block in self.layers:
                x, s = block(x)
                skips += s

            x = F.relu(skips)
            x = F.relu(self.a1x1(x))
            x = self.b1x1(x)  # N, C, W in and out
            x = x.view(N, self.cfg.n_classes, self.cfg.n_audio_chans, W)

            loss = None
            if y is not None:
                loss = F.cross_entropy(x, y)

            return x, loss


class ResBlock(nn.Module):
    """ResBlock, as described in Figure 4 of the paper.

    These blocks use linear gating units instead of tan gating, as
    described in https://arxiv.org/pdf/1612.08083.pdf.
    """

    def __init__(self, dilation, cfg):
        super().__init__()

        self.conv = Causal1d(
            cfg.n_chans,
            cfg.n_chans_res * 2,
            kernel_size=cfg.kernel_size,
            dilation=dilation,
        )

        self.res1x1 = nn.Conv1d(cfg.n_chans_res, cfg.n_chans, kernel_size=1)
        self.skip1x1 = nn.Conv1d(
            cfg.n_chans_res, cfg.n_chans_skip, kernel_size=1
        )

    def forward(self, x):
        gated = F.glu(self.conv(x), dim=1)
        return (self.res1x1(gated) + x, self.skip1x1(gated))


class Causal1d(nn.Conv1d):
    """Causal 1d convolution.

    Left pads s.t. output timesteps depend only on past or present inputs. For
    example, with a kernel size 2 and a dilation of 1, we need to left pad by
    1, so that the first cross product at timestep t=0 uses only the left
    padded zero, and timestep t=0 as input. These units are used everywhere
    except for the first layer, and ensure that information flows directly
    upward from timestep t. The reason we don't left-pad by the full kernel
    size is that with many layers, output timestep t would no longer depend on
    the input at t-1. See `ShiftedCausal1d` for details on masking t.
    """

    def forward(self, x):
        kernel = self.kernel_size[0]
        dilation = self.dilation[0]
        x = F.pad(x, ((kernel - 1) * dilation, 0))
        return super().forward(x)


class ShiftedCausal1d(Causal1d):
    """Shift input one to the right then do a Causal1d convolution.

    Prepends 0 and removes the last element. This convolution can be used
    as an initial layer in a causal stack in order to remove a dependency on
    the current time step t, so we can model P(x_t|x_(t-1),x_(t-2)..., x_t0)
    instead of accidentally modelling P(x_t|x_t, x_(t-1)..., x_t0).
    """

    def forward(self, x):
        x = F.pad(x, (1, -1))
        return super().forward(x)


class InputEmbedding(nn.Embedding):
    """Embed each sample. Also works with stereo.

    N, C, W to N, C * cfg.n_audio_chans, W. Expects the target representation
    y as input. The channel dimension in the input represents mono or stereo.
    In the output, the channel is the embedding dimension.
    """

    def __init__(self, n_classes: int, n_dims: int):
        super().__init__(n_classes, n_dims, padding_idx=0)

    def forward(self, y):
        N, C, W = y.shape
        y = super().forward(y)  # embed into N, C, W, H=embedding_dim
        y = y.permute(0, 1, 3, 2)  # N, C, H, W
        return torch.reshape(y, (N, self.embedding_dim * C, W))  # fold stereo


class HParams(utils.HParams):

    # use mixed precision
    mixed_precision: bool = True

    # resample input dataset to sampling_rate before mu law compansion
    resample: bool = True

    # squashes stereo to mono. otherwise retain stereo in the input dataset
    squash_to_mono: bool = False

    # mu compress the input to n_classes
    compress: bool = True

    # length of a single track in samples
    sample_length: int = 16000

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
    n_chans: int = 32

    # number of embedding dimensions per stereo channel when embedding inputs
    n_chans_embed: int = 64

    # number of channels collected from each layer via `skip1x1`.
    n_chans_skip: int = 512

    # channel depth used in the residual branch of each res blocks. affects
    # the capacity of the conv and gating part in each resblock.
    n_chans_res: int = 32

    # final 1x1 convs at the very end of the network. use to reduce from
    # `n_chans_skip` capacity, down towards `n_classes`.
    n_chans_end: int = 128

    # random seed
    seed: int = 5762

    # run the generator on gpus
    sample_from_gpu: bool = True

    # used when setting the seed. this is an experimental torch feature
    use_deterministic_algorithms: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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

    def device(self):
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        return torch.device("cpu")

    def sampling_device(self):
        if self.sample_from_gpu and torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        return torch.device("cpu")

    def with_all_chans(self, n_chans: int):
        "Set all channel parameters to the same value"
        cfg = copy.copy(self)
        cfg.n_chans = n_chans
        cfg.n_chans_embed = n_chans
        cfg.n_chans_res = n_chans
        cfg.n_chans_skip = n_chans
        cfg.n_chans_end = n_chans
        return cfg

    def with_rescaled_chans(self, factor: float):
        "Set all channel parameters to the scaled value"
        cfg = copy.copy(self)
        cfg.n_chans = int(self.n_chans * factor)
        cfg.n_chans_embed = int(self.n_chans_embed * factor)
        cfg.n_chans_res = int(self.n_chans_res * factor)
        cfg.n_chans_skip = int(self.n_chans_skip * factor)
        cfg.n_chans_end = int(self.n_chans_end * factor)
        return cfg
