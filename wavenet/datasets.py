# Datasets and transforms.
#
# Each element is a training frame in two versions x, y. Both represent the
# same data series, where x is normalised and in the range [-1, 1], while y is
# the integral class index of the value at that point.
#
# The same transforms are applied to all audio datasets, and these assume that
# the dataset emits quantised integral x, y tensors.


import abc
import collections

import numpy as np  # type: ignore
import torch

from wavenet import utils, audio


# data transformations


class Transforms:
    @abc.abstractmethod
    def __call__(self, data):
        "Convert from data to x, y for training"
        raise NotImplementedError

    @abc.abstractmethod
    def normalise(self, y):
        "Convert from y back to x for autoregressive generation"
        raise NotImplementedError


class AudioUnitTransforms(Transforms):
    """Normalise x by dequantising data, convert y to class indices by shift.
    Transforms assume quantised data input.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = 0.0

    def __call__(self, data):
        x = audio.dequantise(data, self.cfg)
        y = utils.audio_to_class_idxs(data, self.cfg.n_classes)
        return x, y

    def normalise(self, y):
        x = utils.audio_from_class_idxs(y, self.cfg.n_classes)
        x = audio.dequantise(x, self.cfg)
        return x


class NormaliseTransforms(Transforms):
    """Normalise x, convert y to class indices.
    Transforms assume quantised x, y inputs.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.eps = 1e-15

    def __call__(self, data):
        y = data
        x = (y - self.mean) / (self.std + self.eps)
        return x, y

    def normalise(self, y):
        return (y - self.mean) / (self.std + self.eps)


# datasets


class Dataset(torch.utils.data.Dataset, collections.abc.Sequence):
    "Map style datasets, but also iterable."

    @property
    @abc.abstractmethod
    def transforms(self) -> Transforms:
        raise NotImplementedError

    def sample(self):
        i = torch.randint(len(self), (1,))
        return i, self[i]


def to_tensor(d: Dataset, n_items=None):
    "Materialize the whole dataset"
    n_items = n_items if n_items else len(d)
    return (
        torch.stack([d[i][0] for i in range(n_items)]),
        torch.stack([d[i][1] for i in range(n_items)]),
    )


def tracks(filename: str, validation_pct: float, p):
    "Train - validation split on a single track"
    return (
        Track(filename, p, 0.0, 1 - validation_pct),
        Track(filename, p, 1 - validation_pct, 1.0),
    )


class Track(Dataset):
    """Dataset constructed from a single track, held in memory.
    Loads μ compressed slices from a single track into N, C, W in [-1., 1.].
    """

    def __init__(self, filename: str, p, start: float = 0.0, end: float = 1.0):
        self.tf = AudioUnitTransforms(p)
        self.filename = filename
        y = audio.load_resampled(filename, p)  # audio data in [-1, 1]
        _, n_samples = y.shape
        y = y[:, int(n_samples * start) : int(n_samples * end)]  # start to end
        y = audio.to_librosa(y)  # different mono channels for librosa
        ys = audio.frame(y, p)  # cut frames from single track
        ys = np.moveaxis(ys, -1, 0)  # reshape back to N, C, W
        ys = torch.tensor(ys, dtype=torch.float32)
        ys = ys[1:, :, :]  # trim hoplength leading silence
        if p.compress:
            ys = audio.mu_compress_batch(ys, p)
        self.data = audio.quantise(ys, p)  # from [-1, 1] to [-n, n+1]

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.tf(self.data[idx])

    def __repr__(self):
        return f"Track({self.filename})"


class StereoImpulse(Dataset):
    """Left and right at t0 are both binomial, modes slightly apart.
    Batch of m impulses at t0, followed by n zero samples, held in memory.
    """

    def __init__(self, n, m, cfg, probs=None):
        self.tf = AudioUnitTransforms(cfg)
        probs = probs if probs else (0.45, 0.55)
        data = np.random.binomial(
            (cfg.n_classes, cfg.n_classes), probs, (n, cfg.n_audio_chans)
        )

        data = utils.audio_from_class_idxs(data, cfg.n_classes)
        data_batched = np.reshape(data, (n, cfg.n_audio_chans, 1))
        data_batched = np.pad(data_batched, ((0, 0), (0, 0), (0, m - 1)))
        self.data = torch.from_numpy(data_batched)

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.tf(self.data[idx])

    def __repr__(self):
        return "StereoImpulse()"


class Sines(Dataset):
    """Each sample is a μ compressed sine wave with random amp and phase.
    Random amplitude and hz, unless given.
    """

    def __init__(
        self,
        n_examples,
        cfg,
        amp: float = None,
        hz: float = None,
        phase: float = None,
        minhz=400,
        maxhz=20000,
    ):

        # config
        self.n_seconds = cfg.sample_size_ms() / 1000
        self.n_examples = n_examples
        self.cfg = cfg

        # normalisations
        self.tf = AudioUnitTransforms(cfg)

        # amp
        default_amps = torch.rand(n_examples)
        self.amp = amp if amp is not None else default_amps

        # hz
        default_hzs = torch.rand(n_examples) * maxhz + minhz
        self.hz = hz if hz is not None else default_hzs

        # phase
        default_phases = torch.rand(n_examples) * np.pi * 2 / self.hz
        self.phase = phase if phase is not None else default_phases

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):

        # retrieve parameters for this example
        amp = self.amp if np.isscalar(self.amp) else self.amp[idx]
        hz = self.hz if np.isscalar(self.hz) else self.hz[idx]
        phase = self.phase if np.isscalar(self.phase) else self.phase[idx]

        # calculate signal
        x = torch.arange(0, self.n_seconds, 1 / self.cfg.sampling_rate)
        y = torch.sin((x - phase) * np.pi * 2 * hz) * amp
        y = y.unsqueeze(0)  # C, W

        # copy to n audio channels
        if self.cfg.n_audio_chans > 1:
            y = y.repeat(self.cfg.n_audio_chans, 1)

        # mu compress
        if self.cfg.compress:
            y = audio.mu_compress(y.numpy(), self.cfg)
            y = torch.from_numpy(y)

        return self.tf(audio.quantise(y, self.cfg))

    def __repr__(self):
        x = [("nseconds", self.n_seconds)]
        if np.isscalar(self.amp):
            x.append(("amp", self.amp))
        if np.isscalar(self.hz):
            x.append(("hz", self.hz))
        if np.isscalar(self.phase):
            x.append(("phase", self.phase))
        x = ", ".join([f"{k}: {v}" for k, v in x])
        return f"Sines({ x })"


class Tiny(Dataset):
    """A toy dataset of non-linear series, as described in
    https://fleuret.org/dlc/materials/dlc-slides-10-1-autoregression.pdf.
    This dataset is tiny, non-linear, and includes local effects (slopes), as
    well as global effects (a single discontinuity).
    """

    def __init__(self, n, m):
        """Produces a dataset of m timeseries, where each one is n long, and
        values are drawn from 0 to 2n.
        """

        # Create m timeseries of length n, with a random split in each one
        splits = torch.randint(0, n, (1, m))

        # Let's create 2m randomly tilted timeseries. First of, we want 2m
        # slopes between -1 and 1.  Then we'll broadcast along the x dimension
        # to obtain a n, 2m tensor, where each column is a tilted timeseries.
        x = torch.arange(n)
        slopes = torch.rand(2 * m) * 2 - 1
        series = slopes.view(1, -1) * x.view(-1, 1)  # (1,2m) * (n,1) => (n,2m)

        # Now we'll add another dimension and fill it with pairs of
        # complementary series, to (n, m, 2).
        series = series.reshape(n, -1, 2)

        # shift the right series such that it starts at zero.
        offsets = series[:, :, 1].gather(0, splits).squeeze()
        series[:, :, 1] -= offsets

        # Now, with the m split points between 0 and n, we create a mask where
        # values are below those cutoff points. We will apply the mask to half
        # the series, and the inverse of the mask to the complementary series.
        mask = torch.arange(n).unsqueeze(-1) > splits
        mask = torch.stack((mask, ~mask), dim=-1)
        series[mask] = 0

        # Now we sum together along the final dimension to obtain the result.
        series = torch.sum(series, dim=-1).long() + n

        # conventionally introduce a single channel dimension
        self.data = series.unsqueeze(0)

        # dataset stats
        mean = torch.mean(self.data.float())
        std = torch.std(self.data.float())
        self.tf = NormaliseTransforms(mean, std)

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, idx):
        return self.tf(self.data[:, :, idx])


class TinySines(Dataset):
    """Somewhat harder than either tiny or sines. Sweeping sines
    """

    def __init__(self, n, m):
        self.tiny = Tiny(n, m)
