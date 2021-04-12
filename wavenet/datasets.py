# Datasets and transforms. Each enumerated element is (x, y) with normalised
# x.

import numpy as np
from torch.utils.data import Dataset
import torch

from wavenet import utils, audio


def to_tensor(d: Dataset, n_items = None):
    "Materialize the whole dataset"
    n_items = n_items if n_items else len(d)
    return (
        torch.stack([d[i][0] for i in range(n_items)]),
        torch.stack([d[i][1] for i in range(n_items)])
    )


def tracks(filename: str, validation_pct: float, p):
    "Train - validation split on a single track"
    return (
        Track(filename, p, 0., 1-validation_pct),
        Track(filename, p, 1-validation_pct, 1.)
    )


class Transforms:
    "Normalise x, convert y to class indices"

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, x, y):
        x = utils.audio_to_unit_loudness(x, self.cfg.n_classes)
        y = utils.audio_to_class_idxs(y, self.cfg.n_classes)
        return x, y


class Track(Dataset):
    """Dataset constructed from a single track, held in memory.
    Loads μ compressed slices from a single track into N, C, W in [-1., 1.].
    """

    def __init__(self, filename: str, p, start: float = 0.0, end: float = 1.0):
        self.transforms = Transforms(p)
        self.filename = filename
        y = audio.load_resampled(filename, p)
        _, n_samples = y.shape
        y = y[:, int(n_samples * start):int(n_samples * end)]  # start to end
        y = audio.to_librosa(y)
        ys = audio.frame(y, p)
        ys = np.moveaxis(ys, -1, 0)
        ys = torch.tensor(ys, dtype=torch.float32)
        ys = ys[1:, :, :]  # trim hoplength leading silence
        ys = audio.mu_compress_batch(ys, p)
        self.X = torch.from_numpy(ys).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.transforms(self.X[idx], self.X[idx])

    def __repr__(self):
        return f'Track({self.filename})'


class StereoImpulse(Dataset):
    """Left and right at t0 are both binomial, modes slightly apart.
    Batch of m impulses at t0, followed by n zero samples, held in memory.
    """

    def __init__(self, n, m, cfg, probs=None):
        self.transforms = Transforms(cfg)
        probs = probs if probs else (0.45, 0.55)
        X = np.random.binomial(
            (cfg.n_classes, cfg.n_classes),
            probs,
            (n, cfg.n_audio_chans))

        X = utils.audio_from_class_idxs(X, cfg.n_classes)
        X_batched = np.reshape(X, (n, cfg.n_audio_chans, 1))
        X_batched = np.pad(X_batched, ((0, 0), (0, 0), (0, m - 1)))
        self.X = torch.from_numpy(X_batched).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.transforms(self.X[idx], self.X[idx])

    def __repr__(self):
        return f'StereoImpulse()'


class Sines(Dataset):
    """Each sample is a μ compressed sine wave with random amp and phase.
    Random amplitude and hz, unless given.
    """

    def __init__(self, n_examples, n_seconds, cfg,
                 amp: float = None, hz: float = None, phase: float = None,
                 minhz = 400, maxhz = 20000):

        # config
        self.n_seconds = n_seconds
        self.n_examples = n_examples
        self.cfg = cfg

        # normalisations
        self.transforms = Transforms(cfg)

        # draw random parameters at init
        self.amp = amp if amp else torch.rand(n_examples)
        self.hz = hz if hz else torch.rand(n_examples) * maxhz + minhz
        self.phase = phase if phase else torch.rand(n_examples) * np.pi * 2 / self.hz

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

        if self.cfg.stereo: y = y.repeat(2, 1)
        y = audio.mu_compress(y.numpy(), self.cfg)
        y = torch.from_numpy(y).float()

        # normalise
        return self.transforms(y, y)

    def __repr__(self):
        x = [('nseconds', self.n_seconds)]
        if np.isscalar(self.amp):
            x.append(('amp', self.amp))
        if np.isscalar(self.hz):
            x.append(('hz', self.hz))
        if np.isscalar(self.phase):
            x.append(('phase', self.phase))
        x = ', '.join([f'{k}: {v}' for k, v in x])
        return f'Sines({ x })'


class Tiny(Dataset):
    """A toy dataset of non-linear series, as described in
    https://fleuret.org/dlc/materials/dlc-slides-10-1-autoregression.pdf.
    This dataset is tiny, non-linear, and includes local effects (slopes), as
    well as global effects (a single discontinuity).
    """

    def __init__(self, n, m):

        # Create m timeseries of length n, with a random split in each one
        splits = torch.randint(0, n, (1, m))

        # Let's create 2m randomly tilted timeseries. First of, we want 2m
        # slopes between -1 and 1.  Then we'll broadcast along the x dimension
        # to obtain a n, 2m tensor, where each column is a tilted timeseries.
        x = torch.arange(n)
        slopes = torch.rand(2*m) * 2 - 1
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
        mask =  torch.arange(n).unsqueeze(-1) > splits
        mask = torch.stack((mask, ~mask), dim=-1)
        series[mask] = 0

        # Now we sum together along the final dimension to obtain the result.
        series = torch.sum(series, dim=-1).long() + n

        # conventionally introduce a single channel dimension
        self.X = series.unsqueeze(0)

        # dataset stats
        self.mean = torch.mean(self.X.float())
        self.std = torch.std(self.X.float())

    def __len__(self):
        return self.X.shape[2]

    def __getitem__(self, idx):
        y = self.X[:, :, idx]
        x = (y - self.mean) / self.std
        return x, y
