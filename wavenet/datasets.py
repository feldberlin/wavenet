import numpy as np
from torch.utils.data import Dataset
import torch

from wavenet import utils, audio


def to_tensor(d: Dataset, n_items = None):
    "Materialize the whole dataset"
    n_items = n_items if n_items else len(d)
    return torch.stack([d[i] for i in range(n_items)])


def tracks(filename: str, validation_pct: float, p):
    "Train - validation split on a single track"
    return (
        Track(filename, p, 0., 1-validation_pct),
        Track(filename, p, 1-validation_pct, 1.)
    )


class Track(Dataset):
    """Dataset constructed from a single track, held in memory.
    Loads μ compressed  slices from a single track into N, C, W in [-1., 1.].
    """

    def __init__(self, filename: str, p, start: float = 0.0, end: float = 1.0):
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
        return self.X[idx]

    def __repr__(self):
        return f'Track({self.filename})'


class StereoImpulse(Dataset):
    """Left and right at t0 are both binomial, modes slightly apart.
    Batch of m impulses at t0, followed by n zero samples, held in memory.
    """

    def __init__(self, n, m, cfg, probs=None):
        probs = probs if probs else (0.45, 0.55)
        X = np.random.binomial(
            (cfg.n_classes, cfg.n_classes),
            probs,
            (n, cfg.n_audio_chans))

        X = utils.quantized_audio_from_class_idxs(X, cfg)
        X_batched = np.reshape(X, (n, cfg.n_audio_chans, 1))
        X_batched = np.pad(X_batched, ((0, 0), (0, 0), (0, m - 1)))
        self.X = torch.from_numpy(X_batched).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

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
        return torch.from_numpy(y).float()

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
