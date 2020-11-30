import numpy as np
from torch.utils.data import Dataset
import torch

from wavenet import utils, audio


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
        y = audio.load_resampled(filename, p)
        _, n_samples = y.shape
        y = y[:, int(n_samples * start):int(n_samples * end)]  # start to end
        y = audio.to_librosa(y)
        ys = audio.frame(y, p)
        ys = np.moveaxis(ys, -1, 0)
        ys = torch.tensor(ys, dtype=torch.float32)
        ys = ys[1:, :, :]  # trim hoplength leading silence
        self.X = audio.mu_compress_batch(ys, p)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


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


class Sines(Dataset):
    "Each sample is a simple sine wave with random amplitude and phase."

    def __init__(self, n_examples, n_seconds, cfg, minhz = 400, maxhz = 20000):
        self.hz = torch.rand(n_examples) * maxhz + minhz
        self.amp = torch.rand(n_examples)
        self.phase = torch.rand(n_examples) * np.pi * 2 / self.hz
        self.n_seconds = n_seconds
        self.cfg = cfg

    def __len__(self):
        return self.hz.shape[0]

    def __getitem__(self, idx):
        hz, amp, phase = self.hz[idx], self.amp[idx], self.phase[idx]
        x = torch.arange(0, self.n_seconds, 1 / self.cfg.sampling_rate)
        y = torch.sin((x - phase) * np.pi * 2 * hz) * amp
        y = y.unsqueeze(0)  # C, W
        if self.cfg.stereo:
            return y.repeat(2, 1)
        else:
            return y
