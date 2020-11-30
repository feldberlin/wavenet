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
    """Dataset constructed from a single track
    Loads μ compressed  slices from a single track into N, C, W in [-1., 1.].
    """

    def __init__(self, filename: str, p, start: float = 0.0, end: float = 1.0):
        y = audio.load_resampled(filename, p)
        _, nsamples = y.shape
        y = y[:, int(nsamples * start):int(nsamples * end)]  # start to end
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
    Batch of m impulses at t0, followed by n zero samples.
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
