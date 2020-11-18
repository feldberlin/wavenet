import numpy as np
import torch

from wavenet import utils, audio


def stereo_impulse_at_t0(n, m, cfg, probs=None):
    "Left and right at t0 are both binomial, modes slightly apart."
    probs = probs if probs else (0.45, 0.55)
    X = np.random.binomial(
        (cfg.n_classes, cfg.n_classes),
        probs,
        (n, cfg.n_audio_chans))

    X = utils.quantized_audio_from_class_idxs(X, cfg)
    X_batched = np.reshape(X, (n, cfg.n_audio_chans, 1))
    X_batched = np.pad(X_batched, ((0, 0), (0, 0), (0, m - 1)))
    return torch.from_numpy(X_batched).float()


def preprocess(X, p, ratio: float = 0.8):
    "Return X, X_test, mean, variance with znormed features and mu law"

    # split the data
    split = int(X.shape[0] * ratio)
    X, X_test = X[:split], X[split:]

    # mu compress
    X = audio.mu_compress_batch(X, p)
    X_test = audio.mu_compress_batch(X_test, p)

    return X, X_test
