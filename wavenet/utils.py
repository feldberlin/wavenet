import numpy as np
import torch
from torch.nn import functional as F

from wavenet import model, audio as waudio


def logits_to_class_idxs(logits, cfg):
    "Convert logits to class indices via softmax argmax"
    return torch.argmax(F.softmax(logits, dim=1), 1)


def logits_to_audio(logits, cfg):
    "Convert logits to audio"
    return quantized_audio_from_class_idxs(logits_to_class_idxs(logits, cfg), cfg)


def quantized_audio_to_class_idxs(audio, cfg):
    "Convert audio [-128, 127] to class indices [0, 255]."
    assert audio.min() >= -cfg.n_classes // 2, audio.min()
    assert audio.max() <= cfg.n_classes // 2 - 1, audio.max()
    return (audio + cfg.n_classes // 2).long()


def quantized_audio_from_class_idxs(idxs, cfg):
    "Convert class indices [0, 255] to audio [-128, 127]."
    assert idxs.min() >= 0, idxs.min()
    assert idxs.max() <= cfg.n_classes - 1, idxs.max()
    return idxs - cfg.n_classes // 2


def quantized_audio_to_unit_loudness(audio, cfg):
    "Convert audio in [-128, 127] to [-1., 1.]."
    assert audio.min() >= -cfg.n_classes // 2, audio.min()
    assert audio.max() <= cfg.n_classes // 2 - 1, audio.max()
    return (audio / (cfg.n_classes / 2.0))


def sample_from_logits(logits):
    "Convert N, K, C, 1 logits into N, C, 1 samples"
    N, K, C, W = logits.shape
    assert W == 1
    posterior = F.softmax(logits, dim=1)
    posterior = posterior.squeeze(-1).permute(0, 2, 1)
    d = torch.distributions.Categorical(posterior)
    return d.sample().unsqueeze(-1)


def stereo_impulse_at_t0(n, m, cfg, probs=None):
    "Left and right at t0 are both binomial, modes slightly apart."
    probs = probs if probs else (0.45, 0.55)
    X = np.random.binomial(
        (cfg.n_classes, cfg.n_classes),
        probs,
        (n, cfg.n_audio_chans))

    X = quantized_audio_from_class_idxs(X, cfg)
    X_batched = np.reshape(X, (n, cfg.n_audio_chans, 1))
    X_batched = np.pad(X_batched, ((0, 0), (0, 0), (0, m - 1)))
    return torch.from_numpy(X_batched).float()


def preprocess(X, p, ratio: float = 0.8):
    "Return X, X_test, mean, variance with znormed features and mu law"

    # split the data
    split = int(X.shape[0] * ratio)
    X, X_test = X[:split], X[split:]

    # mu compress
    X = waudio.mu_compress_batch(X, p)
    X_test = waudio.mu_compress_batch(X_test, p)

    return X, X_test


def load_checkpoint(model, name, p):
    return model.load_state_dict(torch.load(path))
