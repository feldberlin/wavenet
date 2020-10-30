import numpy as np
import torch
from torch.nn import functional as F


def logits_to_audio(logits, cfg):
    "Convert logits to audio"
    return from_class_idxs(logits_to_idxs(logits, cfg), cfg)


def logits_to_idxs(logits, cfg):
    "Convert logits to class indices"
    return torch.argmax(F.softmax(logits, dim=1), 1)


def to_class_idxs(audio, cfg):
    "Convert unnormalised floating point audio data to class indices"
    return audio.long() + (cfg.n_classes // 2) - 1


def from_class_idxs(idxs, cfg):
    "Convert indives back to audio"
    return (idxs - (cfg.n_classes // 2 + 1)).long()


def stereo_impulse_at_t0(n, m, cfg, probs=None):
    "Left and right at t0 are both binomial, modes slightly apart."
    probs = probs if probs else (0.45, 0.55)
    X = np.random.binomial(
        (cfg.n_classes, cfg.n_classes),
        probs,
        (n, cfg.n_audio_chans)) - (cfg.n_classes / 2)

    X_batched = np.reshape(X, (n, cfg.n_audio_chans, 1))
    X_batched = np.pad(X_batched, ((0, 0), (0, 0), (0, m - 1)))
    return torch.from_numpy(X_batched).float()