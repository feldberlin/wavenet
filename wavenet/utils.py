import inspect

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


def load(model, name, p):
    return model.load_state_dict(torch.load(path))


class HParams():
    "Make HParams iterable so we can call dict on it"
    def __iter__(self):
        def f(obj):
            return { k:v for k, v
                     in vars(obj).items()
                     if not k.startswith('__')
                     and not inspect.isfunction(v) }

        return iter({ **f(self), **f(self.__class__) }.items())
