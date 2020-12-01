import inspect

import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
import wandb


# logits and normalisation

def logits_to_class_idxs(logits, cfg):
    "Convert logits to class indices via softmax argmax"
    return torch.argmax(F.softmax(logits, dim=1), 1)


def logits_to_audio(logits, cfg):
    "Convert logits to audio"
    idxs = logits_to_class_idxs(logits, cfg)
    return quantized_audio_from_class_idxs(idxs, cfg)


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


# generator decoders

def decode_random(logits):
    "Convert N, K, C, 1 logits into N, C, 1 samples by random sampling"
    N, K, C, W = logits.shape
    assert W == 1
    posterior = F.softmax(logits, dim=1)
    posterior = posterior.squeeze(-1).permute(0, 2, 1)
    d = torch.distributions.Categorical(posterior)
    return d.sample().unsqueeze(-1)


def decode_argmax(logits):
    "Convert N, K, C, 1 logits into N, C, 1 samples by argmax"
    N, K, C, W = logits.shape
    assert W == 1
    return torch.argmax(F.softmax(logits, dim=1), dim=1)


def decode_nucleus(core_mass: float = 0.95):
    """Convert N, K, C, 1 logits into N, C, 1 samples by nucleus sampling"
    as proposed in https://arxiv.org/pdf/1904.09751.pdf
    """
    def fn(logits):
        N, K, C, W = logits.shape
        assert W == 1
        sorted, idxs = torch.sort(logits, dim=1)
        csum = torch.cumsum(F.softmax(sorted, dim=1), dim=1)
        logits[:, idxs[csum > core_mass]] = -float('Inf')
        return decode_random(logits)
    return fn


# schedules

def lrfinder(optimizer, n_examples, cfg):
    start_lr, final_lr = 1e-8, 10.
    n_steps = cfg.n_steps(n_examples)
    gamma = (final_lr / start_lr) ** (1/n_steps)
    return lr_scheduler.ExponentialLR(optimizer, gamma)


def onecycle(optimizer, n_examples, cfg):
    lr = cfg.learning_rate
    n_steps = cfg.n_steps(n_examples)
    return lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_steps)


# config

def cfgdict(model_cfg, train_cfg):
    return {**dict(model_cfg), 'train': dict(train_cfg)}


class HParams():
    "Make HParams iterable so we can call dict on it"
    def __iter__(self):
        def f(obj):
            return {k: v for k, v
                    in vars(obj).items()
                    if not k.startswith('__')
                    and not inspect.isfunction(v)}

        return iter({**f(self.__class__), **f(self)}.items())


# lifecycle

def load_chkpt(m, run_path):
    chkpt = wandb.restore('checkpoint.best.test', run_path=run_path)
    m.load_state_dict(torch.load(chkpt.name))
    return m
