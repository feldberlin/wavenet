import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd.functional import jacobian

import pytest

from wavenet import model, sample



def test_generator_init():
    m = model.Wavenet(model.HParams())
    assert sample.Generator(m)


def test_generator_forward():
    p = model.HParams()
    m = model.Wavenet(p)
    g = sample.Generator(m)
    x, loss = g.forward(torch.randint(5, (2, 2, 1)).float())
    assert x.shape == (2, p.n_classes, p.n_audio_chans, 1)


def test_sample():
    n_samples = 10
    p = model.HParams()
    m = model.Wavenet(p)
    track = sample.sample(m, p, n_samples)
    assert track.shape == (1, p.n_audio_chans, n_samples)
