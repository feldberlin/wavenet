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
    m = model.Wavenet(model.HParams())
    g = sample.Generator(m)
    x, loss = g.forward(torch.randint(5, (2, 2, 1)).float())
    assert x.shape == (2, m.cfg.n_classes, m.cfg.n_audio_chans, 1)


def test_input_weights_generator_vs_wavenet():
    m = model.Wavenet(model.HParams())
    g = sample.Generator(m)
    assert torch.equal(m.input.weight, g.input.c.weight)


def test_input_units_generator_vs_wavenet():
    m = model.Wavenet(model.HParams())
    g = sample.Generator(m)
    x = torch.zeros((1, m.cfg.n_audio_chans, 1))
    ym, yg = m.input.forward(x).squeeze(), g.input.forward(x).squeeze()
    assert torch.equal(ym, yg)


def test_one_logit_generator_vs_wavenet():
    m = model.Wavenet(model.HParams())
    g = sample.Generator(m)
    x = torch.zeros((1, m.cfg.n_audio_chans, 1))

    # a single forward pass through both networks
    ym, _ = m.forward(x)
    yg, _ = g.forward(x)
    ym = F.softmax(ym.squeeze())
    yg = F.softmax(yg.squeeze())

    assert torch.allclose(ym, yg)


def test_one_sample():
    m = model.Wavenet(model.HParams())
    track = sample.sample(m, 1)
    assert track.shape == (1, m.cfg.n_audio_chans, 1)


def test_two_samples():
    m = model.Wavenet(model.HParams())
    track = sample.sample(m, 2)
    assert track.shape == (1, m.cfg.n_audio_chans, 2)


def test_many_samples():
    m = model.Wavenet(model.HParams())
    track = sample.sample(m, 50)
    assert track.shape == (1, m.cfg.n_audio_chans, 50)
