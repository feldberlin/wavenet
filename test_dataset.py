import torch
from torch.nn import functional as F

from wavenet import datasets, model


def test_stereo_impulse_dataset():
    d = datasets.StereoImpulse(10, 4,  model.HParams())
    assert d[:].shape == (10, 2, 4)


def test_track_dataset():
    d = datasets.Track('data/short.wav', model.HParams())
    assert d[:].shape == (16, 2, 16000)


def test_track():
    p = model.HParams()
    trainset, testset = datasets.tracks('data/short.wav', 0.4, p)
    assert trainset[:].shape == (9, 2, 16000)
    assert testset[:].shape == (5, 2, 16000)
