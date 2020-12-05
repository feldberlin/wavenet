import torch
from torch.nn import functional as F

from wavenet import datasets, model


def test_stereo_impulse_dataset():
    d = datasets.StereoImpulse(10, 4,  model.HParams())
    assert len(d) == 10
    assert d[0].shape == (2, 4)
    assert d[:].shape == (10, 2, 4)
    assert repr(d) == 'StereoImpulse()'


def test_track_dataset():
    d = datasets.Track('fixtures/short.wav', model.HParams())
    assert len(d) == 16
    assert d[:].shape == (16, 2, 16000)
    assert repr(d) == 'Track(fixtures/short.wav)'


def test_track_dataset_stacked():
    d = datasets.Track('fixtures/short.wav', model.HParams())
    stacked = datasets.to_tensor(d)
    assert stacked.shape == (16, 2, 16000)
    assert torch.min(stacked) >= -128.0
    assert torch.max(stacked) <= 127.0


def test_track():
    p = model.HParams()
    trainset, testset = datasets.tracks('fixtures/short.wav', 0.4, p)
    assert trainset[:].shape == (9, 2, 16000)
    assert testset[:].shape == (5, 2, 16000)


def test_sines_dataset():
    d = datasets.Sines(2, 1, model.HParams())
    assert len(d) == 2
    assert d[0].shape == (2, 16000)
    assert d[1].shape == (2, 16000)
    assert repr(d) == 'Sines(nseconds: 1)'


def test_sines_fixed_amp_dataset():
    d = datasets.Sines(2, 1, model.HParams(), amp=0.5, hz=440)
    assert len(d) == 2
    assert d[0].shape == (2, 16000)
    assert d[1].shape == (2, 16000)
    assert repr(d) == 'Sines(nseconds: 1, amp: 0.5, hz: 440)'


def test_sines_dataset_stacked():
    d = datasets.Sines(2, 1, model.HParams())
    stacked = datasets.to_tensor(d)
    assert stacked.shape == (2, 2, 16000)
    assert torch.min(stacked) >= -128.0
    assert torch.max(stacked) <= 127.0


def test_sines_dataloader():
    d = datasets.Sines(10, 1, model.HParams())
    l = torch.utils.data.dataloader.DataLoader(d, batch_size=4)
    assert next(iter(l)).shape == (4, 2, 16000)
