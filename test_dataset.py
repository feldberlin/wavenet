import torch
from torch.nn import functional as F

from wavenet import datasets, model


def test_stereo_impulse_dataset():
    d = datasets.StereoImpulse(10, 4,  model.HParams())
    x, y = d[0]
    assert y.shape == (2, 4)
    assert repr(d) == 'StereoImpulse()'
    assert len(d) == 10


def test_track_dataset():
    d = datasets.Track('fixtures/short.wav', model.HParams())
    x, y = d[0]
    assert y.shape == (2, 16000)
    assert len(d) == 16
    assert repr(d) == 'Track(fixtures/short.wav)'


def test_track_dataset_stacked():
    d = datasets.Track('fixtures/short.wav', model.HParams())
    x, y = datasets.to_tensor(d)
    assert x.shape == (16, 2, 16000)
    assert y.shape == (16, 2, 16000)
    assert torch.min(y) >= 0.
    assert torch.max(y) <= 256.


def test_track():
    p = model.HParams()
    ds, ds_test = datasets.tracks('fixtures/short.wav', 0.4, p)
    x, y = ds[0]
    x_test, y_test = ds_test[0]
    assert len(ds) == 9
    assert len(ds_test) == 5
    assert x.shape == (2, 16000)
    assert y.shape == (2, 16000)
    assert x_test.shape == (2, 16000)
    assert y_test.shape == (2, 16000)


def test_sines_dataset():
    d = datasets.Sines(4, 1, model.HParams())
    x, y = d[0]
    assert y.shape == (2, 16000)  # stereo
    assert y.shape == (2, 16000)  # stereo
    assert len(d) == 4
    assert repr(d) == 'Sines(nseconds: 1)'


def test_sines_fixed_amp_dataset():
    d = datasets.Sines(4, 1, model.HParams(), amp=0.5, hz=440)
    x, y = d[0]
    assert x.shape == (2, 16000)  # stereo
    assert y.shape == (2, 16000)  # stereo
    assert len(d) == 4
    assert repr(d) == 'Sines(nseconds: 1, amp: 0.5, hz: 440)'


def test_sines_dataset_stacked():
    d = datasets.Sines(4, 1, model.HParams())
    x, y = datasets.to_tensor(d)
    assert y.shape == (4, 2, 16000)
    assert torch.min(y) >= 0.
    assert torch.max(y) <= 256.


def test_sines_dataloader():
    d = datasets.Sines(10, 1, model.HParams())
    l = torch.utils.data.dataloader.DataLoader(d, batch_size=4)
    x, y = next(iter(l))
    assert x.shape == (4, 2, 16000)
    assert y.shape == (4, 2, 16000)


def test_tiny_dataloader():
    d = datasets.Tiny(30, 4)
    l = torch.utils.data.dataloader.DataLoader(d, batch_size=4)
    x, y = next(iter(l))
    assert len(d) == 4
    assert x.shape == (4, 1, 30)
    assert y.shape == (4, 1, 30)
