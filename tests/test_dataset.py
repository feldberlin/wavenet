from pathlib import Path

import test_helpers as helpers
import torch
from torch.nn import functional as F

from wavenet import audio, datasets, model

# tracks


def test_tracks():
    with helpers.tempdir() as cache:
        p = model.HParams()
        root = Path("fixtures/goldberg")
        ds = datasets.Tracks.from_dir(p, root, cache_dir=cache)
        assert set(ds.tracks) == set(
            [
                datasets.TrackMeta(root, cache, Path("goldberg.wav"), 1504000),
                datasets.TrackMeta(root, cache, Path("short.wav"), 144000),
                datasets.TrackMeta(root, cache, Path("aria.wav"), 4784000),
            ]
        )

        # spot check that short is actually as long as claimed
        path, duration = root / "short.wav", 144000
        y = audio.load_resampled(path, p)
        _, n_samples = y.shape
        truncated = (n_samples // p.sample_length) * p.sample_length
        assert duration == truncated

        # check that the ds length is consistent with meta durations
        assert len(ds) * p.sample_length == sum([t.duration for t in ds.tracks])

        # check that we can retrieve first and last examples.
        # expensive due to the resampling step.
        for i in [0, len(ds) - 1]:
            x, y, meta = ds[i]
            assert x is not None
            assert x.shape == (2, p.sample_length)
            assert y is not None
            assert y.shape == (2, p.sample_length)
            assert meta is not None


def test_maestro():
    root_dir = Path("fixtures/maestro")
    train, test = datasets.maestro(root_dir, 2018, model.HParams())
    assert len(train) == 1
    assert len(test) == 1


# sines


def test_sines_dataset():
    d = datasets.Sines(4, model.HParams())
    x, y = d[0]
    assert y.shape == (2, 16000)  # stereo
    assert y.shape == (2, 16000)  # stereo
    assert len(d) == 4
    assert repr(d) == "Sines(nseconds: 1.0)"


def test_sines_fixed_amp_dataset():
    d = datasets.Sines(4, model.HParams(), amp=0.5)
    x, y = d[0]
    assert x.shape == (2, 16000)  # stereo
    assert y.shape == (2, 16000)  # stereo
    assert isinstance(d.amp, float)  # one amp for all examples
    assert d.hz.shape == (4,)  # one hz per example
    assert d.phase.shape == (4,)  # one phase per example
    assert len(d) == 4
    assert repr(d) == "Sines(nseconds: 1.0, amp: 0.5)"


def test_sines_fixed_hz_dataset():
    d = datasets.Sines(4, model.HParams(), hz=200.0)
    x, y = d[0]
    assert x.shape == (2, 16000)  # stereo
    assert y.shape == (2, 16000)  # stereo
    assert d.amp.shape == (4,)  # one amp per example
    assert isinstance(d.hz, float)  # one hz for all examples
    assert d.phase.shape == (4,)  # one phase per example
    assert len(d) == 4
    assert repr(d) == "Sines(nseconds: 1.0, hz: 200.0)"


def test_sines_fixed_phase_dataset():
    d = datasets.Sines(4, model.HParams(), phase=0.0)
    x, y = d[0]
    assert x.shape == (2, 16000)  # stereo
    assert y.shape == (2, 16000)  # stereo
    assert d.amp.shape == (4,)  # one amp per example
    assert d.hz.shape == (4,)  # one hz per example
    assert isinstance(d.phase, float)  # one phase for all examples
    assert len(d) == 4
    assert repr(d) == "Sines(nseconds: 1.0, phase: 0.0)"


def test_sines_dataset_stacked():
    d = datasets.Sines(4, model.HParams())
    x, y = datasets.to_tensor(d)
    assert y.shape == (4, 2, 16000)
    assert torch.min(y) >= 0.0
    assert torch.max(y) <= 256.0


def test_sines_dataloader():
    d = datasets.Sines(10, model.HParams())
    l = torch.utils.data.dataloader.DataLoader(d, batch_size=4)
    x, y = next(iter(l))
    assert x.shape == (4, 2, 16000)
    assert y.shape == (4, 2, 16000)


# tiny


def test_tiny_dataloader():
    d = datasets.Tiny(30, 4)
    l = torch.utils.data.dataloader.DataLoader(d, batch_size=4)
    x, y = next(iter(l))
    assert len(d) == 4
    assert x.shape == (4, 1, 30)
    assert y.shape == (4, 1, 30)


# impulse


def test_stereo_impulse_dataset():
    d = datasets.StereoImpulse(10, 4, model.HParams())
    x, y = d[0]
    assert y.shape == (2, 4)
    assert repr(d) == "StereoImpulse()"
    assert len(d) == 10


# track


def test_track_dataset():
    d = datasets.Track(
        "fixtures/goldberg/short.wav", model.HParams(compress=True)
    )
    x, y = d[0]
    assert y.shape == (2, 16000)
    assert len(d) == 16
    assert repr(d) == "Track(fixtures/goldberg/short.wav)"


def test_track_uncompressed():
    d = datasets.Track(
        "fixtures/goldberg/short.wav", model.HParams(compress=False)
    )
    x, y = d[0]
    assert y.shape == (2, 16000)
    assert len(d) == 16
    assert repr(d) == "Track(fixtures/goldberg/short.wav)"


def test_track_dataset_stacked():
    d = datasets.Track("fixtures/goldberg/short.wav", model.HParams())
    x, y = datasets.to_tensor(d)
    assert x.shape == (16, 2, 16000)
    assert y.shape == (16, 2, 16000)
    assert torch.min(y) >= 0.0
    assert torch.max(y) <= 256.0


def test_track():
    p = model.HParams()
    ds, ds_test = datasets.tracks("fixtures/goldberg/short.wav", 0.4, p)
    x, y = ds[0]
    x_test, y_test = ds_test[0]
    assert len(ds) == 9
    assert len(ds_test) == 5
    assert x.shape == (2, 16000)
    assert y.shape == (2, 16000)
    assert x_test.shape == (2, 16000)
    assert y_test.shape == (2, 16000)
