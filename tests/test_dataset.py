import math
from pathlib import Path

import test_helpers as helpers
import torch
from torch.nn import functional as F

from wavenet import audio, datasets, model

# tracks


def test_tracks():

    with helpers.tempdir() as cache:
        p = model.HParams(sample_length=16000, sample_overlap_length=0)
        root = Path("fixtures/goldberg")
        ds = datasets.Tracks.from_dir(p, root, cache_dir=cache)
        cache = cache / p.audio_cache_key()
        assert set(ds.tracks) == set(
            [
                datasets.TrackMeta(root, cache, Path("goldberg.wav"), 1507200),
                datasets.TrackMeta(root, cache, Path("short.wav"), 150431),
                datasets.TrackMeta(root, cache, Path("aria.wav"), 4792320),
            ]
        )

        # spot check that short is actually as short as claimed
        path, duration = root / "short.wav", 150432  # slight error.
        y = audio.load_resampled(path, p)
        _, n_samples = y.shape
        assert duration == n_samples

        # check that the ds length is consistent with meta durations
        got_duration = len(ds) * p.sample_length
        want_duration = sum(
            [audio.prune_duration(t.duration, p) for t in ds.tracks]
        )
        assert got_duration == want_duration

        # check that we can retrieve first and last examples.
        # expensive due to the resampling step.
        for i in [0, len(ds) - 1]:
            x, y, meta = ds[i]
            assert x is not None
            assert x.shape == (2, p.sample_length)
            assert y is not None
            assert y.shape == (2, p.sample_length)
            assert meta is not None


def test_tracks_overlapped_receptive_fields():

    with helpers.tempdir() as cache:
        p = model.HParams(sample_length=16000, sample_overlap_length=8000)
        root = Path("fixtures/goldberg")
        ds = datasets.Tracks.from_dir(p, root, cache_dir=cache)

        # expected audio samples duration
        def expected_duration(ds):
            duration = 0
            for t in ds.tracks:
                # prune so that examples will fit in exactly
                l = audio.prune_duration(t.duration, p)

                # length if you would pad all trailing examples that are
                # actually too long to fit into the track
                l = math.floor(l / p.sample_hop_length()) * p.sample_length

                # length of the overcounted trailing examples. these are too
                # long to be contained in the track
                l -= (
                    math.floor(p.sample_length / p.sample_hop_length()) - 1
                ) * p.sample_length

                # accumulate across all tracks
                duration += l
            return duration

        # number of audio samples across all examples in the dataset
        dataset_duration = 0
        for i in range(len(ds)):
            x, y, meta = ds[i]
            n_samples = x.shape[-1]
            dataset_duration += n_samples
            assert n_samples == p.sample_length, i

        assert expected_duration(ds) == dataset_duration


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
    sr = 16000
    p = model.HParams(compress=True, sample_overlap_length=sr - 2 ** 13)
    d = datasets.Track("fixtures/goldberg/short.wav", p)
    x, y = d[0]
    assert y.shape == (2, sr)
    assert len(d) == 17
    assert repr(d) == "Track(fixtures/goldberg/short.wav)"


def test_track_uncompressed():
    sr = 16000
    p = model.HParams(compress=False, sample_overlap_length=sr - 2 ** 13)
    d = datasets.Track("fixtures/goldberg/short.wav", p)
    x, y = d[0]
    assert y.shape == (2, sr)
    assert len(d) == 17
    assert repr(d) == "Track(fixtures/goldberg/short.wav)"


def test_track_dataset_stacked():
    sr = 16000
    p = model.HParams(sample_overlap_length=sr - 2 ** 13)
    d = datasets.Track("fixtures/goldberg/short.wav", p)
    x, y = datasets.to_tensor(d)
    assert x.shape == (17, 2, sr)
    assert y.shape == (17, 2, sr)
    assert torch.min(y) >= 0.0
    assert torch.max(y) <= 256.0


def test_track():
    sr = 16000
    p = model.HParams(sample_overlap_length=sr - 2 ** 13)
    ds, ds_test = datasets.tracks("fixtures/goldberg/short.wav", 0.4, p)
    x, y = ds[0]
    x_test, y_test = ds_test[0]
    assert len(ds) == 10
    assert len(ds_test) == 6
    assert x.shape == (2, sr)
    assert y.shape == (2, sr)
    assert x_test.shape == (2, sr)
    assert y_test.shape == (2, sr)
