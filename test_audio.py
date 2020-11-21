import numpy as np

from wavenet import model, audio


def test_load_raw_mono():
    y, sr = audio.load_raw('data/steinway.wav', mono=True)
    assert sr == 44100
    assert y.shape == (1, 604800)


def test_load_raw_stereo():
    y, sr = audio.load_raw('data/steinway.wav', mono=False)
    assert sr == 44100
    assert y.shape == (2, 604800)


def test_load_resampled_mono():
    p = model.HParams(stereo=False)
    y = audio.load_resampled('data/steinway.wav', p)
    assert y.shape == (1, 219429)
    assert np.min(y) >= -1.0
    assert np.max(y) <= 1.0


def test_load_resampled_stereo():
    p = model.HParams(stereo=True)
    y = audio.load_resampled('data/steinway.wav', p)
    assert y.shape == (2, 219429)
    assert np.min(y) >= -1.0
    assert np.max(y) <= 1.004  # wtf?


def test_compansion_round_trip():
    p = model.HParams()

    # include one lossy pass
    t1 = np.random.rand(2, 4) * 2 - 1
    t1 = audio.mu_compress(t1, p)
    t1 = audio.mu_expand(t1, p)

    # non lossy second pass
    t2 = audio.mu_compress(t1, p)
    t2 = audio.mu_expand(t2, p)

    assert np.allclose(t1, t2)


def test_load_dataset_from_track():
    p = model.HParams()
    x = audio.load_dataset_from_track('data/steinway.wav', p)
    assert x.shape == (24, 2, 16000)  # N, C, W
