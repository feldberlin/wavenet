import numpy as np
import torch

from wavenet import model, audio


def test_load():
    y, sr = audio.load_raw('data/steinway.wav')
    assert sr == 44100
    assert len(y) == 604800


def test_load_normalised():
    p = model.HParams()
    y = audio.load_resampled('data/steinway.wav', p)
    assert np.min(y) >= -1.0
    assert np.max(y) <= 1.0


def test_compansion_round_trip():
    p = model.HParams(resample=False)

    # include one lossy pass
    t1 = np.random.rand(2, 4) * 2 - 1
    t1 = audio.mu_compress(t1, p)
    t1 = audio.mu_expand(t1, p)

    # non lossy second pass
    t2 = audio.mu_compress(t1, p)
    t2 = audio.mu_expand(t2, p)

    assert np.allclose(t1, t2)


def test_znorm():
    X = np.random.rand(10, 2, 4) * 2 - 1
    X, _, _  = audio.znorm(X)
    assert np.allclose(X.mean(0), 0.)
    assert np.allclose(X.var(0), 1.)
