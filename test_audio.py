import numpy as np
import torch

from wavenet import model, audio


def test_load():
    y, sr = audio.load_raw('data/steinway.wav')
    assert sr == 22050
    assert len(y) == 302400


def test_load_normalised():
    p = model.HParams()
    y = audio.load_resampled_quantised('data/steinway.wav', p)
    assert np.min(y) >= -128
    assert np.max(y) <= 127


def test_normalise_unnormalise_identity():
    p = model.HParams(resample=False)

    # include one lossy pass
    t1 = np.random.rand(2, 4) * 2 - 1
    t1 = audio.resample_quantise(t1, None, p)
    t1 = audio.mu_expand(t1, p)

    # non lossy second pass
    t2 = audio.resample_quantise(t1, None, p)
    t2 = audio.mu_expand(t2, p)

    assert np.all(t1 == t2)
