import numpy as np

from wavenet import audio


def test_load():
    y, sr = audio.load_raw('data/steinway.wav')
    assert sr == 22050
    assert len(y) == 302400


def test_load_normalised():
    p = audio.HParams()
    y = audio.load_normalised('data/steinway.wav', p)
    assert np.min(y) > -127
    assert np.max(y) < 128
