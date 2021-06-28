import numpy as np  # type: ignore

from wavenet import audio, model


def test_load_raw_mono():
    y, sr = audio.load_raw("fixtures/short.wav", mono=True)
    assert sr == 44100
    assert y.shape == (1, 414627)


def test_load_raw_stereo():
    y, sr = audio.load_raw("fixtures/short.wav", mono=False)
    assert sr == 44100
    assert y.shape == (2, 414627)


def test_load_resampled_mono():
    p = model.HParams(squash_to_mono=True)
    y = audio.load_resampled("fixtures/short.wav", p)
    assert y.shape == (1, 150432)
    assert np.min(y) >= -1.0
    assert np.max(y) <= 1.0


def test_load_resampled_stereo():
    p = model.HParams(squash_to_mono=False)
    y = audio.load_resampled("fixtures/short.wav", p)
    assert y.shape == (2, 150432)
    assert np.min(y) >= -1.0
    assert np.max(y) <= 1.0


def test_mu_compress():
    p = model.HParams()
    t1 = np.random.rand(2, 4) * 2 - 1
    t1 = audio.mu_compress(t1, p)
    assert t1.dtype == np.float64
    assert t1.min() >= -1.0
    assert t1.max() <= 1.0


def test_mu_expand():
    p = model.HParams()
    t1 = np.random.rand(2, 4) * 2 - 1
    t1 = audio.mu_expand(t1, p)
    assert t1.dtype == np.float64
    assert t1.min() >= -1.0
    assert t1.max() <= 1.0


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


def test_quantise():
    p = model.HParams(n_classes=8)
    xs = np.arange(-1, 1, 1 / 16)
    got = audio.quantise(xs, p)
    valid_max = p.n_classes // 2 - 1
    valid_min = p.n_classes // -2
    assert got.max() == valid_max
    assert got.min() == valid_min
    assert set(np.unique(got)) == set(range(valid_min, valid_max + 1))
    for i in range(len(got) - 1):
        # monotonic
        assert got[i + 1] >= got[i]
