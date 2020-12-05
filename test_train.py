import pytest

from wavenet import train, datasets, model, audio


def test_trainer_params():
    p = train.HParams()
    assert p.batch_size == 64


def test_trainer_params_override():
    p = train.HParams(batch_size=1)
    assert p.batch_size == 1


@pytest.mark.integration
def test_learn_bimodally_distributed_stereo_at_t0():
    p = model.HParams(n_chans=2)
    X = datasets.StereoImpulse(2**13, 1,  p)
    m = model.Wavenet(p)
    t = train.Trainer(m, X, None, train.HParams(max_epochs=1), None)
    t.train()


@pytest.mark.integration
def test_lr_scheduler_with_less_than_one_full_step():
    p = model.HParams(n_audio_chans=2, n_chans=2, n_layers=8)
    tp = train.HParams(max_epochs=1, batch_size=8)
    X, X_test = datasets.tracks('fixtures/short.wav', 0.2, p)
    m = model.Wavenet(p)
    t = train.Trainer(m, X, X_test, tp, None)
    t.train()
