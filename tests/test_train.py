import pytest

from wavenet import audio, datasets, model, train


def test_trainer_params():
    p = train.HParams()
    assert p.batch_size == 64


def test_trainer_params_override():
    p = train.HParams(batch_size=1)
    assert p.batch_size == 1


def test_hparams_nsteps():
    trainset_size = 80
    tp = train.HParams(batch_size=2, max_epochs=10)
    assert tp.n_steps(trainset_size) == (80 / 2) * 10


def test_hparams_nsteps_batch_too_large():
    trainset_size = 80
    tp = train.HParams(batch_size=80, max_epochs=10)
    assert tp.n_steps(trainset_size) == (80 / 80) * 10


def test_hparams_nsteps_last_batch_small():
    trainset_size = 48
    tp = train.HParams(batch_size=40, max_epochs=4)
    assert tp.n_steps(trainset_size) == 8


@pytest.mark.integration
def test_learn_bimodally_distributed_stereo_at_t0():
    p = model.HParams().with_all_chans(2)
    ds = datasets.StereoImpulse(2 ** 13, 1, p)
    m = model.Wavenet(p)
    t = train.Trainer(m, ds, None, train.HParams(max_epochs=1), None)
    t.train()


@pytest.mark.integration
def test_lr_scheduler_with_less_than_one_full_step():
    p = model.HParams(n_audio_chans=2, n_layers=8).with_all_chans(2)
    tp = train.HParams(max_epochs=1, batch_size=8)
    X, X_test = datasets.tracks("fixtures/goldberg/short.wav", 0.2, p)
    m = model.Wavenet(p)
    t = train.Trainer(m, X, X_test, tp, None)
    t.train()
