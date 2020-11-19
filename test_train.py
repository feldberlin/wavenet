import pytest

from wavenet import train, datasets, model


def test_trainer_params():
    p = train.HParams()
    assert p.batch_size == 64


def test_trainer_params_override():
    p = train.HParams(batch_size=1)
    assert p.batch_size == 1


@pytest.mark.integration
def test_learn_bimodally_distributed_stereo_at_t0():
    p = model.HParams(n_chans=2)
    X = datasets.stereo_impulse_at_t0(2**13, 1,  p)
    m = model.Wavenet(p)
    t = train.Trainer(m, X, None, train.HParams(max_epochs=1), None)
    t.train()
