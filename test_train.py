from wavenet import train


def test_trainer_params():
    p = train.HParams()
    assert p.batch_size == 64


def test_trainer_params_override():
    p = train.HParams(batch_size=1)
    assert p.batch_size == 1
