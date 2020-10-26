from wavenet import train


def test_trainer_params():
    p = train.HParams()
    assert p.batch_size == 64
