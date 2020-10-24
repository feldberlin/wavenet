from wavenet import train


def test_trainer_params():
    p = train.TrainerParams()
    assert p.batch_size == 64

