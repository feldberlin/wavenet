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


def test_state():
    m = model.Wavenet(model.HParams())
    t = train.Trainer(m, [1, 2, 3], [1, 2], train.HParams())
    state = t.state()
    assert "model" in state
    assert "optimizer" in state
    assert "scaler" in state
    assert "schedule" in state
    assert "epoch" in state
    assert "best" in state


def test_load_state():
    m = model.Wavenet(model.HParams())
    t = train.Trainer(m, [1, 2, 3], [1, 2], train.HParams())
    state = t.state()
    t.load_state(state)


def test_shard_cfg():
    tp = train.HParams(batch_size=10, num_workers=8)
    tp.shard(2)
    assert tp.batch_size == 5
    assert tp.num_workers == 4
    assert tp.num_shards == 2
    assert tp.total_batch_size() == 10


def test_sharded_learning_rate_schedule():
    ds, ds_test = range(100), None
    m = model.Wavenet(model.HParams().with_all_chans(2))
    tp = train.HParams(batch_size=10, num_workers=8, max_epochs=1)
    n_steps = int(len(ds) / tp.batch_size)

    # unsharded schedule
    t = train.Trainer(m, ds, ds_test, tp, log=False)
    schedule = t.schedule

    # sharded schedule
    tp.shard(2)
    t = train.Trainer(m, ds, ds_test, tp, log=False)
    sharded_schedule = t.schedule

    for i in range(n_steps):
        schedule.step()
        sharded_schedule.step()
        assert schedule.get_last_lr() == sharded_schedule.get_last_lr()


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
    ds, ds_test = datasets.tracks("fixtures/goldberg/short.wav", 0.2, p)
    m = model.Wavenet(p)
    t = train.Trainer(m, ds, ds_test, tp, None)
    t.train()
