import torch
import numpy as np

from wavenet import utils, model, train


def test_sample_from_logits():
    logits = torch.ones(1, 4, 2, 1) / 4
    sample = utils.sample_from_logits(logits)
    assert sample.shape == (1, 2, 1)


def test_hparams_dict():
    class TestHParams(utils.HParams):
        a = 'b'
        foo = 'bar'

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    p = TestHParams(foo='DELETED', baz='qux')
    assert dict(p) == {'a': 'b', 'foo': 'DELETED', 'baz': 'qux'}


def test_lrfinder():
    m = model.Wavenet(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-8)
    schedule = utils.lrfinder(optimizer, 9, train.HParams(batch_size=1, max_epochs=1))
    assert np.isclose(schedule.gamma, 10.)


def test_onecycle():
    cfg = train.HParams(batch_size=1, max_epochs=1)
    m = model.Wavenet(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate)
    schedule = utils.onecycle(optimizer, 9, cfg)
    assert schedule.total_steps == 9
