import torch

from wavenet import utils


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
    assert dict(p) == { 'a': 'b', 'foo': 'DELETED', 'baz': 'qux' }
