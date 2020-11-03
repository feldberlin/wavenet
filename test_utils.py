import torch

from wavenet import utils


def test_sample_from_logits():
    logits = torch.ones(1, 4, 2, 1) / 4
    sample = utils.sample_from_logits(logits)
    assert sample.shape == (1, 2, 1)

