import numpy as np
import torch
from torch.nn import functional as F

from wavenet import model, train, utils


def test_hparams():
    p = model.HParams()
    assert p.n_chans == 256
    assert p.n_logits() == 512


def test_wavenet_output_shape():
    m = model.Wavenet(model.HParams(n_channels=2))
    x, loss = m.forward(torch.randint(5, (3, 2, 4)).float())
    assert x.shape == (3, 256, 2, 4)


def test_bimodally_distributed_stereo_at_t0_then_silence():
    p, n_examples, n_samples = model.HParams(), 128, 4
    X = utils.stereo_impulse_at_t0(n_examples, n_samples,  p)
    m = model.Wavenet(model.HParams(n_channels=2))
    t = train.Trainer(m, X, None, train.HParams(max_epochs=1))
    t.train()
