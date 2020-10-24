from wavenet import model

import torch


def test_hparams():
    p = model.HParams()
    assert p.n_channels == 256

def test_wavenet():
    m = model.Wavenet(model.HParams())
    m.forward(torch.randn(100))
