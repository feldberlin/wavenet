from wavenet import model

import torch


def test_hparams():
    p = model.HParams()
    assert p.n_chans == 256


def test_wavenet():
    m = model.Wavenet(model.HParams(n_channels=2))
    m.forward(torch.randint(5, (1, 2, 4)).float())
