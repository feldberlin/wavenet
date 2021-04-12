import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd.functional import jacobian

from wavenet import model, utils, datasets

import pytest


def test_hparams():
    p = model.HParams()
    assert p.n_chans == 256
    assert p.n_logits() == 512


def test_hparams_override():
    p = model.HParams(n_chans=2)
    assert p.n_chans == 2


def test_wavenet_output_shape():
    m = model.Wavenet(model.HParams())
    y = torch.randint(256, (3, 2, 4))
    x = y.float()
    y_hat, _ = m.forward(x, y)
    assert y_hat.shape == (3, 256, 2, 4)


def test_wavenet_mono_output_shape():
    m = model.Wavenet(model.HParams(n_audio_chans=1))
    y = torch.randint(256, (3, 1, 4))
    x = y.float()
    x, _ = m.forward(x, y)
    assert x.shape == (3, 256, 1, 4)


def test_wavenet_modules_registered():
    m = model.Wavenet(model.HParams(n_layers=1, dilation_stacks=1))
    got = list(m.state_dict().keys())
    want = [
        'input.weight',
        'input.bias',
        'layers.0.conv.weight',
        'layers.0.conv.bias',
        'layers.0.end1x1.weight',
        'layers.0.end1x1.bias',
        'layers.0.skip1x1.weight',
        'layers.0.skip1x1.bias',
        'a1x1.weight',
        'a1x1.bias',
        'b1x1.weight',
        'b1x1.bias'
    ]

    assert got == want


def test_logit_jacobian_first_sample():
    p = model.HParams()
    X = datasets.StereoImpulse(1, 1,  p)
    m = model.Wavenet(p)

    def logits(x):
        "we are only interested in the time dimensions W. keeping n for loss"
        logits, _ = m.forward(x)
        return logits.sum((1, 2))  # N, K, C, W -> N, W

    # input is N, C, W. output is N, W. jacobian is N, W, N, C, W
    x, _ = X[0]
    j = jacobian(logits, x.unsqueeze(0))

    # sum everything else to obtain WxW
    j = j.sum((0, 2, 3))

    # gradients must remain unaffected by the input
    assert torch.unique(j) == torch.zeros(1)


def test_logit_jacobian_many_samples():
    p = model.HParams()
    X = datasets.StereoImpulse(1, 8,  p)  # 8 samples
    m = model.Wavenet(p)

    def logits(x):
        "we are only interested in the time dimensions W. keeping n for loss"
        logits, _ = m.forward(x)
        return logits.sum((1, 2))  # N, K, C, W -> N, W

    # input is N, C, W. output is N, W. jacobian is N, W, N, C, W
    x, _ = X[0]
    j = jacobian(logits, x.unsqueeze(0))

    # sum everything else to obtain WxW
    j = j.sum((0, 2, 3))

    # jacobian must be lower triangular
    assert torch.equal(torch.tril(j), j)


def test_loss_jacobian_many_samples():
    p = model.HParams()
    X = datasets.StereoImpulse(1, 8,  p)  # 8 samples
    m = model.Wavenet(p)

    def loss(x):
        logits, _ = m.forward(x)
        targets = utils.audio_to_class_idxs(x, p.n_classes)
        losses = F.cross_entropy(logits, targets, reduction='none')
        return losses.sum(1)  # N, C, W -> N, W

    # input is N, C, W. output is N, W. jacobian is N, W, N, C, W
    x, _ = X[0]
    j = jacobian(loss, x.unsqueeze(0))

    # sum everything else to obtain WxW
    j = j.sum((0, 2, 3))

    # jacobian must be lower triangular
    assert torch.equal(torch.tril(j), j)


@pytest.mark.integration
def test_loss_stable_across_batch_sizes():
    batch_sizes = {1: None, 100: None}
    for k in batch_sizes.keys():
        losses = []
        for i in range(100):
            p = model.HParams()
            x, y = datasets.to_tensor(datasets.StereoImpulse(k, 8,  p))
            m = model.Wavenet(p)
            _, loss = m.forward(x, y)
            losses.append(loss.detach().numpy())
        batch_sizes[k] = (np.mean(losses), np.std(losses))

    means = [v[0] for v in batch_sizes.values()]
    assert np.std(means) < 0.25, batch_sizes
