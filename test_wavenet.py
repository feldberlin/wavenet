import torch
from torch.nn import functional as F
from torch.autograd.functional import jacobian

from wavenet import model, utils, datasets


def test_hparams():
    p = model.HParams()
    assert p.n_chans == 256
    assert p.n_logits() == 512


def test_hparams_override():
    p = model.HParams(n_chans=2)
    assert p.n_chans == 2


def test_wavenet_output_shape():
    m = model.Wavenet(model.HParams())
    x, loss = m.forward((torch.rand(3, 2, 4) * 2 - 1).float())
    assert x.shape == (3, 256, 2, 4)


def test_wavenet_mono_output_shape():
    m = model.Wavenet(model.HParams(n_audio_chans=1))
    x, loss = m.forward((torch.rand(3, 1, 4) * 2 - 1).float())
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

    def logits(X):
        "we are only interested in the time dimensions W. keeping n for loss"
        logits, loss = m.forward(X)
        return logits.sum((1, 2))  # N, K, C, W -> N, W

    # input is N, C, W. output is N, W. jacobian is N, W, N, C, W
    j = jacobian(logits, X[:])

    # sum everything else to obtain WxW
    j = j.sum((0, 2, 3))

    # gradients must remain unaffected by the input
    assert torch.unique(j) == torch.zeros(1)


def test_logit_jacobian_many_samples():
    p = model.HParams()
    X = datasets.StereoImpulse(1, 8,  p)  # 8 samples
    m = model.Wavenet(p)

    def logits(X):
        "we are only interested in the time dimensions W. keeping n for loss"
        logits, loss = m.forward(X)
        return logits.sum((1, 2))  # N, K, C, W -> N, W

    # input is N, C, W. output is N, W. jacobian is N, W, N, C, W
    j = jacobian(logits, X[:])

    # sum everything else to obtain WxW
    j = j.sum((0, 2, 3))

    # jacobian must be lower triangular
    assert torch.equal(torch.tril(j), j)


def test_loss_jacobian_many_samples():
    p = model.HParams()
    X = datasets.StereoImpulse(1, 8,  p)  # 8 samples
    m = model.Wavenet(p)

    def loss(audio):
        logits, loss = m.forward(audio)
        targets = utils.quantized_audio_to_class_idxs(audio, p)
        losses = F.cross_entropy(logits, targets, reduction='none')
        return losses.sum(1)  # N, C, W -> N, W

    # input is N, C, W. output is N, W. jacobian is N, W, N, C, W
    j = jacobian(loss, X[:])

    # sum everything else to obtain WxW
    j = j.sum((0, 2, 3))

    # jacobian must be lower triangular
    assert torch.equal(torch.tril(j), j)
