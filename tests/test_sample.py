import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from wavenet import datasets, model, sample, train, utils


def test_generator_init():
    m = model.Wavenet(model.HParams())
    assert sample.Generator(m)


def test_to_conv_1d():
    x = torch.rand((1, 10, 10))
    c = nn.Conv1d(10, 20, 2)
    c_prime = sample.to_conv1d(c)
    assert torch.all(c(x) == c_prime(x))


def test_generator_forward_one_sample():
    m = model.Wavenet(model.HParams(n_classes=2 ** 8))
    g = sample.Generator(m)
    y = torch.randint(0, 2 ** 8, (2, 2, 1))
    x = y.float()
    y, loss = g.forward(x, y)
    assert y.shape == (2, m.cfg.n_classes, m.cfg.n_audio_chans, 1)


def test_shifted_weights_generator_vs_wavenet():
    m = model.Wavenet(model.HParams())
    g = sample.Generator(m)
    assert torch.equal(m.shifted.weight, g.shifted.c.weight)


def test_shifted_units_generator_vs_wavenet_one_sample():
    p = model.HParams(
        mixed_precision=False,
        n_audio_chans=1,
        n_classes=16,
        n_chans=32,
        dilation_stacks=1,
        n_layers=6,
        compress=False,
    )

    m = model.Wavenet(p)
    g = sample.Generator(m)
    x = torch.rand((1, p.n_audio_chans, 1))
    ym = m.shifted.forward(x).squeeze()
    yg = g.shifted.forward(F.pad(x, (1, -1))).squeeze()  # causal
    assert torch.all(ym == yg)


def test_one_sample():
    m = model.Wavenet(model.HParams())
    tf = datasets.AudioUnitTransforms(m.cfg)
    track, *_ = sample.fast(m, tf, utils.decode_random, n_samples=1)
    assert track.shape == (1, m.cfg.n_audio_chans, 1)


def test_two_samples():
    m = model.Wavenet(model.HParams())
    tf = datasets.AudioUnitTransforms(m.cfg)
    track, *_ = sample.fast(m, tf, utils.decode_random, n_samples=2)
    assert track.shape == (1, m.cfg.n_audio_chans, 2)


def test_many_samples():
    m = model.Wavenet(model.HParams())
    tf = datasets.AudioUnitTransforms(m.cfg)
    track, *_ = sample.fast(m, tf, utils.decode_random, n_samples=50)
    assert track.shape == (1, m.cfg.n_audio_chans, 50)


def test_many_samples_with_embedding():
    m = model.Wavenet(model.HParams(embed_inputs=True))
    tf = datasets.AudioUnitTransforms(m.cfg)
    track, *_ = sample.fast(m, tf, utils.decode_random, n_samples=50)
    assert track.shape == (1, m.cfg.n_audio_chans, 50)


def test_one_logit_generator_vs_wavenet():
    p = model.HParams(
        mixed_precision=False,
        n_audio_chans=1,
        n_classes=16,
        n_chans=32,
        dilation_stacks=1,
        n_layers=6,
        compress=False,
    )

    m = model.Wavenet(p)
    g = sample.Generator(m)
    x = torch.rand((1, m.cfg.n_audio_chans, 1))

    # forward pass through original with a single example
    ym, _ = m.forward(x)
    ym = F.softmax(ym.squeeze(), dim=0)

    # forward pass copy with a single example
    yg, _ = g.forward(F.pad(x, (1, -1)))  # causal
    yg = F.softmax(yg.squeeze(), dim=0)

    assert torch.all(ym == yg)


def test_memoed_causal1d():
    """Memoed Causal1d must retain the padding behavior of the original.

    That is, it should left pad by (kernel - 1) * dilation. In this case, we
    have kernel size of 2 and no dilation factor, so we should have a left
    padding of 1. This results in an output of length m given an input of
    length m.

    To understand why this behavior is guaranteed, compare the non memoed vs
    the memoed behavior.

    - Causal1d left pads by 1. This ensures m -> m output size.
    - Causal1d left pads by 1. The first value will therefore be therefore be
      the right side of the kernel multiplied by the input, shifted by the
      convolutional bias.

    In the memoed Causal1d, we are using a normal conv1d. There is no padding.

    - conv1d inputs a single (N, C, W=1) value, which is the first value of
      the input.
    - When there's nothing in the queue, memo will pull in zeros to the left.
      For the first sample into the convolution, we therefore have [0., x_0].
    - This is the same as left padding with one zero.
    - The remaining values will have the same effect as a normal convolution.
    """

    N, C, W = (1, 1, 8)
    dilation = 1
    kernel_size = 2
    conv = model.Causal1d(1, 1, kernel_size, dilation=dilation)

    # with a kernel size 2, you have to remember 1 past input element. this is
    # combined with the current element in order to compute the output.
    memoed = sample.Memo(conv)

    res = []
    x = torch.rand((N, C, W))
    for i in range(W):
        step = memoed(x[:, :, i : i + 1])
        res.append(step)

    want = conv(x)
    got = torch.cat(res, axis=2)
    assert torch.allclose(want, got)


def test_memoed_causal1d_dilated():
    """Memoed Causal1d must retain the padding behavior of the original.

    That is, it should left pad by (kernel - 1) * dilation. Same deal as
    above, except that now we left pad by e.g. 2, with a dilation of 2 and
    kernel size of 1. Again, this is guaranteed by the left zero padding
    behavior of memo, which continues until it finds the first dillated input
    value.
    """

    N, C, W = (1, 1, 8)
    dilation = 2
    kernel_size = 2
    conv = model.Causal1d(1, 1, kernel_size, dilation=dilation)

    # with a kernel size 2, you have to remember 1 past input element. this is
    # combined with the current element in order to compute the output.
    memoed = sample.Memo(conv)

    res = []
    x = torch.rand((N, C, W))
    for i in range(W):
        res.append(memoed(x[:, :, i : i + 1]))

    # want the same padding behavior as Causal1d
    want = conv(x)
    got = torch.cat(res, axis=2)
    assert torch.allclose(want, got)


def test_memoed_shifted_causal1d():
    """The behavior is a bit different here. Before, the first input value was
    x_0. Here, the first input value is always zero, followed by x_0, x_1,
    etc. On top of this, we have a left padding of one, as before.

    In the source network, this is implemented with a right rotation of x,
    followed by left padding.
    """

    p = model.HParams()
    utils.seed(p)  # reset seeds and use deterministic mode

    N, C, W = (1, 1, 8)
    dilation = 1
    kernel_size = 2
    conv = model.ShiftedCausal1d(1, 1, kernel_size, dilation=dilation)

    # the  expected behavior
    x = torch.rand((N, C, W))
    want = conv(x)

    # with a kernel size 2, you have to remember 1 past input element. this is
    # combined with the current element in order to compute the output.
    memoed = sample.Memo(conv)

    res = []
    x = F.pad(x, (1, -1))
    for i in range(W):
        step = memoed(x[:, :, i : i + 1])
        res.append(step)

    # want the same padding behavior as ShiftedCausal1d
    got = torch.cat(res, axis=2)
    assert torch.allclose(want, got)


@pytest.mark.parametrize("n_audio_chans", [1, 2])
@pytest.mark.parametrize("embed_inputs", [False, True])
def test_many_logits_fast_vs_simple(embed_inputs, n_audio_chans):
    n_samples, n_examples = 100, 5
    p = model.HParams(
        mixed_precision=False,
        embed_inputs=embed_inputs,
        n_audio_chans=n_audio_chans,
        n_classes=20,
        n_chans=16,
        dilation_stacks=1,
        n_layers=2,
        compress=False,
        sample_length=n_samples,
        seed=135,
        verbose=True,
    )

    utils.seed(p)
    ds = datasets.Tiny(n_samples, n_examples)
    m = model.Wavenet(p)

    def decoder(logits):
        utils.seed(p)
        return utils.decode_random(logits)

    # simple
    utils.seed(p)
    _, simple_logits = sample.simple(
        m, ds.transforms, decoder, n_samples=n_samples, batch_size=n_examples
    )
    simple_logits = torch.softmax(simple_logits.squeeze(), dim=0)

    # fast
    utils.seed(p)
    _, fast_logits, g = sample.fast(
        m, ds.transforms, decoder, n_samples=n_samples, batch_size=n_examples
    )
    fast_logits = torch.softmax(fast_logits.squeeze(), dim=0)

    assert torch.allclose(fast_logits, simple_logits)


def test_many_logits_generator_vs_wavenet():
    """This test doesn't do any sampling. Instead we compare logits. On the
    wavenet, this can be done with a single forward pass. The generator only
    accepts a one sample input, so we have to generate that by passing through
    one piece of the input at a time while appending all the outputs.
    """

    n_samples = 2
    p = model.HParams(
        mixed_precision=False,
        n_audio_chans=1,
        n_classes=16,
        n_chans=32,
        dilation_stacks=1,
        n_layers=1,
        compress=False,
    )

    utils.seed(p)  # reset seeds and use deterministic mode

    # set up model and generator
    m = model.Wavenet(p)
    g = sample.Generator(m)

    # a single forward pass through the wavenet
    x = torch.rand((1, m.cfg.n_audio_chans, n_samples))
    ym, _ = m.forward(x)

    # iterate forward on generator to accumulate all of the logits that would
    # have been output on a single forward pass of a random input. that's what
    # we see in the line above, for the underlying wavenet. the generator,
    # however, only processes a single sample at a time.
    yg = None
    x = F.pad(x, (1, -1))
    for i in range(n_samples):
        timestep = x[:, :, i : (i + 1)]
        logits, _ = g.forward(timestep)
        if yg is not None:
            yg = torch.cat([yg, logits], -1)
        else:
            yg = logits

    # posterior
    ym = F.softmax(ym.squeeze(), dim=0)
    yg = F.softmax(yg.squeeze(), dim=0)

    assert torch.allclose(ym, yg)
