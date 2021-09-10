import test_helpers as helpers
import torch

from wavenet import model, train, utils

# config


def test_hparams_dict():
    class TestHParams(utils.HParams):
        a = "b"
        foo = "bar"

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    p = TestHParams(foo="DELETED", baz="qux")
    assert dict(p) == {"a": "b", "foo": "DELETED", "baz": "qux"}


def test_load_hparams():
    with open("fixtures/config.yaml", "r") as f:
        p, ptrain = utils.load_hparams(f)
    assert p["dilation_stacks"] == 3
    assert not p.get("train", None)
    assert ptrain["batch_size"] == 64


# schedules


def test_lrfinder():
    m = model.Wavenet(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-8)
    p = train.HParams(batch_size=1, max_epochs=1)
    schedule = utils.lrfinder(optimizer, 9, p)
    assert torch.isclose(torch.tensor(schedule.gamma), torch.tensor(10.0))


def test_onecycle():
    cfg = train.HParams(batch_size=1, max_epochs=1)
    m = model.Wavenet(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate)
    schedule = utils.onecycle(optimizer, 9, cfg)
    assert schedule.total_steps == 9


# decoders


def test_decode_random():
    logits = torch.ones(1, 4, 2, 1) / 4
    sample = utils.decode_random(logits)
    assert sample.shape == (1, 2, 1)


def test_decode_argmax():
    logits = torch.tensor(
        [[[[1.0], [1.0]], [[4.0], [4.0]], [[2.0], [2.0]], [[3.0], [3.0]]]]
    )

    assert utils.decode_argmax(logits).equal(torch.tensor([[[1], [1]]]))


def test_decode_nucleus():
    logits = torch.tensor(
        [[[[1.0], [1.0]], [[4.0], [4.0]], [[2.0], [2.0]], [[3.0], [3.0]]]]
    )

    assert utils.decode_nucleus()(logits).shape == (1, 2, 1)


def test_decode_with_zero_nucleus_is_equivalent_to_argmax():
    logits = torch.tensor(
        [[[[1.0], [1.0]], [[4.0], [4.0]], [[2.0], [2.0]], [[3.0], [3.0]]]]
    )

    want = utils.decode_argmax(logits)
    got = utils.decode_nucleus(0.0)(logits)
    assert torch.equal(want, got)


# checkpointing


def test_checkpoint():
    with helpers.tempdir() as tmp:
        p = model.HParams()
        m = model.Wavenet(p)
        tp = train.HParams()
        t = train.Trainer(m, [1, 2, 3], [4, 5], tp)
        filename = tmp / "checkpoint"
        utils.checkpoint("test", t.state(), tp, filename)
