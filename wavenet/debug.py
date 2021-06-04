import typing

import torch
import torch.nn as nn
import torchviz  # type: ignore

from wavenet import model


def activations(m: nn.Module, memo: dict):
    """Sets up hooks on `m`. Collects fwd activations into `memo`.

    Example
    -------

    m = model.Wavenet(model.HParams())
    activations = {}
    utils.activations(m, activations)
    m(X)
    print(activations)
    """

    def hook_fn(module, input, output): memo[module] = output

    def register(modules: typing.Dict[str, nn.Module]):
        for name, layer in modules.items():
            if isinstance(layer, nn.ModuleList):
                register(layer._modules)
            else:
                print('registered', name, layer)
                layer.register_forward_hook(hook_fn)

    register(m._modules)


def dot(m: model.Wavenet):
    N, C, W = 1, m.cfg.n_audio_chans, 10
    x = torch.rand((N, C, W))
    y, *_ = m(x)
    return torchviz.make_dot(
        y.mean(),
        params=dict(m.named_parameters()),
        show_attrs=True,
        show_saved=True
    )
