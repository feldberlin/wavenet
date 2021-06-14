import typing

from prettytable import PrettyTable  # type: ignore
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

    def hook_fn(module, input, output):
        memo[module] = output

    def register(modules: typing.Dict[str, nn.Module]):
        for name, layer in modules.items():
            if isinstance(layer, nn.ModuleList):
                register(layer._modules)
            else:
                print("registered", name, layer)
                layer.register_forward_hook(hook_fn)

    register(m._modules)


def dot(m: model.Wavenet):
    "Generate a graphviz graph of the network execution graph."

    N, C, W = 1, m.cfg.n_audio_chans, 10
    x = torch.rand((N, C, W))
    y, *_ = m(x)
    return torchviz.make_dot(
        y.mean(),
        params=dict(m.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )


def count_parameters(model) -> typing.Tuple[PrettyTable, int]:
    "Returns a printable table of module parameter counts and the total"

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    return table, total_params


def summarize(model):
    "Pretty print some useful model information."

    table, n_parameters = count_parameters(model)
    print(table)
    print(f'receptive field size: { model.cfg.receptive_field_size() }')
    print(f'model total params: { n_parameters }')
