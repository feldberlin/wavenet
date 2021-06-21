from pathlib import Path
import inspect
import os
import random
import yaml

from torch.nn import functional as F
from torch.optim import lr_scheduler
import numpy as np  # type: ignore
import torch
import wandb  # type: ignore


# base directory that wandb restores old runs to.
WANDB_RESTORE_DIR = Path("wandb") / "restore"


# logits and normalisation


def audio_to_class_idxs(audio: torch.IntTensor, n_classes):
    "Convert audio [-128, 127] to class indices [0, 255]."
    assert audio.min() >= -n_classes // 2, audio.min()
    assert audio.max() <= n_classes // 2 - 1, audio.max()
    return (audio + n_classes // 2).long()


def audio_from_class_idxs(idxs, n_classes):
    "Convert class indices [0, 255] to audio [-128, 127]."
    assert idxs.min() >= 0, idxs.min()
    assert idxs.max() <= n_classes - 1, idxs.max()
    return idxs - n_classes // 2


# generator decoders, randomness


def decode_random(logits):
    "Convert N, K, C, 1 logits into N, C, 1 samples by random sampling"
    N, K, C, W = logits.shape
    assert W == 1, W
    posterior = F.softmax(logits, dim=1)
    posterior = posterior.squeeze(-1).permute(0, 2, 1)
    d = torch.distributions.Categorical(posterior)
    return d.sample().unsqueeze(-1)


def decode_argmax(logits):
    "Convert N, K, C, 1 logits into N, C, 1 samples by argmax"
    N, K, C, W = logits.shape
    assert W == 1, W
    return torch.argmax(F.softmax(logits, dim=1), dim=1)


def decode_nucleus(core_mass: float = 0.95):
    """Convert N, K, C, 1 logits into N, C, 1 samples by nucleus sampling"
    as proposed in https://arxiv.org/pdf/1904.09751.pdf. core_mass is the
    retained probablity mass. Seeting core_mass to 0. is equivalent to argmax.
    """

    def fn(logits):
        N, K, C, W = logits.shape
        assert W == 1, W
        sorted, idxs = torch.sort(logits, dim=1, descending=True)
        _, reverse_idxs = torch.sort(idxs, dim=1)
        csum = torch.cumsum(F.softmax(sorted, dim=1), dim=1)
        remove = csum > core_mass
        remove[:, 0] = False  # always include the top probability
        logits[torch.gather(remove, 1, reverse_idxs)] = -float("Inf")
        return decode_random(logits)

    return fn


def seed(cfg):
    "Set random seeds and use deterministic algorithms. "
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.use_deterministic_algorithms)


# schedules


def lrfinder(optimizer, n_examples, cfg):
    start_lr, final_lr = 1e-8, 10.0
    n_steps = cfg.n_steps(n_examples)
    gamma = (final_lr / start_lr) ** (1 / n_steps)
    return lr_scheduler.ExponentialLR(optimizer, gamma)


def onecycle(optimizer, n_examples, cfg):
    lr = cfg.learning_rate
    n_steps = cfg.n_steps(n_examples)
    return lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_steps)


def lr_schedule(train_cfg, n_examples, optimizer):
    if train_cfg.finder:
        return lrfinder(optimizer, n_examples, train_cfg)
    elif train_cfg.onecycle:
        return onecycle(optimizer, n_examples, train_cfg)


# lifecycle


def load_chkpt(m, run_path, kind: str = 'best.test'):
    chkpt = wandb_restore(f"checkpoints.{kind}", run_path)
    m.load_state_dict(torch.load(chkpt.name))
    return m


# config


def cfgdict(model_cfg, train_cfg):
    return {**dict(model_cfg), "train": dict(train_cfg)}


class HParams:
    "Make HParams iterable so we can call dict on it"

    def __iter__(self):
        def f(obj):
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("__") and not inspect.isfunction(v)
            }

        return iter({**f(self.__class__), **f(self)}.items())


def load_hparams(path):
    "Load model, train cfgs from wandb formatted yaml"
    p = yaml.safe_load(path)
    del p["_wandb"]
    del p["wandb_version"]
    return (
        {k: v["value"] for k, v in p.items() if k != "train"},
        p.pop("train")["value"],
    )


def init_wandb(model, train_cfg, dataset_name: str):
    "Start up wandb"
    wandb.init(project=train_cfg.project_name)
    wandb.config.update(cfgdict(model.cfg, train_cfg))
    wandb.config.update({"dataset": dataset_name})
    wandb.watch(model, log="all")
    wandb.save(os.path.join(wandb.run.dir, "checkpoints.*"))  # type: ignore
    if train_cfg.finder:
        wandb.config.update({"dataset": "lrfinder"})


def finish_wandb():
    "Collect the final telemetry data"
    wandb.save(os.path.join(wandb.run.dir, "checkpoints.*"))
    wandb.finish()


def wandb_restore(filename, run_path):
    root = WANDB_RESTORE_DIR.joinpath(run_path)
    root.mkdir(parents=True, exist_ok=True)
    return wandb.restore(filename, run_path=run_path, root=root, replace=True)


def load_wandb_cfg(run_path):
    "Load model and train cfg from wandb"
    return load_hparams(wandb_restore("config.yaml", run_path))
