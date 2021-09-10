"""
Training loop
"""

import math
import typing
from collections import defaultdict

import numpy as np  # type: ignore
import torch
import torch.cuda.amp as amp
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm  # type: ignore

from wavenet import utils


class Trainer:
    """Training loop."""

    def __init__(self, model, trainset, testset, cfg, log=True, sampler=None):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        self.log = log
        self.sampler = sampler
        self.model_cfg = utils.unwrap(model).cfg
        self.device = self.model_cfg.device
        self.scaler = amp.GradScaler(enabled=self.model_cfg.mixed_precision)
        self.optimizer = self.cfg.optimizer(utils.unwrap(model))
        self.schedule = utils.lr_schedule(cfg, len(trainset), self.optimizer)
        self.epoch = 0
        if log:
            self.metrics = utils.init_wandb(
                utils.unwrap(model), cfg, repr(self.trainset)
            )

    def train(self):
        model, cfg, model_cfg = self.model, self.cfg, self.model_cfg

        # see gh #21
        torch.backends.cudnn.benchmark = True
        self.model = self.model.to(self.device)

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.trainset if is_train else self.testset

            if self.sampler:
                # must be called before data loader init
                self.sampler.set_epoch(self.epoch)

            # loader
            loader = DataLoader(
                data,
                shuffle=not self.sampler,
                pin_memory=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                sampler=self.sampler,
            )

            # progress
            no_pbar = not cfg.progress_bar
            pbar = (
                tqdm(enumerate(loader), total=len(loader), disable=no_pbar)
                if is_train
                else enumerate(loader)
            )

            losses = []
            for it, (x, y, *_) in pbar:

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    with amp.autocast(enabled=model_cfg.mixed_precision):
                        logits, loss = model(x, y)
                        loss = loss.mean()  # collect gpus
                        losses.append(loss.item())

                if is_train:
                    # see gh #21
                    for param in model.parameters():
                        param.grad = None

                    self.scaler.scale(loss).backward()
                    if cfg.grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_norm_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.schedule:
                        self.schedule.step()
                        lr = self.schedule.get_last_lr()[0]
                    else:
                        lr = cfg.learning_rate

                    # progress
                    pbar.set_description(
                        f"{self.epoch+1}:{it} loss {loss.item():.5f} lr {lr:e}"
                    )

                    # log
                    self.logger("learning rate", lr)
                    self.logger("train loss", loss.item())

            return float(np.mean(losses))

        best = defaultdict(lambda: float("inf"))
        for epoch in range(cfg.max_epochs):
            self.epoch = epoch
            train_loss = run_epoch("train")
            if train_loss < best["train"]:
                best["train"] = train_loss
                self.checkpoint("best.train")

            if self.testset is not None:
                test_loss = run_epoch("test")
                self.logger("test loss", test_loss)
                if test_loss < best["test"]:
                    best["test"] = test_loss
                    self.checkpoint("best.test")

    def logger(self, key, value):
        if self.log:
            utils.log_wandb(key, value)

    def checkpoint(self, name):
        if self.log:
            utils.checkpoint(name, self.state(), self.cfg)

    def state(self):
        return {
            "model": utils.unwrap(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "schedule": self.schedule.state_dict(),
            "epoch": self.epoch,
        }

    def load_state(self, state):
        utils.unwrap(self.model).load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.schedule.load_state_dict(state["schedule"])
        self.epoch = state["epoch"]


class HParams(utils.HParams):

    # wandb project
    project_name: str = "feldberlin-wavenet"

    # once over the whole dataset, how many times max
    max_epochs: int = 10

    # number of examples in a single batch
    batch_size: int = 64

    # the learning rate
    learning_rate: float = 3e-4

    # apply a one cycle schedule
    onecycle: bool = True

    # adam betas
    betas: typing.Tuple[float, float] = (0.9, 0.95)

    # training loop clips gradients
    grad_norm_clip: typing.Optional[float] = None

    # how many steps before the callback is invoked
    callback_fq: int = 8

    # how many data loader threads to use
    num_workers: int = 0

    # is this a learning rate finder run
    finder: bool = False

    # random seed
    seed: int = 5763

    # show progress bar
    progress_bar: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def ckpt_path(self, name):
        return f"checkpoints.{name}"

    def n_steps(self, n_examples):
        batch_size = min(n_examples, self.batch_size)
        return math.ceil(n_examples / batch_size) * self.max_epochs

    def resize(self, percentage):
        self.batch_size = int(self.batch_size * percentage)
        self.num_workers = int(self.num_workers * percentage)

    def optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, betas=self.betas
        )
