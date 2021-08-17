"""
Training loop
"""

import math
import os
import typing
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
import torch.nn.parallel as parallel

import wandb  # type: ignore
from wavenet import utils


class DistributedTrainer()
    "Single machine multi gpu training"

    def train():
        mp.set_start_method('forkserver')
        ngpus = torch.cuda.device_count()
        port = np.random.randint(15000, 15025)
        mp.spawn(worker, nprocs=ngpus, args=(ngpus, port))

    def setup(cfg, rank, world_size, port):
        cfg.batch_size = int(cfg.batch_size / world_size)
        cfg.num_workers = int((cfg.num_workers + world_size - 1) / world_size)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            torch.distributed.Backend.NCCL,
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=world_size,
            rank=rank
        )

    def teardown():
        dist.destroy_process_group()


class Trainer:
    """Train wavenet with mixed precision on a one cycle schedule."""

    def __init__(self, model, trainset, testset, cfg, callback=None):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        self.model_cfg = model.cfg
        self.callback = callback
        self.device = self.model_cfg.device()
        self.model = self.model.to(self.device)
        self.model = parallel.DistributedDataParallel(self.model)
        self.scaler = amp.GradScaler(enabled=self.model_cfg.mixed_precision)
        self.optimizer = self.cfg.optimizer(self.model)
        self.schedule = utils.lr_schedule(cfg, len(trainset), self.optimizer)
        self.metrics = utils.init_wandb(model, cfg, repr(self.trainset))

    def train(self):
        model, cfg, model_cfg = self.model, self.cfg, self.model_cfg

        # see gh #21
        torch.backends.cudnn.benchmark = True

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.trainset if is_train else self.testset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )

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

                    # logging
                    msg = f"{epoch+1}:{it} loss {loss.item():.5f} lr {lr:e}"
                    pbar.set_description(msg)
                    utils.log_wandb("learning rate", lr)
                    utils.log_wandb("train loss", loss.item())

                if self.callback and it % cfg.callback_fq == 0:
                    self.callback.tick(model, self.trainset, self.testset)

            return float(np.mean(losses))

        best = defaultdict(lambda: float("inf"))
        for epoch in range(cfg.max_epochs):

            train_loss = run_epoch("train")
            if train_loss < best["train"]:
                best["train"] = train_loss
                self.checkpoint("best.train", epoch)

            if self.testset is not None:
                test_loss = run_epoch("test")
                utils.log_wandb("test loss", test_loss)
                if test_loss < best["test"]:
                    best["test"] = test_loss
                    self.checkpoint("best.test", epoch)

    def checkpoint(self, name, epoch):
        base = wandb.run.dir if wandb.run.dir != "/" else "."
        filename = os.path.join(base, self.cfg.ckpt_path(name))
        torch.save(self._state(epoch), filename)
        wandb.save(filename, base_path=base)

    def restore(self, run_path, kind='train'):
        chkpt = utils.wandb_restore(f"checkpoints.{kind}", run_path)
        state_dict = torch.load(chkpt.name)
        self._model().load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
        self.schedule.load_state_dict(state_dict["schedule"])
        return state_dict["epoch"]

    def _model(self):
        is_data_paralell = hasattr(self.model, "module")
        return self.model.module if is_data_paralell else self.model

    def _state(self, epoch):
        return {
            "model": self._model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "schedule": self.schedule.state_dict(),
            "epoch": epoch,
        }


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

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def ckpt_path(self, name):
        return f"checkpoints.{name}"

    def n_steps(self, n_examples):
        batch_size = min(n_examples, self.batch_size)
        return math.ceil(n_examples / batch_size) * self.max_epochs

    def optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, betas=self.betas
        )
