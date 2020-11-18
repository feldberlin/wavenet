"""
Training loop
"""

from collections import defaultdict
import logging
import math
import os

from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from wavenet import utils

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, model, trainset, testset, cfg, callback):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        self.callback = callback
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def checkpoint(self, name):
        raw = self.model.module if hasattr(self.model, 'module') else self.model
        filename = os.path.join(wandb.run.dir, self.cfg.ckpt_path(name))
        torch.save(raw.state_dict(), filename)

    def train(self):
        model, cfg = self.model, self.cfg
        raw_model = model.module if hasattr(self.model, 'module') else model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas
        )

        # telemetry
        wandb.init(project=cfg.project_name)
        wandb.config.update({ **dict(self.model.cfg), 'train': dict(self.cfg) })
        wandb.watch(model, log='all')

        def run_epoch(split):
            is_train = split == 'train'
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

            for it, x in pbar:

                x = x.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x)
                    loss = loss.mean()  # collect gpus
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    if cfg.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_norm_clip
                        )
                    optimizer.step()
                    lr = cfg.learning_rate
                    msg = f'{epoch+1}:{it} loss {loss.item():.5f} lr {lr:e}'
                    wandb.log({'train loss': loss})
                    pbar.set_description(msg)

                if self.callback and it % self.cfg.callback_fq == 0:
                    self.callback.tick(self.model, self.trainset, self.testset)

            return float(np.mean(losses))

        best = defaultdict(lambda: float('inf'))
        for epoch in range(cfg.max_epochs):

            train_loss = run_epoch('train')
            if train_loss < best['train']:
                best['train'] = train_loss
                self.checkpoint('best.train')

            if self.testset is not None:
                test_loss = run_epoch('test')
                wandb.log({'test loss': test_loss})
                if test_loss < best['test']:
                    best['test'] = test_loss
                    self.checkpoint('best.test')


class HParams(utils.HParams):

    # wandb project
    project_name = 'feldberlin-wavenet'

    # once over the whole dataset, how many times max
    max_epochs = 10

    # number of examples in a single batch
    batch_size = 64

    # the learning rate
    learning_rate = 3e-4

    # adam betas
    betas = (0.9, 0.95)

    # training loop clips gradients
    grad_norm_clip = None

    # how many steps before the callback is invoked
    callback_fq = 8

    # how many data loader threads to use
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def ckpt_path(self, name):
        return f'checkpoints.{name}'
