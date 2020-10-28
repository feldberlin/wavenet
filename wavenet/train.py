"""
Training loop
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, model, trainset, testset, cfg, callback):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        self.callback = callback
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def checkpoint(self):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.cfg.ckpt_path)
        torch.save(raw.state_dict(), self.cfg.ckpt_path)

    def train(self):
        model, cfg = self.model, self.cfg
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas
        )

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

            for it, x in pbar:

                x = x.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x)
                    loss = loss.mean()  # collect gpus
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_norm_clip
                    )
                    optimizer.step()
                    lr = cfg.learning_rate

                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

                if self.callback and it % self.cfg.callback_fq == 0:
                    self.callback.tick(self.model, self.trainset, self.testset)

            if not is_train:

                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float("inf")
        for epoch in range(cfg.max_epochs):

            run_epoch("train")
            if self.testset is not None:
                test_loss = run_epoch("test")

            # early stopping, or just save always if no test set is provided
            good_model = self.testset is None or test_loss < best_loss
            if self.cfg.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.checkpoint()


class HParams:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    ckpt_path = None
    callback_fq = 8
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
