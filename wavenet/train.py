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

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def checkpoint(self):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas
        )

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
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
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()
                    lr = config.learning_rate

                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:

                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float("inf")
        for epoch in range(config.max_epochs):

            run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")

            # early stopping, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.checkpoint()


class HParams:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
