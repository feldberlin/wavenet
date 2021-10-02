"""
Extends the training loop to be data distributed across multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data as data

from wavenet import train


class DP:
    """Training loop with nn.DataParallel. One machine only."""

    def __init__(self, model, trainset, testset, cfg, log=True):
        self.trainer = train.Trainer(
            nn.DataParallel(model), trainset, testset, cfg, log
        )
        self.log = log
        if log:
            self.metrics = self.trainer.metrics

    def train(self):
        self.trainer.train()

    def finish(self):
        if self.log:
            self.metrics.finish()


def worker(gpu: int, ngpus: int, port: int, model, trainset, testset, cfg):
    "Worker for DPP. This launches a single train.Trainer()"

    # set up multiprocessing
    dist.init_process_group(
        torch.distributed.Backend.NCCL,
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=ngpus,
        rank=gpu,
    )

    # configure the worker
    leader = gpu == 0
    device = torch.device(gpu)
    cfg.resize(1 / ngpus)
    model.cfg.device = device
    torch.cuda.set_device(device)

    # trainset sampler for distributed training
    train_sampler: data.Sampler = data.distributed.DistributedSampler(
        trainset,
        num_replicas=ngpus,
        rank=gpu,
        shuffle=True,
        seed=cfg.seed,
    )

    # testset sampler for distributed training
    test_sampler: data.Sampler = data.distributed.DistributedSampler(
        testset,
        num_replicas=ngpus,
        rank=gpu,
        shuffle=False,
        seed=cfg.seed,
    )

    # distribute the model
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True,
    )

    # configure a single trainer
    t = train.Trainer(
        model,
        trainset,
        testset,
        cfg,
        log=leader,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
    )

    # don't forget to train
    t.train()


class DDP:
    """Training loop with nn.DistributedDataParallel. One machine only."""

    def __init__(self, model, trainset, testset, cfg):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg

    def train(self):

        # fork the process
        mp.set_start_method("forkserver")
        ngpus = torch.cuda.device_count()
        port = torch.randint(15000, 15025, ()).item()
        mp.spawn(
            worker,
            nprocs=ngpus,
            args=(
                ngpus,
                port,
                self.model,
                self.trainset,
                self.testset,
                self.cfg.clone(),
            ),
        )

    def finish(self):
        dist.destroy_process_group()
