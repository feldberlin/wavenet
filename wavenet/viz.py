"""
Notebook tools
"""

import numpy as np
import matplotlib.pyplot as plt
import celluloid
import torch
from torch.nn import functional as F


def plot_track(batch: torch.tensor, i: int = None,
               offset: int = 0, n_samples: int = 350, title: str = 'track',
               style: str = '-'):

    N, C, W = batch.shape
    i = i if i is not None else np.random.randint(N)

    plt.figure(figsize=(15, 7))
    for channel in range(C):
        data = batch[i, channel, i:i+n_samples]
        legend = f'{title}: track {i}, offset {offset}, n_samples {n_samples}'
        plt.title(legend)
        plt.plot(data, style)

    plt.tight_layout()
    return i


def plot_stereo_sample_distributions(logits: torch.tensor, n: int):
    N, K, C, W = logits.shape

    def channels(pos):
        left = F.softmax(logits[n, :, 0, pos], dim=0).detach().cpu().numpy()
        right = F.softmax(logits[n, :, 1, pos], dim=0).detach().cpu().numpy()
        return left, right

    fig, axs = plt.subplots(1, W, figsize=(W * 9, 8))
    axs = [axs] if W == 1 else axs
    for i, ax in enumerate(axs):
        ll, rr = channels(i)
        ax.bar(list(range(len(ll))), ll, color="#00f")
        ax.bar(list(range(len(rr))), rr, color="#0f0")

    plt.tight_layout()


def animate_stereo_sample_distributions(camera, axs, logits, n):
    N, K, C, W = logits.shape

    def channels(pos):
        left = F.softmax(logits[n, :, 0, pos], dim=0).detach().cpu().numpy()
        right = F.softmax(logits[n, :, 1, pos], dim=0).detach().cpu().numpy()
        return left, right

    for i, ax in enumerate(axs):
        ll, rr = channels(i)
        ax.bar(list(range(len(ll))), ll, color="#00f")
        ax.bar(list(range(len(rr))), rr, color="#0f0")

    plt.tight_layout()
    camera.snap()


class LearningAnimation():

    def __init__(self, W):
        plt.ioff()
        fig, self.axs = plt.subplots(1, W, figsize=(W * 9, 8))
        self.camera = celluloid.Camera(fig)
        if W == 1:
            self.axs = [self.axs]

    def tick(self, model, trainset, testset):
        x = trainset[:128][0]
        logits, _ = model.forward(x)
        animate_stereo_sample_distributions(self.camera, self.axs, logits, 0)

    def render(self, filename):
        animation = self.camera.animate()
        animation.save(filename, writer='imagemagick')
