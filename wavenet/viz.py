"""
Notebook tools
"""

import matplotlib.pyplot as plt
import celluloid
from torch.nn import functional as F


def plot_stereo_sample_distributions(logits, n):
    N, K, C, W = logits.shape

    def channels(pos):
        l = F.softmax(logits[n, :, 0, pos], dim=0).detach().numpy()
        r = F.softmax(logits[n, :, 1, pos], dim=0).detach().numpy()
        return l, r

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
        l = F.softmax(logits[n, :, 0, pos], dim=0).detach().numpy()
        r = F.softmax(logits[n, :, 1, pos], dim=0).detach().numpy()
        return l, r

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
        logits, _ = model.forward(trainset[:128])
        animate_stereo_sample_distributions(self.camera, self.axs, logits, 0)

    def render(self, filename):
        animation = self.camera.animate()
        animation.save(filename, writer='imagemagick')
