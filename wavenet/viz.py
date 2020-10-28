"""
Notebook tools
"""

import matplotlib.pyplot as plt
from torch.nn import functional as F


def plot_stereo_sample_distributions(logits, n):
    N, K, C, W = logits.shape

    def channels(pos):
        l = F.softmax(logits[n, :, 0, pos], dim=0).detach().numpy()
        r = F.softmax(logits[n, :, 1, pos], dim=0).detach().numpy()
        return l, r

    fig, axs = plt.subplots(1, W, figsize=(W * 9, 8))
    for i, ax in enumerate(axs):
        ll, rr = channels(i)
        ax.bar(list(range(len(ll))), ll)
        ax.bar(list(range(len(rr))), rr)

    plt.tight_layout()


def animate_stereo_sample_distributions(camera, axs, logits, n):
    N, K, C, W = logits.shape

    def channels(pos):
        l = F.softmax(logits[n, :, 0, pos], dim=0).detach().numpy()
        r = F.softmax(logits[n, :, 1, pos], dim=0).detach().numpy()
        return l, r

    for i, ax in enumerate(axs):
        ll, rr = channels(i)
        ax.bar(list(range(len(ll))), ll)
        ax.bar(list(range(len(rr))), rr)
        camera.snap()

    plt.tight_layout()


class LearningAnimation():

    def __initialize__(self, W):
        fig, self.axs = plt.subplots(1, W, figsize=(W * 9, 8))
        self.camera = Camera(fig)

    def tick(self, X):
        logits = X[:128]
        animate_stereo_sample_distributions(self.camera, self.axs, logits, 0)

    def render(self, fileaname):
        animation = self.camera.animate()
        animation.save(filename, writer = 'imagemagick')
