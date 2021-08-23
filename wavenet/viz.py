"""
Notebook tools
"""

import celluloid  # type: ignore
import IPython.display as ipd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from IPython.core.display import HTML  # type: ignore
from torch.nn import functional as F

from wavenet import utils


def plot_track(
    track,
    offset: int = 0,
    n_samples: int = 350,
    title: str = "track",
    style: str = "-",
):
    "Expects C, W inputs"

    plt.figure(figsize=(15, 7))
    for channel in range(len(track)):
        data = track[channel, offset : offset + n_samples]
        msg = f"{title}: chan {channel}, offset {offset}, samples {n_samples}"
        plt.title(msg)
        plt.plot(data, style)

    plt.tight_layout()


def plot_random_track(
    batch,
    i: int = None,
    offset: int = 0,
    n_samples: int = 350,
    title: str = "track",
    style: str = "-",
):
    "Expects C, W inputs"

    i = i if i is not None else np.random.randint(len(batch))
    _, track, *_ = batch[i]
    plot_track(track, offset, n_samples, title, style)
    return i


def plot_audio_dataset(ds, cfg, n_examples=10):
    "Try to get an overview of the dataset."

    track_i = plot_random_track(ds, style=".")
    _, track, *_ = ds[track_i]
    ipd.display(ipd.Audio(track, rate=cfg.sampling_rate))

    plt.figure(figsize=(20, 6))
    for i in range(n_examples):
        i, (x, y) = ds.sample()
        plt.plot(x[0, :])


def plot_model_samples(
    m, transforms, sampler, cfg, n_samples=256, batch_size=10
):
    "Plot samples drawn from a trained model."

    def generate(m, transforms, decoder):
        track, *_ = sampler(
            m, transforms, decoder, n_samples=n_samples, batch_size=batch_size
        )

        plt.figure(figsize=(15, 8))
        for i in range(batch_size):
            plt.plot(track.cpu()[i, 0, :])

        plt.show()

    decoders = [
        (utils.decode_argmax, "argmax"),
        (utils.decode_nucleus(core_mass=0.3), "likely nucleus sampling"),
        (utils.decode_nucleus(core_mass=0.7), "relaxed nucleus sampling"),
        (utils.decode_random, "random sampling"),
    ]

    for decoder, name in decoders:
        title = f"Decoding with {name}"
        ipd.display(HTML(f"<h2>{title}</h2>"))
        utils.seed(cfg)
        for _ in range(3):
            generate(m, transforms, decoder)


def plot_stereo_sample_distributions(logits, n: int):
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


class LearningAnimation:
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
        animation.save(filename, writer="imagemagick")
