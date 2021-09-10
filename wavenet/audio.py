import math
import typing
from pathlib import Path

import librosa  # type: ignore
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import torch

# loading, resampling, framing


def load_raw(
    filename: str,
    mono: bool = False,
    offset_seconds: float = 0.0,
    duration_seconds: typing.Optional[float] = None,
):
    "Load a track off disk into C, W in [-1., 1.]."
    y, sr = librosa.load(
        filename,
        sr=None,
        mono=mono,
        offset=offset_seconds,
        duration=duration_seconds,
    )
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    return y, sr


def write_raw(path: Path, y: np.array, p):
    "Write audio to disk at path, possibly in 8bit PCM"
    subtype = "PCM_U8" if p.n_classes == 2 ** 8 else None
    sf.write(str(path), np.transpose(y), p.sampling_rate, subtype=subtype)


def load_resampled(filename: str, p):
    "Load a resampled track off disk into C, W in [-1., 1.]"
    y, sr = load_raw(filename, mono=p.squash_to_mono)
    return resample(y, sr, p)


def resample(y: np.array, input_sr: int, p):
    "Resample from and to C, W in [-1., 1.]"
    if p.resample and input_sr != p.sampling_rate:
        y = librosa.resample(
            to_librosa(y),
            input_sr,
            p.sampling_rate,
            res_type=p.resampling_method,
        )

        return from_librosa(y)
    return y


def frame(y, p):
    "Cut frames from a single track"
    y = librosa.util.frame(
        y, frame_length=p.sample_length, hop_length=p.sample_hop_length
    )
    y = np.expand_dims(y, axis=0) if y.ndim == 2 else y  # mono case
    return y


def duration(path: Path, p) -> int:
    "Length of track in samples after processing."
    info = sf.info(path)
    n_samples = info.frames
    if p.resample:
        factor = p.sampling_rate / info.samplerate
        n_samples = math.floor(n_samples * factor)
    return n_samples


# compression and quantisation


def mu_compress(x: np.array, p):
    "Mu expand from C, W in [-1., 1.] to C, W in [-1., 1.] "
    return librosa.mu_compress(x, mu=p.n_classes - 1, quantize=False)


def mu_expand(x: np.array, p):
    "Mu expand from C, W in [-1., 1.] to C, W in [-1., 1.]"
    return librosa.mu_expand(x, mu=p.n_classes - 1, quantize=False)


def mu_compress_batch(x: np.array, p):
    "Mu compress from and to N, C, W in [-1., 1.]"

    def fn(x):
        return mu_compress(x, p)

    return np.apply_along_axis(fn, 0, x)


def quantise(x, p):
    "Quantise signal from [-1, 1]"
    buckets = np.linspace(-1, 1, num=p.n_classes, endpoint=True)
    x = np.digitize(x, buckets, right=True)
    return torch.from_numpy(x - p.n_classes // 2)


def dequantise(x, p):
    "Convert x in [-n, n-1] to [-1., 1.] tensor."
    assert x.min() >= -p.n_classes // 2, x.min()
    assert x.max() <= p.n_classes // 2 - 1, x.max()
    return x / (p.n_classes / 2.0)


# librosa


def to_librosa(y):
    "Librosa wants (W,) mono but we want (1, W) for consistency"
    return y.squeeze(0) if y.shape[0] == 1 else y


def from_librosa(y):
    "Librosa wants (W,) mono but we want (1, W) for consistency"
    return np.expand_dims(y, axis=0) if y.ndim == 1 else y
