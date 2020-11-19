import librosa
import torch
import numpy as np


def load_raw(filename: str, mono: bool = False):
    "Load a track off disk into C, W in [-1., 1.]"
    y, sr = librosa.load(filename, sr=None, mono=mono)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    return y, sr


def load_resampled(filename: str, p):
    "Load a resampled track off disk into C, W in [-1., 1.]"
    y, sr = load_raw(filename)
    return resample(y, sr, p)


def resample(y: np.array, input_sr: int, p):
    "Resample from and to C, W in [-1., 1.]"
    if p.resample and input_sr != p.sampling_rate:
        return librosa.resample(y, input_sr, p.sampling_rate)
    return y


def mu_compress(x: np.array, p):
    "Mu expand from C, W in [-1., 1.] to C, W in [-128, 127]"
    return librosa.mu_compress(x, mu=p.n_classes-1, quantize=True)


def mu_expand(x: np.array, p):
    "Mu expand from C, W in [-128, 127] to C, W in [-1., 1.]"
    return librosa.mu_expand(x, mu=p.n_classes-1, quantize=True)


def load_dataset_from_track(filename: str, p):
    "Load many slices from a single track into N, C, W in [-1., 1.]"
    y = load_resampled(filename, p)
    ys = librosa.util.frame(y, frame_length=p.sample_length, hop_length=2**13)
    ys = np.moveaxis(ys, -1, 0)
    ys = torch.tensor(ys, dtype=torch.float32)
    return ys[1:, :, :]  # remove hoplength leading silence


def mu_compress_batch(x: np.array, p):
    "Mu compress from and to N, C, W in [-1., 1.]"
    def fn(x): return mu_compress(x, p)
    return np.apply_along_axis(fn, 0, x)
