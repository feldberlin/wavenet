import librosa
import torch
import numpy as np


def load_raw(filename: str):
    "Load a track off disk into C, W in [-1., 1.]"
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def load_resampled(filename: str, p):
    "Load a resampled track off disk into C, W in [-1., 1.]"
    y, sr = load_raw(filename)
    return resample(y, sr, p)


def load_dataset_from_track(filename: str, p):
    "Load many slices from a single track into N, C, W in [-1., 1.]"
    y = load_resampled(filename, p)
    ys = librosa.util.frame(y, frame_length=p.sample_length, hop_length=64)
    ys = np.expand_dims(np.moveaxis(ys, 0, -1), axis=1)
    return torch.tensor(ys, dtype=torch.float)


def resample(y: np.array, input_sr: int, p):
    "Resample from and to C, W in [-1., 1.]"
    if p.resample and input_sr != p.sampling_rate:
        return librosa.resample(y, input_sr, p.sampling_rate)
    return y


def znorm(y: np.array, mean=None, variance=None, eps=1e-6):
    "Scale N, C, W to unit variance and 0 mean along the batch dimension N."
    mean = y.mean(0) if mean is None else mean
    variance = y.std(0) if variance is None else variance
    return (y - mean) / (variance + eps), mean, variance


def mu_compress(x: np.array, p):
    "Mu expand from and to C, W in [-1., 1.]"
    return librosa.mu_compress(x, mu=p.n_classes-1, quantize=False)


def mu_expand(x: np.array, p):
    "Mu expand from and to C, W in [-1., 1.]"
    return librosa.mu_expand(x, mu=p.n_classes-1, quantize=False)


def mu_compress_batch(x: np.array, p):
    "Mu compress from and to N, C, W in [-1., 1.]"
    def fn(x):
        return mu_compress(x, p)
    return np.apply_along_axis(fn, 0, x)
