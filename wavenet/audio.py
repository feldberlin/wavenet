import librosa
import torch
import numpy as np


def load_raw(filename):
    y, sr = librosa.load(filename)
    return y, sr


def load_normalised(filename, p):
    y, sr = load_raw(filename)
    return normalise(y, sr, p)


def load_dataset_from_track(filename, p):
    y = load_normalised(filename, p)
    ys = librosa.util.frame(y, frame_length=p.sample_length, hop_length=64)
    ys = np.expand_dims(np.moveaxis(ys, 0, -1), axis=1)
    return torch.tensor(ys, dtype=torch.float)


def normalise(y, sr, p):
    if sr != p.resampling_rate:
        y = librosa.resample(y, sr=p.resampling_rate, mono=not p.stereo)
    return librosa.mu_compress(y)


def mu_expand(x):
    return librosa.mu_expand(x)


class HParams:

    # resample input audio to this sampling rate
    resampling_rate = 22050

    # retain stereo
    stereo = True

    # length of a single snippet in number of samples
    sample_length = 22050

    def __init__(self, **kwargs):
       for k, v in kwargs.items():
           setattr(self, k, v)
