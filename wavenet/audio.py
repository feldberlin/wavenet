import librosa
import torch
import numpy as np


def load_raw(filename: str):
    y, sr = librosa.load(filename)
    return y, sr


def load_normalised(filename: str, p):
    y, sr = load_raw(filename)
    return normalise(y, sr, p)


def load_dataset_from_track(filename: str, p):
    y = load_normalised(filename, p)
    ys = librosa.util.frame(y, frame_length=p.sample_length, hop_length=64)
    ys = np.expand_dims(np.moveaxis(ys, 0, -1), axis=1)
    return torch.tensor(ys, dtype=torch.float)


def normalise(y: np.array, sr: int, p):
    if sr != p.sampling_rate and p.resample:
        y = librosa.resample(y, sr=p.sampling_rate, mono=not p.stereo)
    return librosa.mu_compress(y, mu=p.n_classes, quantize=True)


def mu_expand(x: np.array, p):
    return librosa.mu_expand(x, mu=p.n_classes, quantize=True)


class HParams:

    # resample to sampling_rate before mu law compansion
    resample = True

    # resample input audio to this sampling rate
    sampling_rate = 22050

    # retain stereo
    stereo = True

    # length of a single snippet in number of samples
    sample_length = 22050

    # what mu law bit depth to use
    n_classes = 2**8

    def __init__(self, **kwargs):
       for k, v in kwargs.items():
           setattr(self, k, v)
