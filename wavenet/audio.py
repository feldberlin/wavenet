import librosa
import torch
import numpy as np


def load_raw(filename: str):
    """Load a track off disk.

    Arguments:

    returns: floating point C, W data in the range [-1.0, 1.0]"
    """

    y, sr = librosa.load(filename)
    return y, sr


def load_resampled_quantised(filename: str, p):
    """Load a track off disk and `resample_quantise` as per hyperparams p.

    Arguments:

    returns: integral C, W data in the range [-128, 127]"
    """

    y, sr = load_raw(filename)
    return resample_quantise(y, sr, p)


def load_dataset_from_track(filename: str, p):
    """Load a dataset of many slices of data from a single track on disk.

    Arguments:

    returns: floating point N, C, W data in the range [-128.0, 127.0]"
    """

    y = load_resampled_quantised(filename, p)
    ys = librosa.util.frame(y, frame_length=p.sample_length, hop_length=64)
    ys = np.expand_dims(np.moveaxis(ys, 0, -1), axis=1)
    return torch.tensor(ys, dtype=torch.float)


def mu_expand(x: np.array, p):
    """Undo the mu law compansion done in the resample_quantise function.

    Arguments:

    y: C, W floating point input data in the range [-1.0, 1.0]
    returns: integral C, W data in the range [-128, 127]"
    """

    return librosa.mu_expand(x, mu=p.n_classes-1, quantize=True)


def resample_quantise(y: np.array, input_sr: int, p):
    """Resample and quantise audio input.

    Arguments:

    y: C, W floating point input data in the range [-1.0, 1.0]
    returns: integral C, W data in the range [-128, 127]"
    """

    if p.resample and input_sr != p.sampling_rate:
        y = librosa.resample(y, sr=p.sampling_rate, mono=not p.stereo)
    return librosa.mu_compress(y, mu=p.n_classes-1, quantize=True)
