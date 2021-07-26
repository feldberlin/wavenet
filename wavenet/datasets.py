# Datasets and transforms.
#
# Each element is a training frame in two versions x, y. Both represent the
# same data series, where x is normalised and in the range [-1, 1], while y is
# the integral class index of the value at that point.
#
# The same transforms are applied to all audio datasets, and these assume that
# the dataset emits quantised integral x, y tensors.

import abc
import collections
import datetime
import json
import math
import typing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np  # type: ignore
import torch

from wavenet import audio, utils

# data transformations


class Transforms:
    @abc.abstractmethod
    def __call__(self, data):
        "Convert from data to x, y for training"
        raise NotImplementedError

    @abc.abstractmethod
    def normalise(self, y):
        "Convert from y back to x for autoregressive generation"
        raise NotImplementedError


class AudioUnitTransforms(Transforms):
    """Normalise x by dequantising data, convert y to class indices by shift.
    Transforms assume quantised data input.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = 0.0

    def __call__(self, data):
        x = audio.dequantise(data, self.cfg)
        y = utils.audio_to_class_idxs(data, self.cfg.n_classes)
        return x, y

    def normalise(self, y):
        x = utils.audio_from_class_idxs(y, self.cfg.n_classes)
        x = audio.dequantise(x, self.cfg)
        return x


class NormaliseTransforms(Transforms):
    """Normalise x, convert y to class indices.
    Transforms assume quantised x, y inputs.
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self.eps = 1e-15

    def __call__(self, data):
        y = data
        x = (y - self.mean) / (self.std + self.eps)
        return x, y

    def normalise(self, y):
        return (y - self.mean) / (self.std + self.eps)


# datasets


class Dataset(torch.utils.data.Dataset, collections.abc.Sequence):
    "Map style datasets, but also iterable."

    @property
    @abc.abstractmethod
    def transforms(self) -> Transforms:
        raise NotImplementedError

    def sample(self):
        i = torch.randint(len(self), (1,))
        return i, self[i]


def to_tensor(d: Dataset, n_items=None):
    "Materialize the whole dataset"
    n_items = n_items if n_items else len(d)
    return (
        torch.stack([d[i][0] for i in range(n_items)]),
        torch.stack([d[i][1] for i in range(n_items)]),
    )


@dataclass(frozen=True)
class TrackMeta:
    "Info about the location and duration of an audio source file"

    source_dir: Path  # root dataset dir for source file
    cache_dir: typing.Optional[Path]  # root dataset dir for cached file
    file_path: Path  # relative path to file
    duration: int  # duration in seconds. final incomplete sample dropped

    @property
    def path(self):
        return self.source_dir.joinpath(self.file_path)

    @property
    def cache_path(self):
        if self.cache_dir:
            return self.cache_dir.joinpath(self.file_path)

    def asdict(self):
        "pytorch only allows certain primitives when collating batches"
        return {"path": str(self.path), "duration": self.duration}


def tracks(filename: str, validation_pct: float, p):
    "Train - validation split on a single track"
    return (
        Track(filename, p, 0.0, 1 - validation_pct),
        Track(filename, p, 1 - validation_pct, 1.0),
    )


def trackmetas(
    root: Path,
    cache: typing.Optional[Path],
    p,
    tracks: typing.List[Path] = [],
) -> typing.List[TrackMeta]:
    "Find tracks in a path and return path and duration metadata."

    def meta(path):
        cache_key = cache / p.audio_cache_key() if cache else None
        duration = audio.duration(root / path, p)
        return TrackMeta(root, cache_key, path, duration)

    return [meta(path) for path in tracks or glob_tracks(root)]


def glob_tracks(path: Path) -> typing.List[Path]:
    "Return a glob of audio files in the given subtree."
    return [
        p.relative_to(path)
        for p in path.rglob("*")
        if p.suffix in [".wav", ".mp3"]
    ]


# core datasets


class Tracks(Dataset):
    """Create a single dataset out of a bunch of tracks on disk.

    Set a `cache_dir` to store the resampled and compressed files in
    a different location, because:

    a. Resampling is quite slow. This is mitigated by a faster
       `resampling_method` like sox_hq, but it's still slow.
    b. mu compression is also quite slow.
    c. Source files may be stored on network attached storage. In this case,
       the cache location can be on a local ssd.
    """

    def __init__(self, cfg, root_dir: Path, tracks: typing.List[TrackMeta]):
        self.tf = AudioUnitTransforms(cfg)
        self.cfg = cfg
        self.root_dir = root_dir
        self.tracks = tracks
        self.n_samples_total = sum(t.duration for t in tracks)
        self.n_examples = 0

        # position i in the offsets array corresponds to track i in this
        # dataset. self.offsets[i] is the training example number (idx) in the
        # logically concatenated dataset.
        self.offsets = [0]
        for t in tracks:
            duration = t.duration - cfg.sample_overlap_length
            self.n_examples += math.floor(duration / cfg.sample_hop_length())
            self.offsets.append(self.n_examples)

    @staticmethod
    def from_dir(cfg, root_dir: Path, cache_dir: typing.Optional[Path] = None):
        "Build a dataset with all audio files under root_dir"
        metas = trackmetas(root_dir, cache_dir, cfg)
        return Tracks(cfg, root_dir, metas)

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        "Get the given example index."

        @lru_cache()
        def meta(example_idx):
            "get the correct (TrackMeta, track_offset) for this example idx"
            assert example_idx >= 0 and example_idx < len(self)
            for i, _ in enumerate(self.offsets):
                if example_idx < self.offsets[i]:
                    track = self.tracks[i - 1]
                    track_idx = example_idx - self.offsets[i - 1]
                    track_offset = track_idx * self.cfg.sample_hop_length()
                    assert track_offset <= track.duration, track_offset
                    return track, track_offset

        track, track_offset = meta(idx)
        ys = self.cached_read(track, track_offset)
        ys = audio.quantise(ys, self.cfg)  # from [-1, 1] to [-n, n+1]
        x, y = self.tf(ys)
        return x, y, track.asdict()

    def cached_read(self, meta: TrackMeta, offset: int) -> np.array:
        "Returns compressed and resampled tracks. Caches on disk"
        if meta.cache_path and meta.cache_path.exists():
            offset_seconds = offset / self.cfg.sampling_rate
            duration_seconds = self.cfg.sample_size_ms() / 1000
            y, sr = audio.load_raw(
                meta.cache_path,
                mono=self.cfg.squash_to_mono,
                offset_seconds=offset_seconds,
                duration_seconds=duration_seconds,
            )

            # make sure that the disk cache is at the expected sampling rate
            assert sr == self.cfg.sampling_rate
            return y
        else:
            y = audio.load_resampled(meta.path, self.cfg)
            if self.cfg.compress:
                y = audio.mu_compress(y, self.cfg)
            y = torch.tensor(y, dtype=torch.float32)

            # write the whole file
            if meta.cache_path:
                meta.cache_path.parent.mkdir(parents=True, exist_ok=True)
                audio.write_raw(meta.cache_path, y, self.cfg)

            # return just the example sliced from the file
            y = y[:, offset : offset + self.cfg.sample_length]
            return y

    def duration(self) -> datetime.timedelta:
        "The duration of this dataset"
        n_seconds = int(len(self) / self.cfg.sample_length)
        return datetime.timedelta(seconds=n_seconds)

    def __repr__(self):
        attrs = {
            "path": str(self.root_dir),
            "n_examples": self.n_examples,
            "n_seconds": self.cfg.sample_size_ms() * self.n_examples / 1000,
        }

        return f"Tracks({ json.dumps(attrs, sort_keys=True) })"


# Maestro 2.0.0


def maestro(
    root_dir: Path,
    cfg,
    cache_dir: typing.Optional[Path] = None,
    year: typing.Optional[int] = None,
) -> typing.Tuple[Tracks, Tracks]:

    tracks = []
    json_path = root_dir / "maestro-v2.0.0.json"
    with open(json_path) as f:
        tracks = json.load(f)

    def metas(split):
        return trackmetas(
            root_dir,
            cache_dir,
            cfg,
            [
                Path(t["audio_filename"])
                for t in tracks
                if (year is None or t["year"] == year) and t["split"] == split
            ],
        )

    return (
        Tracks(cfg, root_dir, metas("train")),
        Tracks(cfg, root_dir, metas("test")),
    )


# toy datasets


class StereoImpulse(Dataset):
    """Left and right at t0 are both binomial, modes slightly apart.
    Batch of m impulses at t0, followed by n zero samples, held in memory.
    """

    def __init__(self, n, m, cfg, probs=None):
        self.tf = AudioUnitTransforms(cfg)
        probs = probs if probs else (0.45, 0.55)
        data = np.random.binomial(
            (cfg.n_classes, cfg.n_classes), probs, (n, cfg.n_audio_chans)
        )

        data = utils.audio_from_class_idxs(data, cfg.n_classes)
        data_batched = np.reshape(data, (n, cfg.n_audio_chans, 1))
        data_batched = np.pad(data_batched, ((0, 0), (0, 0), (0, m - 1)))
        self.data = torch.from_numpy(data_batched)

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.tf(self.data[idx])

    def __repr__(self):
        return "StereoImpulse()"


class Sines(Dataset):
    """Each sample is a μ compressed sine wave with random amp and phase.
    Random amplitude and hz, unless given.
    """

    def __init__(
        self,
        n_examples,
        cfg,
        amp: float = None,
        hz: float = None,
        phase: float = None,
        minhz=400,
        maxhz=20000,
    ):

        # config
        self.n_seconds = cfg.sample_size_ms() / 1000
        self.n_examples = n_examples
        self.cfg = cfg

        # normalisations
        self.tf = AudioUnitTransforms(cfg)

        # amp
        default_amps = torch.rand(n_examples)
        self.amp = amp if amp is not None else default_amps

        # hz
        default_hzs = torch.rand(n_examples) * maxhz + minhz
        self.hz = hz if hz is not None else default_hzs

        # phase
        default_phases = torch.rand(n_examples) * np.pi * 2 / self.hz
        self.phase = phase if phase is not None else default_phases

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):

        # retrieve parameters for this example
        amp = self.amp if np.isscalar(self.amp) else self.amp[idx]
        hz = self.hz if np.isscalar(self.hz) else self.hz[idx]
        phase = self.phase if np.isscalar(self.phase) else self.phase[idx]

        # calculate signal
        x = torch.arange(0, self.n_seconds, 1 / self.cfg.sampling_rate)
        y = torch.sin((x - phase) * np.pi * 2 * hz) * amp
        y = y.unsqueeze(0)  # C, W

        # copy to n audio channels
        if self.cfg.n_audio_chans > 1:
            y = y.repeat(self.cfg.n_audio_chans, 1)

        # mu compress
        if self.cfg.compress:
            y = audio.mu_compress(y.numpy(), self.cfg)
            y = torch.from_numpy(y)

        return self.tf(audio.quantise(y, self.cfg))

    def __repr__(self):
        x = [("nseconds", self.n_seconds)]
        if np.isscalar(self.amp):
            x.append(("amp", self.amp))
        if np.isscalar(self.hz):
            x.append(("hz", self.hz))
        if np.isscalar(self.phase):
            x.append(("phase", self.phase))
        x = ", ".join([f"{k}: {v}" for k, v in x])
        return f"Sines({ x })"


class Tiny(Dataset):
    """A toy dataset of non-linear series, as described in
    https://fleuret.org/dlc/materials/dlc-slides-10-1-autoregression.pdf.
    This dataset is tiny, non-linear, and includes local effects (slopes), as
    well as global effects (a single discontinuity).
    """

    def __init__(self, n, m):
        """Produces a dataset of m timeseries, where each one is n long, and
        values are drawn from 0 to 2n.
        """

        # Create m timeseries of length n, with a random split in each one
        splits = torch.randint(0, n, (1, m))

        # Let's create 2m randomly tilted timeseries. First of, we want 2m
        # slopes between -1 and 1.  Then we'll broadcast along the x dimension
        # to obtain a n, 2m tensor, where each column is a tilted timeseries.
        x = torch.arange(n)
        slopes = torch.rand(2 * m) * 2 - 1
        series = slopes.view(1, -1) * x.view(-1, 1)  # (1,2m) * (n,1) => (n,2m)

        # Now we'll add another dimension and fill it with pairs of
        # complementary series, to (n, m, 2).
        series = series.reshape(n, -1, 2)

        # shift the right series such that it starts at zero.
        offsets = series[:, :, 1].gather(0, splits).squeeze()
        series[:, :, 1] -= offsets

        # Now, with the m split points between 0 and n, we create a mask where
        # values are below those cutoff points. We will apply the mask to half
        # the series, and the inverse of the mask to the complementary series.
        mask = torch.arange(n).unsqueeze(-1) > splits
        mask = torch.stack((mask, ~mask), dim=-1)
        series[mask] = 0

        # Now we sum together along the final dimension to obtain the result.
        series = torch.sum(series, dim=-1).long() + n

        # conventionally introduce a single channel dimension
        self.data = series.unsqueeze(0)

        # dataset stats
        mean = torch.mean(self.data.float())
        std = torch.std(self.data.float())
        self.tf = NormaliseTransforms(mean, std)

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, idx):
        return self.tf(self.data[:, :, idx])


class Track(Dataset):
    """Dataset constructed from a single track, held in memory.
    Loads μ compressed slices from a single track into N, C, W in [-1., 1.].
    """

    def __init__(
        self,
        filename: str,
        p,
        start: float = 0.0,
        end: float = 1.0,
        tf: Transforms = None,
    ):
        # transforms can have corpus-wide stats
        if tf:
            self.tf = tf
        else:
            self.tf = AudioUnitTransforms(p)

        self.filename = filename
        y = audio.load_resampled(filename, p)  # audio data in [-1, 1]
        _, n_samples = y.shape
        y = y[:, int(n_samples * start) : int(n_samples * end)]  # start to end
        y = audio.to_librosa(y)  # different mono channels for librosa
        ys = audio.frame(y, p)  # cut frames from single track
        ys = np.moveaxis(ys, -1, 0)  # reshape back to N, C, W
        ys = torch.tensor(ys, dtype=torch.float32)
        if p.compress:
            ys = audio.mu_compress_batch(ys, p)
        self.data = audio.quantise(ys, p)  # from [-1, 1] to [-n, n+1]

    @property
    def transforms(self):
        return self.tf

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.tf(self.data[idx])

    def __repr__(self):
        return f"Track({self.filename})"
