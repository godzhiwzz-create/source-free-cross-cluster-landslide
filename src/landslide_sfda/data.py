"""Memory-mapped Sen12Landslides data access and split utilities."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .constants import LANDSLIDE_PIXEL_MIN, SELECTED_CHANNELS


class Entry(NamedTuple):
    cluster: str
    index: int
    positive_pixels: int
    positive_ratio: float


class MemmapCluster:
    """Read one cluster from the documented memory-mapped layout."""

    def __init__(self, root: str | Path, cluster: str) -> None:
        self.root = Path(root)
        self.cluster = cluster
        meta_path = self.root / f"{cluster}.meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        n = int(self.meta["n_patches"])
        x_shape = (n, *self.meta["X_shape_per_patch"])
        y_shape = (n, *self.meta["Y_shape_per_patch"])
        self.X = np.memmap(
            self.root / f"{cluster}.X.dat", dtype=np.float16, mode="r", shape=x_shape
        )
        self.Y = np.memmap(
            self.root / f"{cluster}.Y.dat", dtype=np.uint8, mode="r", shape=y_shape
        )

    def __len__(self) -> int:
        return int(self.meta["n_patches"])


def index_cluster(
    root: str | Path,
    cluster: str,
    *,
    labeled_only: bool = False,
    min_positive_pixels: int = LANDSLIDE_PIXEL_MIN,
    limit: int | None = None,
) -> list[Entry]:
    """Build patch entries, optionally retaining only labels above the LD cutoff."""
    data = MemmapCluster(root, cluster)
    entries: list[Entry] = []
    pixels_per_patch = int(np.prod(data.Y.shape[-2:]))
    block_size = 256
    for start in range(0, len(data), block_size):
        stop = min(start + block_size, len(data))
        block = np.asarray(data.Y[start:stop])
        counts = (block > 0).reshape(len(block), -1).sum(axis=1)
        for offset, count_value in enumerate(counts):
            count = int(count_value)
            if labeled_only and count <= min_positive_pixels:
                continue
            entries.append(
                Entry(cluster, start + offset, count, count / pixels_per_patch)
            )
            if limit is not None and len(entries) >= limit:
                return entries
    return entries


def filtered_evaluation_pool(
    entries: Sequence[Entry], *, negative_drop_fraction: float = 0.8, seed: int = 42
) -> list[Entry]:
    """Keep all positive patches and a deterministic subset of negative patches."""
    positives = [entry for entry in entries if entry.positive_pixels > 0]
    negatives = [entry for entry in entries if entry.positive_pixels == 0]
    rng = np.random.default_rng(seed)
    keep = round(len(negatives) * (1.0 - negative_drop_fraction))
    indices = rng.choice(len(negatives), size=keep, replace=False) if keep else []
    return positives + [negatives[int(i)] for i in indices]


class ClusterEntryDataset(Dataset):
    """Apply the paper's channel selection, fold normalization, and augmentation."""

    def __init__(
        self,
        root: str | Path,
        cluster: str,
        indices: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        *,
        augment: bool = False,
    ) -> None:
        self.data = MemmapCluster(root, cluster)
        self.indices = list(indices)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        x = np.asarray(self.data.X[index], dtype=np.float32)[
            :, SELECTED_CHANNELS
        ].copy()
        y = np.asarray(self.data.Y[index], dtype=np.float32).copy()
        x = (x - self.mean) / (self.std + 1e-7)
        if self.augment:
            if random.random() < 0.5:
                x = x[:, :, :, ::-1].copy()
                y = y[:, ::-1].copy()
            if random.random() < 0.5:
                x = x[:, :, ::-1, :].copy()
                y = y[::-1, :].copy()
            rotations = random.randrange(4)
            if rotations:
                x = np.rot90(x, rotations, axes=(2, 3)).copy()
                y = np.rot90(y, rotations, axes=(0, 1)).copy()
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y).unsqueeze(0),
            "index": torch.tensor(index),
        }


def build_dataset(
    root: str | Path,
    entries: Sequence[Entry],
    mean: Sequence[float],
    std: Sequence[float],
    *,
    augment: bool = False,
) -> ConcatDataset:
    grouped: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        grouped[entry.cluster].append(entry.index)
    if not grouped:
        raise ValueError("cannot build a dataset from an empty entry list")
    datasets = [
        ClusterEntryDataset(root, cluster, indices, mean, std, augment=augment)
        for cluster, indices in grouped.items()
    ]
    return ConcatDataset(datasets)


def compute_fold_normalization(
    root: str | Path,
    entries: Sequence[Entry],
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 11-channel statistics using training entries only."""
    selected = list(entries)
    if max_samples is not None and len(selected) > max_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(selected), size=max_samples, replace=False)
        selected = [selected[int(index)] for index in chosen]
    cache: dict[str, MemmapCluster] = {}
    sums = np.zeros(len(SELECTED_CHANNELS), dtype=np.float64)
    square_sums = np.zeros_like(sums)
    count = 0
    for entry in selected:
        cache.setdefault(entry.cluster, MemmapCluster(root, entry.cluster))
        x = np.asarray(cache[entry.cluster].X[entry.index], dtype=np.float64)[
            :, SELECTED_CHANNELS
        ]
        sums += x.sum(axis=(0, 2, 3))
        square_sums += np.square(x).sum(axis=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
    if count == 0:
        raise ValueError("cannot compute normalization from an empty training set")
    mean = sums / count
    variance = square_sums / count - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-8)).astype(
        np.float32
    )


def draw_support(
    entries: Sequence[Entry],
    size: int,
    *,
    seed: int,
    strategy: str = "random",
) -> list[Entry]:
    """Draw target support patches under one of the paper's sampling conventions."""
    if size <= 0 or size >= len(entries):
        raise ValueError(
            "support size must be positive and smaller than the target pool"
        )
    rng = np.random.default_rng(seed)
    if strategy == "random":
        chosen = rng.choice(len(entries), size=size, replace=False)
        return [entries[int(index)] for index in chosen]
    if strategy == "positive-aware":
        candidates = [entry for entry in entries if entry.positive_pixels > 0]
        if len(candidates) < size:
            raise ValueError(
                f"positive-aware sampling requested {size} patches but only "
                f"{len(candidates)} contain positives"
            )
        chosen = rng.choice(len(candidates), size=size, replace=False)
        return [candidates[int(index)] for index in chosen]
    raise ValueError(f"unknown support strategy: {strategy}")


def exclude_entries(entries: Iterable[Entry], excluded: Iterable[Entry]) -> list[Entry]:
    keys = {(entry.cluster, entry.index) for entry in excluded}
    return [entry for entry in entries if (entry.cluster, entry.index) not in keys]
