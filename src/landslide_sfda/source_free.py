"""Losses and exact static masks for target-unlabelled control experiments."""

from __future__ import annotations

import math

import numpy as np
import torch


def binary_entropy(probability: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    probability = probability.clamp(epsilon, 1.0 - epsilon)
    return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log()).mean()


def exact_class_balanced_selection(
    probabilities: np.ndarray, *, fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int | float | bool]]]:
    """Select exactly ceil(fraction*n) most confident pixels per predicted class."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite (N,H,W) array")
    pseudo = probabilities >= 0.5
    confidence = np.abs(probabilities - 0.5)
    selected = np.zeros(probabilities.shape, dtype=bool)
    report = {}
    flat_class = pseudo.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    flat_selected = selected.reshape(-1)
    for value, name in ((False, "background"), (True, "foreground")):
        indices = np.flatnonzero(flat_class == value)
        count = len(indices)
        keep = math.ceil(fraction * count) if count else 0
        if keep:
            local = np.argpartition(flat_confidence[indices], count - keep)[count - keep :]
            flat_selected[indices[local]] = True
        report[name] = {
            "present": bool(count),
            "n_pixels": count,
            "n_selected": keep,
            "selected_fraction": keep / count if count else 0.0,
        }
    return pseudo.astype(np.uint8), selected, report
