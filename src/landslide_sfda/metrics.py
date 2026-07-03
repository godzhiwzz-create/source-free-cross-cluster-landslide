"""Pixel and connected-component evaluation used by the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from .constants import DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class PixelMetrics:
    f1: float
    precision: float
    recall: float
    iou: float
    accuracy: float
    mean_iou: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentMetrics:
    f1: float
    precision: float
    recall: float
    quality: float
    tp: int
    fp: int
    fn: int
    fp_per_patch: float
    patches: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class PaperComponentMetrics:
    """Historical component read-out used to generate the manuscript tables."""

    f1: float
    precision: float
    recall: float
    quality: float
    matched_predictions: int
    matched_targets: int
    predicted_components: int
    target_components: int
    fp: int
    fn: int
    fp_per_patch: float
    patches: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def pixel_metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> PixelMetrics:
    pred = np.asarray(probabilities) > threshold
    truth = np.asarray(targets) > 0
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return PixelMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        iou=iou,
        accuracy=accuracy,
        mean_iou=(iou + background_iou) / 2,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def select_pixel_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> tuple[float, PixelMetrics]:
    scored = [
        (float(threshold), pixel_metrics(probabilities, targets, float(threshold)))
        for threshold in thresholds
    ]
    return max(scored, key=lambda item: (item[1].f1, -abs(item[0] - 0.5)))


def _component_counts(
    prediction: np.ndarray, target: np.ndarray, iou_threshold: float
) -> tuple[int, int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, pred_count = ndimage.label(prediction, structure=structure)
    target_labels, target_count = ndimage.label(target, structure=structure)
    candidates: list[tuple[float, int, int]] = []
    for pred_id in range(1, pred_count + 1):
        pred_mask = pred_labels == pred_id
        overlapping = np.unique(target_labels[pred_mask])
        for target_id in overlapping:
            if target_id == 0:
                continue
            target_mask = target_labels == target_id
            intersection = int(np.logical_and(pred_mask, target_mask).sum())
            union = int(np.logical_or(pred_mask, target_mask).sum())
            iou = intersection / union
            if iou > iou_threshold:
                candidates.append((iou, pred_id, int(target_id)))
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    for _, pred_id, target_id in sorted(candidates, reverse=True):
        if pred_id in matched_predictions or target_id in matched_targets:
            continue
        matched_predictions.add(pred_id)
        matched_targets.add(target_id)
    tp = len(matched_predictions)
    return tp, pred_count - tp, target_count - tp


def _overlap_component_counts(
    prediction: np.ndarray, target: np.ndarray, iou_threshold: float
) -> tuple[int, int, int, int]:
    """Match every component independently, mirroring the AutoDL table script."""
    # No structure argument is intentional. SciPy's 2-D default is
    # 4-connectivity, which is the implementation that generated the tables.
    pred_labels, pred_count = ndimage.label(prediction)
    target_labels, target_count = ndimage.label(target)
    prediction_masks = [
        pred_labels == component for component in range(1, pred_count + 1)
    ]
    target_masks = [
        target_labels == component for component in range(1, target_count + 1)
    ]
    matched_targets = 0
    matched_predictions = 0
    if prediction_masks and target_masks:
        for target_mask in target_masks:
            best = max(
                np.logical_and(target_mask, prediction_mask).sum()
                / np.logical_or(target_mask, prediction_mask).sum()
                for prediction_mask in prediction_masks
            )
            matched_targets += int(best > iou_threshold)
        for prediction_mask in prediction_masks:
            best = max(
                np.logical_and(target_mask, prediction_mask).sum()
                / np.logical_or(target_mask, prediction_mask).sum()
                for target_mask in target_masks
            )
            matched_predictions += int(best > iou_threshold)
    return target_count, pred_count, matched_targets, matched_predictions


def paper_component_metrics(
    probabilities: np.ndarray | Sequence[np.ndarray],
    targets: np.ndarray | Sequence[np.ndarray],
    threshold: float,
    *,
    iou_threshold: float = 0.3,
) -> PaperComponentMetrics:
    """Reproduce the component convention used by the paper result scripts.

    This convention uses 4-connectivity and independent target-to-prediction
    and prediction-to-target overlap tests. It is not one-to-one matching.
    """
    probs = np.asarray(probabilities)
    truth = np.asarray(targets)
    if probs.ndim == 2:
        probs = probs[None]
        truth = truth[None]
    if probs.ndim != 3 or truth.shape != probs.shape:
        raise ValueError("component evaluation expects matching (N, H, W) arrays")
    target_count = predicted_count = matched_targets = matched_predictions = 0
    for probability, target in zip(probs, truth, strict=True):
        counts = _overlap_component_counts(
            probability > threshold, target > 0, iou_threshold
        )
        target_count += counts[0]
        predicted_count += counts[1]
        matched_targets += counts[2]
        matched_predictions += counts[3]
    recall = matched_targets / max(target_count, 1)
    precision = matched_predictions / max(predicted_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    # The table's "quality" is the Jaccard transform of component F1 because
    # independent overlap matching does not have one shared TP count.
    quality = f1 / max(2.0 - f1, 1e-12)
    false_positives = predicted_count - matched_predictions
    false_negatives = target_count - matched_targets
    return PaperComponentMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        quality=quality,
        matched_predictions=matched_predictions,
        matched_targets=matched_targets,
        predicted_components=predicted_count,
        target_components=target_count,
        fp=false_positives,
        fn=false_negatives,
        fp_per_patch=false_positives / len(probs),
        patches=len(probs),
    )


def component_metrics(
    probabilities: np.ndarray | Sequence[np.ndarray],
    targets: np.ndarray | Sequence[np.ndarray],
    threshold: float,
    *,
    iou_threshold: float = 0.3,
) -> ComponentMetrics:
    """Aggregate one-to-one 8-connected matching over individual patches."""
    probs = np.asarray(probabilities)
    truth = np.asarray(targets)
    if probs.ndim == 2:
        probs = probs[None]
        truth = truth[None]
    if probs.ndim != 3 or truth.shape != probs.shape:
        raise ValueError("component evaluation expects matching (N, H, W) arrays")
    tp = fp = fn = 0
    for probability, target in zip(probs, truth, strict=True):
        counts = _component_counts(probability > threshold, target > 0, iou_threshold)
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    quality = tp / max(tp + fp + fn, 1)
    return ComponentMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        quality=quality,
        tp=tp,
        fp=fp,
        fn=fn,
        fp_per_patch=fp / len(probs),
        patches=len(probs),
    )


def select_component_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    *,
    iou_threshold: float = 0.3,
) -> tuple[float, ComponentMetrics]:
    scored = [
        (
            float(threshold),
            component_metrics(
                probabilities,
                targets,
                float(threshold),
                iou_threshold=iou_threshold,
            ),
        )
        for threshold in thresholds
    ]
    return max(scored, key=lambda item: (item[1].f1, -abs(item[0] - 0.5)))
