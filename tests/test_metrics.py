import numpy as np

from landslide_sfda.metrics import (
    component_metrics,
    pixel_metrics,
    select_pixel_threshold,
)


def test_pixel_metrics_known_confusion():
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    targets = np.array([1, 0, 1, 0])
    metrics = pixel_metrics(probabilities, targets, 0.5)
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (1, 1, 1, 1)
    assert metrics.f1 == 0.5
    assert metrics.iou == 1 / 3
    assert metrics.accuracy == 0.5


def test_threshold_selection_uses_support_labels():
    probabilities = np.array([0.45, 0.40, 0.10, 0.05])
    targets = np.array([1, 1, 0, 0])
    threshold, metrics = select_pixel_threshold(
        probabilities, targets, thresholds=(0.2, 0.5)
    )
    assert threshold == 0.2
    assert metrics.f1 == 1.0


def test_component_matching_is_one_to_one_and_patchwise():
    probabilities = np.zeros((2, 8, 8), dtype=np.float32)
    targets = np.zeros_like(probabilities, dtype=np.uint8)
    targets[0, 1:3, 1:3] = 1
    probabilities[0, 1:3, 1:3] = 0.9
    probabilities[0, 5:7, 5:7] = 0.9
    targets[1, 2:5, 2:5] = 1
    metrics = component_metrics(probabilities, targets, 0.5)
    assert (metrics.tp, metrics.fp, metrics.fn) == (1, 1, 1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.quality == 1 / 3
    assert metrics.fp_per_patch == 0.5
