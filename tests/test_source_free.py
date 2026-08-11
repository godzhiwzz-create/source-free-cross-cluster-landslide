import numpy as np
import pytest
import torch

from landslide_sfda.source_free import binary_entropy, exact_class_balanced_selection


def test_exact_class_balanced_selection_uses_ceil_per_class():
    probabilities = np.array([[[0.01, 0.10, 0.20, 0.60, 0.80, 0.99]]], dtype=np.float32)
    pseudo, selected, report = exact_class_balanced_selection(probabilities, fraction=0.2)
    assert pseudo.tolist() == [[[0, 0, 0, 1, 1, 1]]]
    assert selected.sum() == 2
    assert report["background"]["n_selected"] == 1
    assert report["foreground"]["n_selected"] == 1
    assert selected.tolist() == [[[True, False, False, False, False, True]]]


def test_selection_rejects_invalid_fraction_and_nonfinite_values():
    with pytest.raises(ValueError):
        exact_class_balanced_selection(np.zeros((1, 2, 2)), fraction=0)
    with pytest.raises(ValueError):
        exact_class_balanced_selection(np.array([[[np.nan]]]), fraction=0.2)


def test_binary_entropy_is_finite_and_lower_for_confident_predictions():
    uncertain = binary_entropy(torch.full((2, 1, 2, 2), 0.5))
    confident = binary_entropy(torch.tensor([[[[0.001, 0.999], [0.001, 0.999]]]]))
    assert torch.isfinite(uncertain)
    assert confident < uncertain
