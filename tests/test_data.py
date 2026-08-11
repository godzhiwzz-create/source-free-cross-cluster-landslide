import pytest
import json
import numpy as np

from landslide_sfda.data import (
    ClusterInputDataset,
    Entry,
    draw_support,
    exclude_entries,
    input_indices,
)


def entries():
    return [
        Entry("Africa", index, 10 if index % 2 else 0, 0.01 if index % 2 else 0.0)
        for index in range(20)
    ]


def test_random_support_is_deterministic_and_excluded_from_query():
    first = draw_support(entries(), 5, seed=7, strategy="random")
    second = draw_support(entries(), 5, seed=7, strategy="random")
    assert first == second
    query = exclude_entries(entries(), first)
    assert len(query) == 15
    assert not set(first).intersection(query)


def test_random_support_matches_paper_randomstate_stream():
    support = draw_support(entries(), 5, seed=2000, strategy="random")
    assert [entry.index for entry in support] == [6, 10, 0, 3, 11]


def test_stratified_support_preserves_pool_ratio_and_forces_positive():
    support = draw_support(entries(), 5, seed=3, strategy="stratified-prevalence")
    # The pool is 50% positive, so round(5 * 0.5) gives 2 positive patches.
    assert sum(entry.positive_pixels > 0 for entry in support) == 2
    assert sum(entry.positive_pixels == 0 for entry in support) == 3


def test_support_rejects_impossible_budget():
    with pytest.raises(ValueError):
        draw_support(entries(), 20, seed=3, strategy="stratified-prevalence")


def test_historical_positive_aware_name_is_an_exact_alias():
    historical = draw_support(entries(), 5, seed=3, strategy="positive-aware")
    explicit = draw_support(entries(), 5, seed=3, strategy="stratified-prevalence")
    assert historical == explicit


def test_input_only_dataset_does_not_require_target_label_file(tmp_path):
    metadata = {
        "n_patches": 2,
        "X_shape_per_patch": [4, 14, 4, 4],
        "Y_shape_per_patch": [4, 4],
    }
    (tmp_path / "Africa.meta.json").write_text(json.dumps(metadata))
    x = np.memmap(
        tmp_path / "Africa.X.dat",
        dtype=np.float16,
        mode="w+",
        shape=(2, 4, 14, 4, 4),
    )
    x[:] = 1
    x.flush()
    assert input_indices(tmp_path, "Africa") == [0, 1]
    dataset = ClusterInputDataset(
        tmp_path, "Africa", [1], mean=np.zeros(11), std=np.ones(11)
    )
    sample = dataset[0]
    assert sample["x"].shape == (4, 11, 4, 4)
    assert not (tmp_path / "Africa.Y.dat").exists()
