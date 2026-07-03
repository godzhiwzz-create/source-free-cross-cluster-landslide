import pytest

from landslide_sfda.data import Entry, draw_support, exclude_entries


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


def test_positive_aware_support_preserves_pool_ratio_and_forces_positive():
    support = draw_support(entries(), 5, seed=3, strategy="positive-aware")
    # The pool is 50% positive, so round(5 * 0.5) gives 2 positive patches.
    assert sum(entry.positive_pixels > 0 for entry in support) == 2
    assert sum(entry.positive_pixels == 0 for entry in support) == 3


def test_support_rejects_impossible_budget():
    with pytest.raises(ValueError):
        draw_support(entries(), 20, seed=3, strategy="positive-aware")
