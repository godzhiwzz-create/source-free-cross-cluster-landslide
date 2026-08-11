import importlib.util
from pathlib import Path

from landslide_sfda.constants import CLUSTERS


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_revision_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_revision_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CAS_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_cas_directional_check.py"
CAS_SPEC = importlib.util.spec_from_file_location("run_cas_directional_check", CAS_SCRIPT)
CAS_MODULE = importlib.util.module_from_spec(CAS_SPEC)
assert CAS_SPEC.loader is not None
CAS_SPEC.loader.exec_module(CAS_MODULE)


def test_revision_aggregator_accepts_complete_frozen_campaigns():
    bn_records = []
    for seed in MODULE.BN_CLEAN_SEEDS:
        for cluster_index, cluster in enumerate(CLUSTERS):
            for draw in (0, 1, 2):
                value = seed / 1000 + cluster_index / 100 + draw / 1000
                bn_records.append(
                    (
                        Path(f"bn_{seed}_{cluster}_{draw}.json"),
                        {
                            "status": "complete",
                            "source_seed": seed,
                            "held": cluster,
                            "support_draw": draw,
                            "source_epoch": 75,
                            "support": {"identity_sha256": f"support-{seed}-{cluster}-{draw}"},
                            "query": {"identity_sha256": f"query-{seed}-{cluster}-{draw}"},
                            "source_fixed_0_5": {"f1": value},
                            "source_query_label_oracle": {"pixel": {"f1": value + 0.01}},
                            "decoder_clean": {
                                "fixed_0_5": {"f1": value + 0.02},
                                "frozen_encoder_state_invariant": {"pass": True},
                            },
                            "full": {"fixed_0_5": {"f1": value + 0.03}},
                        },
                    )
                )
    bn = MODULE.aggregate_bn_clean(bn_records)
    assert set(bn) == {"seed42", "seed123", "seed777"}
    assert set(bn["seed42"]["clusters"]) == set(CLUSTERS)

    budget_records = []
    for seed in MODULE.BUDGET_SEEDS:
        for cluster_index, cluster in enumerate(CLUSTERS):
            for draw in (0, 1, 2):
                for support_size in (25, 50, 100):
                    for steps in (10, 20, 50):
                        for threshold in ("fixed", "support"):
                            budget_records.append(
                                (
                                    Path(
                                        f"budget_{seed}_{cluster}_{draw}_{support_size}_"
                                        f"{steps}_{threshold}.json"
                                    ),
                                    {
                                        "source_seed": seed,
                                        "held": cluster,
                                        "support_draw": draw,
                                        "support_size": support_size,
                                        "steps": steps,
                                        "threshold_mode": threshold,
                                        "support_sampling": "stratified-prevalence",
                                        "adapt_mode": "full",
                                        "source_epoch": 75,
                                        "pixel": {"f1": cluster_index / 10 + draw / 100},
                                    },
                                )
                            )
    budget = MODULE.aggregate_budget(budget_records)
    assert len(budget) == 36
    assert all(len(row["cluster_redraws"]) == 6 for row in budget)

    control_records = []
    for method in ("target-entropy", "class-balanced-pseudo"):
        for seed in MODULE.SOURCE_FREE_SEEDS:
            for cluster_index, cluster in enumerate(CLUSTERS):
                source = 0.1 + cluster_index / 100
                control_records.append(
                    (
                        Path(f"control_{method}_{seed}_{cluster}.json"),
                        {
                            "method": method,
                            "source_seed": seed,
                            "held": cluster,
                            "source_fixed_0_5": {"f1": source},
                            "adapted_fixed_0_5": {"f1": source + 0.01},
                        },
                    )
                )
    controls = MODULE.aggregate_source_free(control_records)
    assert controls["target-entropy"]["seed42"]["clusters_improved"] == 6
    assert controls["class-balanced-pseudo"]["seed123"]["clusters_improved"] == 6
    assert controls["class-balanced-pseudo"]["seed777"]["clusters_improved"] == 6


def test_cas_event_contract_defaults_and_custom_names():
    assert set(CAS_MODULE.parse_events(None)) == {
        "Palu",
        "Lombok",
        "Hokkaido",
        "Tiburon_S",
        "Tiburon_P",
    }
    assert CAS_MODULE.parse_events(["A=event_a", "B=event_b"]) == {
        "A": "event_a",
        "B": "event_b",
    }


def test_cas_ranking_uses_every_reported_candidate_and_both_full_variants():
    ranking = CAS_MODULE.rank_candidate_scores(
        source_f1=0.3539,
        oracle_f1=0.3539,
        threshold_f1=0.3525,
        decoder_f1=0.3179,
        full_fixed_f1=0.4714,
        full_recipe_f1=0.4710,
    )
    assert list(ranking["scores"]) == [
        "source_0_5",
        "query_label_oracle",
        "threshold_only",
        "decoder_fixed",
        "full_fixed",
        "full_recipe",
    ]
    assert ranking["best_strategy"] == "full_fixed"
    assert ranking["best_full_strategy"] == "full_fixed"
    assert ranking["best_full_regret"] == 0.0
