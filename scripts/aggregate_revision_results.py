#!/usr/bin/env python3
"""Validate and aggregate the paper-facing Scientific Reports revision cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from landslide_sfda.constants import CLUSTERS


BN_CLEAN_SEEDS = (42, 123, 777)
BUDGET_SEEDS = (42, 123)
SOURCE_FREE_SEEDS = (42, 123, 777)
SUPPORT_DRAWS = (0, 1, 2)
BUDGET_SIZES = (25, 50, 100)
BUDGET_STEPS = (10, 20, 50)
BUDGET_THRESHOLDS = ("fixed", "support")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_directory(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for source in sorted(path.rglob("*.json")):
        payload = json.loads(source.read_text())
        records.append((source, payload))
    if not records:
        raise ValueError(f"no JSON results found under {path}")
    return records


def mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def unique(records, key: Callable[[dict[str, Any]], tuple[Any, ...]], label: str) -> None:
    keys = [key(payload) for _, payload in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {label} cells")


def require_keys(payload: dict[str, Any], keys: tuple[str, ...], source: Path) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{source}: missing {missing}")


def aggregate_bn_clean(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    expected = len(BN_CLEAN_SEEDS) * len(CLUSTERS) * len(SUPPORT_DRAWS)
    if len(records) != expected:
        raise ValueError(f"BN-clean campaign has {len(records)} cells, expected {expected}")
    unique(records, lambda p: (p["source_seed"], p["held"], p["support_draw"]), "BN-clean")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for source, payload in records:
        require_keys(payload, ("status", "source_seed", "held", "support_draw", "source_epoch", "support", "query", "source_fixed_0_5", "source_query_label_oracle", "decoder_clean", "full"), source)
        if payload["status"] != "complete" or payload["source_epoch"] != 75:
            raise ValueError(f"{source}: incomplete or non-epoch-75 cell")
        if payload["source_seed"] not in BN_CLEAN_SEEDS or payload["held"] not in CLUSTERS or payload["support_draw"] not in SUPPORT_DRAWS:
            raise ValueError(f"{source}: unexpected seed, cluster, or draw")
        if not payload["decoder_clean"]["frozen_encoder_state_invariant"]["pass"]:
            raise ValueError(f"{source}: frozen encoder state changed")
        if payload["support"]["identity_sha256"] == payload["query"]["identity_sha256"]:
            raise ValueError(f"{source}: support/query identity collision")
        grouped[(payload["source_seed"], payload["held"])].append(payload)

    fields = {
        "source_fixed": lambda p: p["source_fixed_0_5"]["f1"],
        "query_label_oracle": lambda p: p["source_query_label_oracle"]["pixel"]["f1"],
        "decoder_clean_fixed": lambda p: p["decoder_clean"]["fixed_0_5"]["f1"],
        "full_fixed": lambda p: p["full"]["fixed_0_5"]["f1"],
    }
    summary: dict[str, Any] = {}
    for seed in BN_CLEAN_SEEDS:
        seed_key = f"seed{seed}"
        summary[seed_key] = {"clusters": {}}
        for cluster in CLUSTERS:
            cells = sorted(grouped[(seed, cluster)], key=lambda p: p["support_draw"])
            if len(cells) != len(SUPPORT_DRAWS):
                raise ValueError(f"missing BN-clean redraw: seed={seed}, cluster={cluster}")
            summary[seed_key]["clusters"][cluster] = {
                name: mean_sd([float(getter(cell)) for cell in cells])
                for name, getter in fields.items()
            }
        summary[seed_key]["unweighted_cluster_means"] = {
            name: statistics.mean(
                summary[seed_key]["clusters"][cluster][name]["mean"]
                for cluster in CLUSTERS
            )
            for name in fields
        }
    return summary


def aggregate_budget(records: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    expected = (
        len(BUDGET_SEEDS)
        * len(CLUSTERS)
        * len(SUPPORT_DRAWS)
        * len(BUDGET_SIZES)
        * len(BUDGET_STEPS)
        * len(BUDGET_THRESHOLDS)
    )
    if len(records) != expected:
        raise ValueError(f"budget campaign has {len(records)} cells, expected {expected}")
    groups: dict[tuple[int, int, int, str, str], list[float]] = defaultdict(list)
    unique(
        records,
        lambda p: (p["source_seed"], p["held"], p["support_draw"], p["support_size"], p["steps"], p["threshold_mode"]),
        "budget",
    )
    for source, payload in records:
        require_keys(payload, ("source_seed", "held", "support_draw", "support_size", "steps", "threshold_mode", "support_sampling", "adapt_mode", "source_epoch", "pixel"), source)
        if payload["source_seed"] not in BUDGET_SEEDS or payload["held"] not in CLUSTERS:
            raise ValueError(f"{source}: unexpected seed or cluster")
        if payload["support_draw"] not in SUPPORT_DRAWS or payload["support_size"] not in BUDGET_SIZES:
            raise ValueError(f"{source}: unexpected budget or draw")
        if payload["source_epoch"] != 75:
            raise ValueError(f"{source}: non-epoch-75 source checkpoint")
        if payload["steps"] not in BUDGET_STEPS or payload["threshold_mode"] not in BUDGET_THRESHOLDS:
            raise ValueError(f"{source}: unexpected step budget or threshold rule")
        if payload["support_sampling"] not in ("stratified-prevalence", "positive-aware") or payload["adapt_mode"] != "full":
            raise ValueError(f"{source}: wrong sampler or trainable scope")
        key = (payload["source_seed"], payload["support_size"], payload["steps"], payload["threshold_mode"], payload["held"])
        groups[key].append(float(payload["pixel"]["f1"]))

    rows = []
    for seed in BUDGET_SEEDS:
        for support_size in BUDGET_SIZES:
            for steps in BUDGET_STEPS:
                for threshold in BUDGET_THRESHOLDS:
                    cluster_redraws = {}
                    for cluster in CLUSTERS:
                        values = groups[(seed, support_size, steps, threshold, cluster)]
                        if len(values) != len(SUPPORT_DRAWS):
                            raise ValueError(
                                "missing budget redraws: "
                                f"seed={seed}, K={support_size}, steps={steps}, "
                                f"threshold={threshold}, cluster={cluster}"
                            )
                        cluster_redraws[cluster] = mean_sd(values)
                    rows.append(
                        {
                            "source_seed": seed,
                            "support_size": support_size,
                            "steps": steps,
                            "threshold_mode": threshold,
                            "cluster_redraws": cluster_redraws,
                            "unweighted_cluster_mean_f1": statistics.mean(
                                row["mean"] for row in cluster_redraws.values()
                            ),
                        }
                    )
    return rows


def aggregate_source_free(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    expected = len(SOURCE_FREE_SEEDS) * len(CLUSTERS) * 2
    if len(records) != expected:
        raise ValueError(f"source-free campaign has {len(records)} cells, expected {expected}")
    unique(records, lambda p: (p["method"], p["source_seed"], p["held"]), "source-free")
    summary: dict[str, Any] = {}
    for method in ("target-entropy", "class-balanced-pseudo"):
        summary[method] = {}
        for seed in SOURCE_FREE_SEEDS:
            cells = {
                payload["held"]: payload
                for _, payload in records
                if payload["method"] == method and payload["source_seed"] == seed
            }
            if set(cells) != set(CLUSTERS):
                raise ValueError(f"source-free cluster mismatch: method={method}, seed={seed}")
            source_values = [float(cells[cluster]["source_fixed_0_5"]["f1"]) for cluster in CLUSTERS]
            adapted_values = [float(cells[cluster]["adapted_fixed_0_5"]["f1"]) for cluster in CLUSTERS]
            summary[method][f"seed{seed}"] = {
                "clusters": {
                    cluster: {
                        "source_fixed_f1": source,
                        "adapted_fixed_f1": adapted,
                        "change_f1": adapted - source,
                    }
                    for cluster, source, adapted in zip(CLUSTERS, source_values, adapted_values)
                },
                "unweighted_cluster_mean_source_f1": statistics.mean(source_values),
                "unweighted_cluster_mean_adapted_f1": statistics.mean(adapted_values),
                "clusters_improved": sum(adapted > source for source, adapted in zip(source_values, adapted_values)),
            }
    return summary


def manifest(*record_groups: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
        for records in record_groups
        for path, _ in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bn-clean-dir", required=True, type=Path)
    parser.add_argument("--budget-dir", required=True, type=Path)
    parser.add_argument("--source-free-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bn_records = read_directory(args.bn_clean_dir)
    budget_records = read_directory(args.budget_dir)
    source_free_records = read_directory(args.source_free_dir)
    payload = {
        "status": "complete_audit",
        "aggregation": "three redraws within cluster, followed by unweighted arithmetic mean across six clusters; source seeds remain separate; the BN-clean and source-free campaigns use seeds 42/123/777, whereas the frozen budget grid uses seeds 42/123",
        "bn_clean": aggregate_bn_clean(bn_records),
        "budget": aggregate_budget(budget_records),
        "source_free_transductive_controls": aggregate_source_free(source_free_records),
        "source_manifest": manifest(bn_records, budget_records, source_free_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
