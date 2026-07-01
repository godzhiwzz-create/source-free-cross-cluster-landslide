#!/usr/bin/env python3
"""Evaluate fixed, source-prior, support, and label-peeking oracle thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landslide_sfda.constants import CLUSTERS, DEFAULT_THRESHOLDS
from landslide_sfda.data import (
    build_dataset,
    draw_support,
    exclude_entries,
    index_cluster,
)
from landslide_sfda.engine import collect_predictions, load_checkpoint, make_loader
from landslide_sfda.metrics import (
    component_metrics,
    pixel_metrics,
    select_pixel_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--held", choices=CLUSTERS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-size", type=int, default=50)
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def loader(entries, args, mean, std, device):
    dataset = build_dataset(args.data_root, entries, mean, std)
    return make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        device=device,
    )


def prior_threshold(probabilities: np.ndarray, prior: float) -> float:
    return min(
        DEFAULT_THRESHOLDS,
        key=lambda threshold: abs(float((probabilities > threshold).mean()) - prior),
    )


def report(probabilities, targets, threshold):
    return {
        "threshold": threshold,
        "pixel": pixel_metrics(probabilities, targets, threshold).to_dict(),
        "component": component_metrics(
            probabilities, targets, threshold, iou_threshold=0.3
        ).to_dict(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    mean = checkpoint["mean11"]
    std = checkpoint["std11"]
    entries = index_cluster(args.data_root, args.held)
    support = draw_support(
        entries, args.support_size, seed=args.support_seed, strategy="random"
    )
    query = exclude_entries(entries, support)
    support_probabilities, support_targets = collect_predictions(
        model,
        loader(support, args, mean, std, device),
        device=device,
        amp=not args.no_amp,
    )
    query_probabilities, query_targets = collect_predictions(
        model,
        loader(query, args, mean, std, device),
        device=device,
        amp=not args.no_amp,
    )
    support_threshold, _ = select_pixel_threshold(
        support_probabilities, support_targets
    )
    oracle_threshold, _ = select_pixel_threshold(query_probabilities, query_targets)
    source_prior = checkpoint.get("metadata", {}).get("source_prior")
    payload = {
        "held": args.held,
        "checkpoint": str(args.checkpoint),
        "support_size": len(support),
        "support_seed": args.support_seed,
        "support_indices": [entry.index for entry in support],
        "fixed": report(query_probabilities, query_targets, 0.5),
        "support": report(query_probabilities, query_targets, support_threshold),
        "oracle": report(query_probabilities, query_targets, oracle_threshold),
        "oracle_uses_query_labels": True,
    }
    if source_prior is not None:
        threshold = prior_threshold(query_probabilities, float(source_prior))
        payload["source_prior"] = report(
            query_probabilities, query_targets, threshold
        ) | {"source_foreground_prior": source_prior}
    output = args.output or Path("results") / f"{args.held}_threshold_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
