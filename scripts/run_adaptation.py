#!/usr/bin/env python3
"""Run a paper-aligned source-free K-shot adaptation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landslide_sfda.constants import CLUSTERS
from landslide_sfda.data import (
    Entry,
    build_dataset,
    draw_support,
    exclude_entries,
    index_cluster,
)
from landslide_sfda.engine import (
    collect_predictions,
    configure_adaptation,
    load_checkpoint,
    make_loader,
    set_seed,
    train_steps,
)
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
    parser.add_argument(
        "--support-sampling",
        choices=("random", "positive-aware"),
        default="random",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("cross-fit", "support", "fixed"),
        default="cross-fit",
    )
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument(
        "--adapt-mode", choices=("full", "decoder", "head", "bn"), default="full"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_entry_loader(
    entries: list[Entry],
    args: argparse.Namespace,
    mean,
    std,
    device: torch.device,
    *,
    training: bool,
):
    dataset = build_dataset(args.data_root, entries, mean, std, augment=training)
    return make_loader(
        dataset,
        batch_size=args.batch_size if training else args.eval_batch_size,
        shuffle=training,
        workers=args.workers,
        device=device,
    )


def new_adapted_model(args, device, support, mean, std, seed):
    set_seed(seed)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    trainable = configure_adaptation(model, args.adapt_mode)
    losses = train_steps(
        model,
        make_entry_loader(support, args, mean, std, device, training=True),
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        amp=not args.no_amp,
    )
    return model, trainable, losses


def cross_fit_threshold(args, device, support, mean, std) -> float:
    if args.cross_fit_folds < 2 or args.cross_fit_folds > len(support):
        raise ValueError("cross-fit-folds must be between 2 and support size")
    rng = np.random.default_rng(args.support_seed)
    order = rng.permutation(len(support))
    folds = np.array_split(order, args.cross_fit_folds)
    probabilities = []
    targets = []
    for fold_index, held_indices in enumerate(folds):
        held_set = {int(index) for index in held_indices}
        train_entries = [
            entry for index, entry in enumerate(support) if index not in held_set
        ]
        held_entries = [support[index] for index in sorted(held_set)]
        model, _, _ = new_adapted_model(
            args,
            device,
            train_entries,
            mean,
            std,
            args.support_seed + fold_index + 1,
        )
        probability, target = collect_predictions(
            model,
            make_entry_loader(held_entries, args, mean, std, device, training=False),
            device=device,
            amp=not args.no_amp,
        )
        probabilities.append(probability)
        targets.append(target)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    threshold, _ = select_pixel_threshold(
        np.concatenate(probabilities), np.concatenate(targets)
    )
    return threshold


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, checkpoint = load_checkpoint(args.checkpoint, device=device)
    mean = checkpoint["mean11"]
    std = checkpoint["std11"]
    target_entries = index_cluster(args.data_root, args.held)
    support = draw_support(
        target_entries,
        args.support_size,
        seed=args.support_seed,
        strategy=args.support_sampling,
    )
    query = exclude_entries(target_entries, support)
    source_model, _ = load_checkpoint(args.checkpoint, device=device)
    source_probabilities, source_targets = collect_predictions(
        source_model,
        make_entry_loader(query, args, mean, std, device, training=False),
        device=device,
        amp=not args.no_amp,
    )
    source_pixel = pixel_metrics(source_probabilities, source_targets, 0.5)
    source_component = component_metrics(
        source_probabilities, source_targets, 0.5, iou_threshold=0.3
    )
    del source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.threshold_mode == "cross-fit":
        threshold = cross_fit_threshold(args, device, support, mean, std)
    else:
        threshold = 0.5
    model, trainable, losses = new_adapted_model(
        args, device, support, mean, std, args.support_seed
    )
    if args.threshold_mode == "support":
        probabilities, targets = collect_predictions(
            model,
            make_entry_loader(support, args, mean, std, device, training=False),
            device=device,
            amp=not args.no_amp,
        )
        threshold, _ = select_pixel_threshold(probabilities, targets)
    query_probabilities, query_targets = collect_predictions(
        model,
        make_entry_loader(query, args, mean, std, device, training=False),
        device=device,
        amp=not args.no_amp,
    )
    payload = {
        "held": args.held,
        "checkpoint": str(args.checkpoint),
        "source_epoch": checkpoint.get("epoch"),
        "support_size": len(support),
        "support_seed": args.support_seed,
        "support_sampling": args.support_sampling,
        "support_indices": [entry.index for entry in support],
        "query_size": len(query),
        "adapt_mode": args.adapt_mode,
        "trainable_parameters": trainable,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "threshold_mode": args.threshold_mode,
        "cross_fit_folds": (
            args.cross_fit_folds if args.threshold_mode == "cross-fit" else None
        ),
        "threshold": threshold,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "source_at_0.5": {
            "pixel": source_pixel.to_dict(),
            "component": source_component.to_dict(),
        },
        "pixel": pixel_metrics(query_probabilities, query_targets, threshold).to_dict(),
        "component": component_metrics(
            query_probabilities, query_targets, threshold, iou_threshold=0.3
        ).to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
