#!/usr/bin/env python3
"""Train one fixed-budget leave-one-cluster-out source model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from landslide_sfda.constants import CLUSTERS
from landslide_sfda.data import (
    build_dataset,
    compute_fold_normalization,
    index_cluster,
)
from landslide_sfda.engine import (
    SegmentationLoss,
    autocast_context,
    collect_predictions,
    make_loader,
    set_seed,
)
from landslide_sfda.metrics import pixel_metrics
from landslide_sfda.model import UNet3DPaper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--held", choices=CLUSTERS, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="optional intermediate held-out reporting interval; 0 evaluates only at the final epoch",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--limit-per-cluster",
        type=int,
        default=None,
        help="debug only: cap patches in every cluster",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp
    source_entries = []
    held_entries = []
    for cluster in CLUSTERS:
        entries = index_cluster(args.data_root, cluster, limit=args.limit_per_cluster)
        if cluster == args.held:
            held_entries.extend(entries)
        else:
            source_entries.extend(entries)
    mean, std = compute_fold_normalization(
        args.data_root, source_entries, seed=args.seed
    )
    source_dataset = build_dataset(
        args.data_root, source_entries, mean, std, augment=True
    )
    held_dataset = build_dataset(args.data_root, held_entries, mean, std, augment=False)
    train_loader = make_loader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        device=device,
    )
    held_loader = make_loader(
        held_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        device=device,
    )
    model = UNet3DPaper(in_channels=11).to(device)
    criterion = SegmentationLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_dir = args.output_dir / f"source_{args.held}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    positive_pixels = sum(entry.positive_pixels for entry in source_entries)
    pixels_per_patch = int(np.prod(held_dataset.datasets[0].data.Y.shape[-2:]))
    source_prior = positive_pixels / (len(source_entries) * pixels_per_patch)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with autocast_context(device, amp):
                logits = model(x)
                if logits.shape != y.shape:
                    logits = F.interpolate(
                        logits,
                        size=y.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "seconds": time.time() - start,
        }
        if (
            args.eval_every > 0 and epoch % args.eval_every == 0
        ) or epoch == args.epochs:
            probabilities, targets = collect_predictions(
                model, held_loader, device=device, amp=amp
            )
            record["held_at_0.5"] = pixel_metrics(probabilities, targets, 0.5).to_dict()
        history.append(record)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "mean11": mean.tolist(),
            "std11": std.tolist(),
            "metadata": {
                "held": args.held,
                "seed": args.seed,
                "source_prior": source_prior,
                "fixed_budget_checkpoint": True,
                "train_patches": len(source_entries),
                "held_patches": len(held_entries),
                "arguments": vars(args)
                | {
                    "data_root": str(args.data_root),
                    "output_dir": str(args.output_dir),
                },
            },
        }
        torch.save(checkpoint, run_dir / "last.pt")
        print(json.dumps(record), flush=True)
    summary = {
        "held": args.held,
        "seed": args.seed,
        "checkpoint": str(run_dir / "last.pt"),
        "source_prior": source_prior,
        "history": history,
    }
    (run_dir / "history.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
