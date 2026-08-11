#!/usr/bin/env python3
"""Run one frozen target-unlabelled transductive control on a held-out cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from landslide_sfda.constants import CLUSTERS
from landslide_sfda.data import (
    ClusterInputDataset,
    build_dataset,
    index_cluster,
    input_indices,
)
from landslide_sfda.engine import (
    autocast_context,
    collect_predictions,
    configure_adaptation,
    load_checkpoint,
    make_loader,
    set_adaptation_train_mode,
    set_seed,
)
from landslide_sfda.metrics import pixel_metrics
from landslide_sfda.source_free import binary_entropy, exact_class_balanced_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--held", required=True, choices=CLUSTERS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--method", required=True, choices=("target-entropy", "class-balanced-pseudo"))
    parser.add_argument("--query-indices", type=Path, help="optional JSON list or {'indices': [...]} manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pseudo-fraction", type=float, default=0.2)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_indices(args: argparse.Namespace) -> list[int]:
    if args.query_indices is None:
        return input_indices(args.data_root, args.held)
    payload = json.loads(args.query_indices.read_text())
    values = payload["indices"] if isinstance(payload, dict) else payload
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError("query indices contain duplicates")
    return indices


def identity_hash(cluster: str, indices: list[int]) -> str:
    text = "\n".join(f"{cluster}\t{index}" for index in sorted(indices)).encode()
    return hashlib.sha256(text).hexdigest()


@torch.no_grad()
def input_probabilities(model, loader, device, amp) -> np.ndarray:
    model.eval()
    values = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(x)
        values.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
    return np.concatenate(values)


class StaticPseudoDataset(Dataset):
    def __init__(self, args, indices, mean, std, pseudo, selected) -> None:
        self.inputs = ClusterInputDataset(
            args.data_root, args.held, indices, mean, std, augment=False
        )
        self.indices = indices
        self.pseudo = pseudo
        self.selected = selected

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        x = self.inputs[item]["x"].numpy()
        pseudo = self.pseudo[item].copy()
        selected = self.selected[item].copy()
        if np.random.random() < 0.5:
            x, pseudo, selected = x[:, :, :, ::-1].copy(), pseudo[:, ::-1].copy(), selected[:, ::-1].copy()
        if np.random.random() < 0.5:
            x, pseudo, selected = x[:, :, ::-1, :].copy(), pseudo[::-1, :].copy(), selected[::-1, :].copy()
        rotations = int(np.random.randint(0, 4))
        if rotations:
            x = np.rot90(x, rotations, axes=(2, 3)).copy()
            pseudo = np.rot90(pseudo, rotations, axes=(0, 1)).copy()
            selected = np.rot90(selected, rotations, axes=(0, 1)).copy()
        return {
            "x": torch.from_numpy(x),
            "pseudo": torch.from_numpy(pseudo.astype(np.float32)).unsqueeze(0),
            "selected": torch.from_numpy(selected).unsqueeze(0),
        }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = load_indices(args)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    configure_adaptation(model, "full")
    mean, std = checkpoint["mean11"], checkpoint["std11"]
    input_dataset = ClusterInputDataset(args.data_root, args.held, indices, mean, std)
    input_loader = make_loader(input_dataset, batch_size=args.eval_batch_size, shuffle=False, workers=args.workers, device=device)
    source_probability = input_probabilities(model, input_loader, device, not args.no_amp)

    pseudo_report = None
    if args.method == "class-balanced-pseudo":
        pseudo, selected, pseudo_report = exact_class_balanced_selection(source_probability, fraction=args.pseudo_fraction)
        training_dataset = StaticPseudoDataset(args, indices, mean, std, pseudo, selected)
    else:
        training_dataset = ClusterInputDataset(args.data_root, args.held, indices, mean, std, augment=True)
    set_seed(args.seed)
    training_loader = make_loader(training_dataset, batch_size=args.batch_size, shuffle=True, workers=args.workers, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    set_adaptation_train_mode(model)
    losses = []
    selected_pixels = 0
    seen_pixels = 0
    for batch in training_loader:
        x = batch["x"].to(device, non_blocking=True)
        with autocast_context(device, not args.no_amp):
            logits = model(x)
            if args.method == "target-entropy":
                loss = binary_entropy(torch.sigmoid(logits.float()))
                seen_pixels += logits.numel()
                selected_pixels += logits.numel()
            else:
                pseudo = batch["pseudo"].to(device, non_blocking=True)
                selected = batch["selected"].to(device, non_blocking=True)
                if logits.shape != pseudo.shape:
                    logits = F.interpolate(logits, size=pseudo.shape[-2:], mode="bilinear", align_corners=False)
                per_pixel = F.binary_cross_entropy_with_logits(logits.float(), pseudo, reduction="none")
                selected_count = int(selected.sum().item())
                if selected_count == 0:
                    raise RuntimeError("pseudo-label batch has no selected pixels")
                loss = (per_pixel * selected).sum() / selected.sum()
                seen_pixels += selected.numel()
                selected_pixels += selected_count
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    # Target labels are opened only after adaptation has finished, and are used
    # exclusively to report source/adapted metrics on the frozen query order.
    entry_map = {entry.index: entry for entry in index_cluster(args.data_root, args.held)}
    evaluation_entries = [entry_map[index] for index in indices]
    evaluation_loader = make_loader(
        build_dataset(args.data_root, evaluation_entries, mean, std),
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.workers,
        device=device,
    )
    adapted_probability, targets = collect_predictions(
        model, evaluation_loader, device=device, amp=not args.no_amp
    )
    expected_updates = math.ceil(len(indices) / args.batch_size)
    if len(losses) != expected_updates or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("incomplete or non-finite one-pass training")
    payload = {
        "status": "complete",
        "access": "source-free, target-unlabelled, transductive_same_query_after_adaptation",
        "held": args.held,
        "source_seed": args.source_seed,
        "method": args.method,
        "adaptation_seed": args.seed,
        "source_epoch": checkpoint.get("epoch"),
        "query_size": len(indices),
        "query_identity_sha256": identity_hash(args.held, indices),
        "training_passes": 1,
        "updates": len(losses),
        "expected_updates": expected_updates,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "pseudo_fraction": args.pseudo_fraction if args.method == "class-balanced-pseudo" else None,
        "pseudo_selection": pseudo_report,
        "selected_pixels": selected_pixels,
        "seen_pixels": seen_pixels,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "source_fixed_0_5": pixel_metrics(source_probability, targets, 0.5).to_dict(),
        "adapted_fixed_0_5": pixel_metrics(adapted_probability, targets, 0.5).to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
