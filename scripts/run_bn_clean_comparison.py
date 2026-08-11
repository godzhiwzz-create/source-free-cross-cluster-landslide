#!/usr/bin/env python3
"""Run one paired source/oracle/BN-clean-decoder/full revision cell."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

import landslide_sfda.engine as engine_module
import landslide_sfda.model as model_module
from landslide_sfda.constants import CLUSTERS
from landslide_sfda.data import build_dataset, draw_support, exclude_entries, index_cluster
from landslide_sfda.engine import (
    adaptation_scope_metadata,
    collect_predictions,
    configure_adaptation,
    load_checkpoint,
    make_loader,
    set_seed,
    train_steps,
)
from landslide_sfda.metrics import pixel_metrics, select_pixel_threshold


ENCODER_LIKE = ("en3", "en4", "center_in", "center_out")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(entries) -> str:
    text = "\n".join(
        f"{entry.cluster}\t{entry.index}"
        for entry in sorted(entries, key=lambda item: (item.cluster, item.index))
    ).encode()
    return hashlib.sha256(text).hexdigest()


def summarize(entries) -> dict:
    positive = sum(entry.positive_pixels > 0 for entry in entries)
    return {
        "n_tiles": len(entries),
        "n_positive_tiles": positive,
        "n_negative_tiles": len(entries) - positive,
        "positive_pixels": sum(entry.positive_pixels for entry in entries),
        "identity_sha256": identity_hash(entries),
        "indices": [entry.index for entry in entries],
    }


def loader(args, entries, mean, std, device, *, training):
    return make_loader(
        build_dataset(args.data_root, entries, mean, std, augment=training),
        batch_size=args.batch_size if training else args.eval_batch_size,
        shuffle=training,
        workers=args.workers,
        device=device,
    )


def capture_encoder_state(model) -> dict[str, torch.Tensor]:
    return {
        f"{module_name}.{state_name}": value.detach().cpu().clone()
        for module_name in ENCODER_LIKE
        for state_name, value in getattr(model, module_name).state_dict().items()
    }


def adapt_and_evaluate(args, mode, support, query, mean, std, device):
    set_seed(args.model_seed)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    trainable = configure_adaptation(model, mode)
    before = capture_encoder_state(model) if mode == "decoder-clean" else None
    losses = train_steps(
        model,
        loader(args, support, mean, std, device, training=True),
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        amp=not args.no_amp,
    )
    invariant = None
    if before is not None:
        after = capture_encoder_state(model)
        changed = [key for key, value in before.items() if not torch.equal(value, after[key])]
        invariant = {"pass": not changed, "changed_tensors": changed}
        if changed:
            raise RuntimeError(f"clean decoder changed frozen encoder state: {changed}")
    probabilities, targets = collect_predictions(
        model,
        loader(args, query, mean, std, device, training=False),
        device=device,
        amp=not args.no_amp,
    )
    return {
        "mode": mode,
        "trainable_parameters": trainable,
        "scope": adaptation_scope_metadata(mode),
        "frozen_encoder_state_invariant": invariant,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "fixed_0_5": pixel_metrics(probabilities, targets, 0.5).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--held", required=True, choices=CLUSTERS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--support-size", type=int, default=50)
    parser.add_argument("--support-draw", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, checkpoint = load_checkpoint(args.checkpoint, device=device)
    mean, std = checkpoint["mean11"], checkpoint["std11"]
    target = index_cluster(args.data_root, args.held)
    support = draw_support(
        target,
        args.support_size,
        seed=args.support_draw,
        strategy="stratified-prevalence",
    )
    query = exclude_entries(target, support)
    if set(support).intersection(query) or len(support) + len(query) != len(target):
        raise RuntimeError("support/query partition failed")
    source_model, _ = load_checkpoint(args.checkpoint, device=device)
    source_probability, source_target = collect_predictions(
        source_model,
        loader(args, query, mean, std, device, training=False),
        device=device,
        amp=not args.no_amp,
    )
    oracle_threshold, _ = select_pixel_threshold(source_probability, source_target)
    payload = {
        "status": "complete",
        "held": args.held,
        "source_seed": args.source_seed,
        "source_epoch": checkpoint.get("epoch"),
        "support_draw": args.support_draw,
        "support_sampling": "stratified-prevalence",
        "support": summarize(support),
        "query": summarize(query),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "source_fixed_0_5": pixel_metrics(source_probability, source_target, 0.5).to_dict(),
        "source_query_label_oracle": {
            "threshold": oracle_threshold,
            "pixel": pixel_metrics(source_probability, source_target, oracle_threshold).to_dict(),
            "access": "non-deployable query-label diagnostic ceiling",
        },
        "decoder_clean": adapt_and_evaluate(args, "decoder-clean", support, query, mean, std, device),
        "full": adapt_and_evaluate(args, "full", support, query, mean, std, device),
        "provenance": {
            "checkpoint_sha256": sha256(args.checkpoint),
            "runner_sha256": sha256(__file__),
            "engine_sha256": sha256(inspect.getsourcefile(engine_module)),
            "model_sha256": sha256(inspect.getsourcefile(model_module)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
