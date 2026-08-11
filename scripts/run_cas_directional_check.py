#!/usr/bin/env python3
"""Run the manuscript's single-run five-event directional CAS check."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from landslide_sfda.engine import (
    SegmentationLoss,
    autocast_context,
    configure_adaptation,
    set_adaptation_train_mode,
    set_seed,
)
from landslide_sfda.metrics import pixel_metrics, select_pixel_threshold
from landslide_sfda.model import UNet3DPaper


DEFAULT_EVENTS = {
    "Palu": "palu_x",
    "Lombok": "Lombok_x",
    "Hokkaido": "Hokkaido Iburi-Tobu_x",
    "Tiburon_S": "Tiburon Peninsula（Sentinel）_x",
    "Tiburon_P": "Tiburon Peninsula（planet）_x",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raster(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path))


def parse_events(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_EVENTS)
    events = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory or name in events:
            raise ValueError(f"invalid or duplicate --event value: {value!r}")
        events[name] = directory
    if len(events) < 2:
        raise ValueError("CAS leave-one-event-out evaluation requires at least two events")
    return events


def list_event(root: Path, directory: str) -> list[tuple[Path, str]]:
    items = [
        (path.parent.parent, path.name)
        for path in sorted((root / directory).glob("**/img/*.tif"))
    ]
    if not items:
        raise ValueError(f"no CAS TIFF images found under {root / directory}")
    return items


def mask_path(item: tuple[Path, str]) -> Path:
    return item[0] / "mask" / item[1]


def identity(item: tuple[Path, str], root: Path) -> str:
    return str((item[0] / "img" / item[1]).relative_to(root))


def positive_fraction(item: tuple[Path, str]) -> float:
    return float((read_raster(mask_path(item)) > 0).mean())


class CASDataset(Dataset):
    def __init__(self, items: list[tuple[Path, str]], *, augment: bool = False) -> None:
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base, name = self.items[index]
        image = read_raster(base / "img" / name).astype(np.float32)
        mask = read_raster(base / "mask" / name).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        image_tensor = torch.from_numpy(image[..., :3]).permute(2, 0, 1) / 255.0
        image_tensor = F.interpolate(
            image_tensor[None], size=(128, 128), mode="bilinear", align_corners=False
        )[0]
        mask_tensor = F.interpolate(
            torch.from_numpy(mask)[None, None], size=(128, 128), mode="nearest"
        )[0]
        if self.augment and np.random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, (-1,))
            mask_tensor = torch.flip(mask_tensor, (-1,))
        return {
            "x": image_tensor[None].repeat(15, 1, 1, 1),
            "y": (mask_tensor > 0.5).float(),
        }


def loader(items, batch_size, shuffle, device, *, augment=False, workers=4):
    return DataLoader(
        CASDataset(items, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def collect(model, items, args, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for batch in loader(items, args.eval_batch_size, False, device, workers=args.workers):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].numpy()[:, 0]
            with autocast_context(device, not args.no_amp):
                logits = model(x)
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
            targets.append(y)
    return np.concatenate(probabilities), np.concatenate(targets)


def train_source(items, args, device) -> UNet3DPaper:
    set_seed(args.seed)
    model = UNet3DPaper(in_channels=3).to(device)
    criterion = SegmentationLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    training_loader = loader(items, 16, True, device, augment=True, workers=args.workers)
    model.train()
    for _ in range(args.source_epochs):
        for batch in training_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with autocast_context(device, not args.no_amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def adapt(source_state, support, mode, args, device) -> UNet3DPaper:
    model = UNet3DPaper(in_channels=3).to(device)
    model.load_state_dict(source_state)
    configure_adaptation(model, mode)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    criterion = SegmentationLoss().to(device)
    training_loader = loader(support, 8, True, device, augment=True, workers=args.workers)
    iterator = iter(training_loader)
    set_adaptation_train_mode(model)
    for _ in range(args.adaptation_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(training_loader)
            batch = next(iterator)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with autocast_context(device, not args.no_amp):
            loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    return model


def select_support(items, size, seed):
    rng = np.random.RandomState(100 + seed)
    positive = [item for item in items if positive_fraction(item) > 0.001]
    if not positive:
        raise ValueError("CAS support sampler found no positive event tiles")
    positive_set = set(positive)
    negative = [item for item in items if item not in positive_set]
    n_positive = min(len(positive), max(1, round(size * len(positive) / len(items))))
    n_negative = min(size - n_positive, len(negative))
    support = [positive[index] for index in rng.choice(len(positive), n_positive, replace=False)]
    if n_negative:
        support.extend(negative[index] for index in rng.choice(len(negative), n_negative, replace=False))
    return support, n_positive


def rank_candidate_scores(
    *,
    source_f1: float,
    oracle_f1: float,
    threshold_f1: float,
    decoder_f1: float,
    full_fixed_f1: float,
    full_recipe_f1: float,
) -> dict[str, object]:
    """Rank the exact finite candidate set reported in the manuscript."""

    scores = {
        "source_0_5": source_f1,
        "query_label_oracle": oracle_f1,
        "threshold_only": threshold_f1,
        "decoder_fixed": decoder_f1,
        "full_fixed": full_fixed_f1,
        "full_recipe": full_recipe_f1,
    }
    best_strategy = max(scores, key=scores.get)
    best_full_strategy = max(("full_fixed", "full_recipe"), key=scores.get)
    return {
        "scores": scores,
        "best_strategy": best_strategy,
        "best_full_strategy": best_full_strategy,
        "best_full_regret": scores[best_strategy] - scores[best_full_strategy],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--event", action="append", help="NAME=relative_directory; repeat for every event")
    parser.add_argument("--source-epochs", type=int, default=30)
    parser.add_argument("--support-size", type=int, default=50)
    parser.add_argument("--adaptation-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    events = parse_events(args.event)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    event_items = {name: list_event(args.data_root, directory) for name, directory in events.items()}
    output = {
        "status": "in_progress",
        "run_count": 1,
        "source_seed": args.seed,
        "source_epochs": args.source_epochs,
        "support_size": args.support_size,
        "adaptation_steps": args.adaptation_steps,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "support_threshold_rule": "best F1 on support over 0.05 to 0.95 in increments of 0.05",
        "query_label_oracle": "non-deployable diagnostic ceiling",
        "decoder_policy": "historical decoder weights with global training-mode BatchNorm state",
        "events": {},
        "runner_sha256": sha256(Path(__file__)),
    }
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for held, held_items in event_items.items():
        train_items = [item for name, items in event_items.items() if name != held for item in items]
        support, n_positive = select_support(held_items, args.support_size, args.seed)
        support_set = set(support)
        query = [item for item in held_items if item not in support_set]
        source = train_source(train_items, args, device)
        checkpoint = args.checkpoint_dir / f"cas_source_{held}_seed{args.seed}.pt"
        torch.save(
            {"model": source.state_dict(), "epoch": args.source_epochs, "seed": args.seed, "held": held},
            checkpoint,
        )
        source_state = copy.deepcopy(source.state_dict())
        source_probability, query_target = collect(source, query, args, device)
        support_probability, support_target = collect(source, support, args, device)
        oracle_threshold, _ = select_pixel_threshold(source_probability, query_target)
        support_threshold, _ = select_pixel_threshold(support_probability, support_target)
        source_f1 = pixel_metrics(source_probability, query_target, 0.5).f1
        oracle_f1 = pixel_metrics(source_probability, query_target, oracle_threshold).f1
        threshold_f1 = pixel_metrics(source_probability, query_target, support_threshold).f1
        del source

        decoder = adapt(source_state, support, "decoder", args, device)
        decoder_probability, _ = collect(decoder, query, args, device)
        decoder_f1 = pixel_metrics(decoder_probability, query_target, 0.5).f1
        del decoder

        full = adapt(source_state, support, "full", args, device)
        full_probability, _ = collect(full, query, args, device)
        full_support_probability, full_support_target = collect(full, support, args, device)
        full_threshold, _ = select_pixel_threshold(full_support_probability, full_support_target)
        full_fixed_f1 = pixel_metrics(full_probability, query_target, 0.5).f1
        full_recipe_f1 = pixel_metrics(full_probability, query_target, full_threshold).f1
        ranking = rank_candidate_scores(
            source_f1=source_f1,
            oracle_f1=oracle_f1,
            threshold_f1=threshold_f1,
            decoder_f1=decoder_f1,
            full_fixed_f1=full_fixed_f1,
            full_recipe_f1=full_recipe_f1,
        )
        output["events"][held] = {
            "train": len(train_items),
            "held": len(held_items),
            "support": len(support),
            "positive_support": n_positive,
            "query": len(query),
            "support_identities": [identity(item, args.data_root) for item in support],
            "checkpoint_sha256": sha256(checkpoint),
            "source_0_5": source_f1,
            "query_label_oracle": oracle_f1,
            "threshold_only": threshold_f1,
            "decoder_fixed": decoder_f1,
            "full_fixed": full_fixed_f1,
            "full_recipe": full_recipe_f1,
            "candidate_scores": ranking["scores"],
            "best_strategy_in_finite_set": ranking["best_strategy"],
            "best_full_strategy": ranking["best_full_strategy"],
            "best_full_regret_in_finite_set": ranking["best_full_regret"],
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2) + "\n")
        temporary.replace(args.output)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output["status"] = "complete"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
