"""Training, adaptation, inference, and checkpoint helpers."""

from __future__ import annotations

import contextlib
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import UNet3DPaper


ENCODER_LIKE_MODULES = ("en3", "en4", "center_in", "center_out")
DECODER_HEAD_MODULES = ("dc4", "trans3", "dc3", "final")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class SegmentationLoss(nn.Module):
    def __init__(
        self, pos_weight: float = 5.0, dice_weight: float = 0.5, smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        probability = torch.sigmoid(logits)
        intersection = (probability * target).sum()
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (probability.sum() + target.sum() + self.smooth)
        )
        return bce + self.dice_weight * dice_loss


def make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return patch-shaped probabilities and binary targets."""
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(x)
            if logits.shape != y.shape:
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False
                )
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
        targets.append(y.cpu().numpy()[:, 0].astype(np.uint8))
    if not probabilities:
        raise ValueError("cannot evaluate an empty loader")
    return np.concatenate(probabilities), np.concatenate(targets)


def train_steps(
    model: nn.Module,
    loader: DataLoader,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    amp: bool = True,
    grad_clip: float = 1.0,
    criterion: nn.Module | None = None,
) -> list[float]:
    """Fine-tune for exactly ``steps`` optimizer updates."""
    if steps <= 0:
        return []
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("the selected adaptation mode has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    criterion = criterion or SegmentationLoss()
    criterion = criterion.to(device)
    losses: list[float] = []
    set_adaptation_train_mode(model)
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(x)
            if logits.shape != y.shape:
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def configure_adaptation(model: UNet3DPaper, mode: str) -> int:
    """Select trainable weights and record the associated model-state policy.

    ``decoder`` reproduces the historical protocol: encoder-like weights are
    frozen but a later global training-mode call permits their BatchNorm
    running statistics to update. ``decoder-clean`` freezes both those weights
    and their BatchNorm state by keeping the encoder-like modules in evaluation
    mode during adaptation.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True
    if mode == "full":
        pass
    elif mode in {"decoder", "decoder-clean"}:
        for name in ENCODER_LIKE_MODULES:
            module = getattr(model, name)
            for parameter in module.parameters():
                parameter.requires_grad = False
    elif mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.final.parameters():
            parameter.requires_grad = True
    elif mode == "bn":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in model.modules():
            if isinstance(module, nn.BatchNorm3d):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    else:
        raise ValueError(f"unknown adaptation mode: {mode}")
    model._landslide_adaptation_mode = mode
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def set_adaptation_train_mode(model: UNet3DPaper) -> None:
    """Apply the training/evaluation state required by the configured mode."""

    model.train()
    if getattr(model, "_landslide_adaptation_mode", None) == "decoder-clean":
        for name in ENCODER_LIKE_MODULES:
            getattr(model, name).eval()


def adaptation_scope_metadata(mode: str) -> dict[str, object]:
    """Return an explicit, serializable scope and BatchNorm-state description."""

    if mode == "full":
        return {
            "trainable_modules": "all network modules",
            "frozen_modules": [],
            "batchnorm_state_policy": "ordinary training-mode updates throughout network",
        }
    if mode in {"decoder", "decoder-clean"}:
        return {
            "trainable_modules": list(DECODER_HEAD_MODULES),
            "frozen_modules": list(ENCODER_LIKE_MODULES),
            "batchnorm_state_policy": (
                "frozen encoder-like modules remain in evaluation mode"
                if mode == "decoder-clean"
                else "historical global training mode; frozen-module running statistics may update"
            ),
        }
    if mode == "head":
        return {
            "trainable_modules": ["final"],
            "frozen_modules": "all remaining weight tensors",
            "batchnorm_state_policy": "historical global training mode; frozen-module running statistics may update",
        }
    if mode == "bn":
        return {
            "trainable_modules": "all BatchNorm affine parameters",
            "frozen_modules": "all non-BatchNorm weight tensors",
            "batchnorm_state_policy": "ordinary training-mode running-statistic updates",
        }
    raise ValueError(f"unknown adaptation mode: {mode}")


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[UNet3DPaper, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = UNet3DPaper(in_channels=11).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    mean: Iterable[float],
    std: Iterable[float],
    metadata: dict,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "mean11": list(mean),
            "std11": list(std),
            "metadata": metadata,
        },
        destination,
    )
