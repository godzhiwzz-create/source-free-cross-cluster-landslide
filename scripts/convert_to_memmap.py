#!/usr/bin/env python3
"""Convert preprocessed Sen12Landslides NPZ archives to cluster memmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from landslide_sfda.constants import CLUSTERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", nargs="+", default=list(CLUSTERS))
    parser.add_argument(
        "--glob-template",
        default="{cluster}__*.npz",
        help="NPZ glob relative to input-dir; must include {cluster}",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing cluster"
    )
    return parser.parse_args()


def inspect(
    files: list[Path],
) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    total = 0
    x_shape = y_shape = m_shape = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "X" not in archive or "Y" not in archive:
                raise KeyError(f"{path} must contain X and Y")
            total += len(archive["X"])
            x_shape = x_shape or archive["X"].shape[1:]
            y_shape = y_shape or archive["Y"].shape[1:]
            current_m = archive["M_mod"].shape[1:] if "M_mod" in archive else (15, 3)
            m_shape = m_shape or current_m
            if archive["X"].shape[1:] != x_shape or archive["Y"].shape[1:] != y_shape:
                raise ValueError(f"inconsistent patch shape in {path}")
    if not total or x_shape is None or y_shape is None or m_shape is None:
        raise ValueError("no patches found")
    return total, tuple(x_shape), tuple(y_shape), tuple(m_shape)


def convert_cluster(
    input_dir: Path,
    output_dir: Path,
    cluster: str,
    glob_template: str,
    overwrite: bool,
) -> None:
    files = sorted(input_dir.glob(glob_template.format(cluster=cluster)))
    if not files:
        raise FileNotFoundError(f"no archives found for {cluster}")
    total, x_shape, y_shape, m_shape = inspect(files)
    if len(x_shape) != 4 or x_shape[1] != 14:
        raise ValueError(
            f"{cluster}: expected X patch shape (T, 14, H, W), found {x_shape}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "X": output_dir / f"{cluster}.X.dat",
        "Y": output_dir / f"{cluster}.Y.dat",
        "M": output_dir / f"{cluster}.M.dat",
        "meta": output_dir / f"{cluster}.meta.json",
    }
    if paths["meta"].exists() and not overwrite:
        print(f"[skip] {cluster}: metadata already exists")
        return
    x_map = np.memmap(paths["X"], dtype=np.float16, mode="w+", shape=(total, *x_shape))
    y_map = np.memmap(paths["Y"], dtype=np.uint8, mode="w+", shape=(total, *y_shape))
    m_map = np.memmap(paths["M"], dtype=np.uint8, mode="w+", shape=(total, *m_shape))
    offset = 0
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            size = len(archive["X"])
            x_map[offset : offset + size] = archive["X"]
            y_map[offset : offset + size] = archive["Y"]
            if "M_mod" in archive:
                m_map[offset : offset + size] = archive["M_mod"]
            else:
                m_map[offset : offset + size] = 1
            offset += size
    x_map.flush()
    y_map.flush()
    m_map.flush()
    metadata = {
        "n_patches": total,
        "X_shape_per_patch": list(x_shape),
        "Y_shape_per_patch": list(y_shape),
        "M_shape_per_patch": list(m_shape),
        "X_dtype": "float16",
        "Y_dtype": "uint8",
        "M_dtype": "uint8",
        "source_archives": [path.name for path in files],
    }
    paths["meta"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] {cluster}: {total} patches")


def main() -> None:
    args = parse_args()
    for cluster in args.clusters:
        convert_cluster(
            args.input_dir,
            args.output_dir,
            cluster,
            args.glob_template,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
