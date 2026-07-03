#!/usr/bin/env python3
"""Convert official harmonized Sen12Landslides NetCDF files to experiment NPZs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import xarray as xr


S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
S1_BANDS = ("VV", "VH")
SAR_TOLERANCE_SECONDS = 30 * 86400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s2-root",
        type=Path,
        required=True,
        help="extracted data_harmonized/s2 directory",
    )
    parser.add_argument(
        "--s1asc-root",
        type=Path,
        default=None,
        help="optional extracted data_harmonized/s1asc directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cluster-map",
        type=Path,
        default=Path("configs/region_to_cluster.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="debug-only file cap")
    return parser.parse_args()


def patch_id(path: Path) -> str:
    name = path.stem
    return name.replace("_s2_", "_").replace("_s1asc_", "_")


def region(path: Path) -> str:
    return path.name.split("_s2_")[0].split("_s1asc_")[0].lower()


def load_cluster_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    mapping = {}
    for cluster, regions in payload["clusters"].items():
        for name in regions:
            if name in mapping:
                raise ValueError(f"region appears in multiple clusters: {name}")
            mapping[name.lower()] = cluster
    return mapping


def read_s2(path: Path) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as dataset:
        bands = (
            np.stack([dataset[band].values for band in S2_BANDS], axis=1).astype(
                np.float32
            )
            / 10000.0
        )
        scl = dataset["SCL"].values.astype(np.int16)
        mask = (dataset["MASK"].values[0] > 0).astype(np.uint8)
        dem = dataset["DEM"].values[0].astype(np.float32) / 1000.0
        times = dataset["time"].values.astype("datetime64[s]").astype(np.int64)
    return {
        "bands": bands.astype(np.float16),
        "scl": scl,
        "mask": mask,
        "dem": dem.astype(np.float16),
        "times": times,
    }


def read_s1(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as dataset:
        bands = np.stack([dataset[band].values for band in S1_BANDS], axis=1).astype(
            np.float32
        )
        bands = np.clip(bands, -30.0, 0.0) / 15.0 - 1.0
        times = dataset["time"].values.astype("datetime64[s]").astype(np.int64)
    return bands.astype(np.float16), times


def align_s1(
    s2_times: np.ndarray, s1_bands: np.ndarray, s1_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.zeros(
        (len(s2_times), 2, s1_bands.shape[2], s1_bands.shape[3]), dtype=np.float16
    )
    available = np.zeros(len(s2_times), dtype=np.uint8)
    for time_index, timestamp in enumerate(s2_times):
        distances = np.abs(s1_times - timestamp)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= SAR_TOLERANCE_SECONDS:
            aligned[time_index] = s1_bands[nearest]
            available[time_index] = 1
    return aligned, available


def build_patch(
    s2_path: Path, s1_path: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s2 = read_s2(s2_path)
    time_count = len(s2["times"])
    height, width = s2["mask"].shape
    if s1_path is None:
        sar = np.zeros((time_count, 2, height, width), dtype=np.float16)
        sar_available = np.zeros(time_count, dtype=np.uint8)
    else:
        s1_bands, s1_times = read_s1(s1_path)
        sar, sar_available = align_s1(s2["times"], s1_bands, s1_times)
    dem = np.broadcast_to(s2["dem"][None, None], (time_count, 1, height, width)).copy()
    scl_clean = (~np.isin(s2["scl"], [1, 3, 8, 9, 10])).astype(np.float16)
    x = np.concatenate((s2["bands"], sar, dem, scl_clean[:, None]), axis=1).astype(
        np.float16
    )
    s2_available = (scl_clean.mean(axis=(1, 2)) > 0.3).astype(np.uint8)
    dem_available = np.full(
        time_count, int(not np.isnan(s2["dem"]).any()), dtype=np.uint8
    )
    modalities = np.stack((s2_available, sar_available, dem_available), axis=1)
    return x, s2["mask"], modalities, s2["times"]


def flush(
    output_dir: Path, cluster: str, chunk_index: int, records: list[dict]
) -> None:
    if not records:
        return
    output = output_dir / f"{cluster}__public__c{chunk_index:04d}.npz"
    np.savez_compressed(
        output,
        X=np.stack([record["X"] for record in records]),
        Y=np.stack([record["Y"] for record in records]),
        M_mod=np.stack([record["M_mod"] for record in records]),
        region=np.asarray([record["region"] for record in records]),
        times=np.stack([record["times"] for record in records]),
    )
    print(f"[saved] {output.name}: {len(records)} patches", flush=True)


def main() -> None:
    args = parse_args()
    cluster_map = load_cluster_map(args.cluster_map)
    s2_files = sorted(args.s2_root.rglob("*.nc"))
    if args.limit is not None:
        s2_files = s2_files[: args.limit]
    if not s2_files:
        raise FileNotFoundError(f"no NetCDF files below {args.s2_root}")
    s1_index = {}
    if args.s1asc_root is not None:
        s1_index = {patch_id(path): path for path in args.s1asc_root.rglob("*.nc")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, list[dict]] = defaultdict(list)
    chunk_indices: dict[str, int] = defaultdict(int)
    skipped_regions: dict[str, int] = defaultdict(int)
    for path in s2_files:
        location = region(path)
        cluster = cluster_map.get(location)
        if cluster is None:
            skipped_regions[location] += 1
            continue
        x, y, modality_mask, times = build_patch(path, s1_index.get(patch_id(path)))
        buffers[cluster].append(
            {
                "X": x,
                "Y": y,
                "M_mod": modality_mask,
                "region": location,
                "times": times,
            }
        )
        if len(buffers[cluster]) >= args.chunk_size:
            flush(
                args.output_dir,
                cluster,
                chunk_indices[cluster],
                buffers[cluster],
            )
            chunk_indices[cluster] += 1
            buffers[cluster].clear()
    for cluster, records in buffers.items():
        flush(args.output_dir, cluster, chunk_indices[cluster], records)
    if skipped_regions:
        print(f"[warning] unmapped regions: {dict(skipped_regions)}", flush=True)


if __name__ == "__main__":
    main()
