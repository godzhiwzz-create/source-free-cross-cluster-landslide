#!/usr/bin/env python3
"""Create a support-excluded query-index manifest from an adaptation JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from landslide_sfda.constants import CLUSTERS
from landslide_sfda.data import input_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--held", required=True, choices=CLUSTERS)
    parser.add_argument("--support-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.support_result.read_text())
    if source.get("held") != args.held:
        raise ValueError("held cluster differs between arguments and support result")
    raw_support = [int(index) for index in source["support_indices"]]
    support = set(raw_support)
    all_indices = input_indices(args.data_root, args.held)
    if len(support) != len(raw_support) or not support.issubset(all_indices):
        raise ValueError("support indices are duplicated or outside the target pool")
    query = [index for index in all_indices if index not in support]
    encoded = "\n".join(f"{args.held}\t{index}" for index in query).encode()
    payload = {
        "held": args.held,
        "indices": query,
        "n_query": len(query),
        "n_support": len(support),
        "identity_sha256": hashlib.sha256(encoded).hexdigest(),
        "rule": "all target indices in canonical order minus support_indices",
        "support_result": str(args.support_result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
