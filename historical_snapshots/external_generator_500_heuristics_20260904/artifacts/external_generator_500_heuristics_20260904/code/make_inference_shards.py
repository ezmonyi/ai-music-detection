#!/usr/bin/env python3
"""Freeze balanced, disjoint inference shards from the combined manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, default=6)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[str]] = [[] for _ in range(args.n_shards)]
    # Round-robin separately inside each domain so every long-running process
    # receives approximately the same mixture and duration.
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    offset = 0
    for source in sorted(by_source):
        source_rows = sorted(by_source[source], key=lambda row: row["track"])
        for index, row in enumerate(source_rows):
            shards[(offset + index) % args.n_shards].append(str(args.audio_dir / row["audio_filename"]))
        offset = (offset + len(source_rows)) % args.n_shards
    metadata = {"n_shards": args.n_shards, "n_rows": len(rows), "shards": []}
    observed: list[str] = []
    for index, paths in enumerate(shards):
        output = args.output_dir / f"shard_{index:02d}.txt"
        body = "\n".join(paths) + "\n"
        output.write_text(body, encoding="utf-8")
        metadata["shards"].append({
            "index": index,
            "count": len(paths),
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
        })
        observed.extend(paths)
    if len(observed) != len(set(observed)) or Counter(observed) != Counter(
        str(args.audio_dir / row["audio_filename"]) for row in rows
    ):
        raise RuntimeError("Shards are not an exact disjoint partition of the manifest")
    (args.output_dir / "shard_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
