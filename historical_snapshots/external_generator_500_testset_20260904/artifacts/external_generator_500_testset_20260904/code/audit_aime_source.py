#!/usr/bin/env python3
"""Hash downloaded AIME shards and audit per-model row counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.parquet_dir.glob("*.parquet"))
    if len(paths) != 202:
        raise RuntimeError(f"expected 202 retained shards, found {len(paths)}")

    rows = []
    total_counts: Counter[str] = Counter()
    hash_lines = []
    for index, path in enumerate(paths, 1):
        table = pq.read_table(path, columns=["model"])
        counts = Counter(str(value) for value in table.column("model").to_pylist())
        total_counts.update(counts)
        digest = sha256(path)
        hash_lines.append(f"{digest}  {path.name}\n")
        for model, count in sorted(counts.items()):
            rows.append(
                {
                    "shard": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "model": model,
                    "rows": count,
                }
            )
        if index % 10 == 0 or index == len(paths):
            print(f"audited {index}/{len(paths)} shards", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "shard_model_index.csv"
    temporary = csv_path.with_name(csv_path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    atomic_text(args.output_dir / "source_parquet_sha256.txt", "".join(hash_lines))
    summary = {
        "shards": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "model_counts_in_retained_shards": dict(sorted(total_counts.items())),
        "first_shard": paths[0].name,
        "last_shard": paths[-1].name,
    }
    atomic_text(
        args.output_dir / "source_audit_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
