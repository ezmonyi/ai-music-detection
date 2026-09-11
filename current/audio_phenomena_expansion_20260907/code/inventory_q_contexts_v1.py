#!/usr/bin/env python3
"""Read-only metadata feasibility inventory; no audio or classifier processing."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def inspect(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError("metadata changed during read")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate recording ID")
    dev = [r for r in rows if r["role"] == "development"]
    mapping = defaultdict(set)
    family_rows = Counter()
    for row in dev:
        if row["label"] == "1":
            family = row.get("generator_family", "")
            mapping[row["source_group"]].add(family)
            family_rows[family] += 1
    result = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
              "rows": len(rows), "roles": dict(Counter(r["role"] for r in rows)),
              "development_labels": dict(Counter(r["label"] for r in dev)),
              "development_sources": dict(Counter(r["source_group"] for r in dev)),
              "source_to_declared_generator_family": {k: sorted(v) for k, v in sorted(mapping.items())},
              "declared_generator_family_rows": dict(family_rows),
              "development_group_count": len({r["group_id"] for r in dev}),
              "audio_bytes_revalidated": False, "features_extracted": False, "classifier_fits": 0}
    for field in ("available", "eligible_common8", "eligible_fullband", "evaluation_allowed", "duration_view"):
        if rows and field in rows[0]:
            result[field + "_development_values"] = dict(Counter(r[field] for r in dev))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect(args.metadata)
    result["inventory_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
