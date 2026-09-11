#!/usr/bin/env python3
"""Read-only coverage/group summary of the accepted v5 package; no scores or fits."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

EXPECTED_COMMIT = "7cc03aa840c910df919e531153b61daba864e80794abc91ebf29d3859f70947a"
PREFIXES = {"S": "s8__", "D": "d__", "R": "r__", "P": "p__", "F": "F_", "H": "H_", "M": "M_"}
WIDTHS = {"S": 15, "D": 3, "R": 3, "P": 6, "F": 15, "H": 6, "M": 6}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    marker = root / "COMMIT.json"
    require(sha(marker) == EXPECTED_COMMIT, "Not the accepted real v5 input package")
    commit = json.loads(marker.read_text())
    for name in ("metadata_60s.csv", "features_60s.csv", "family_config.json"):
        proof = commit["files"][name]
        require(sha(root / name) == proof["sha256"] and (root / name).stat().st_size == proof["bytes"], name)
    with (root / "metadata_60s.csv").open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    with (root / "features_60s.csv").open(newline="") as stream:
        features = list(csv.DictReader(stream))
    require(len(metadata) == len(features) == 2207, "Wrong population")
    require(all(m["id"] == f["id"] for m, f in zip(metadata, features)), "ID order mismatch")
    require(len({m["id"] for m in metadata}) == 2207, "Duplicate identity")
    columns = {k: [c for c in features[0] if c.startswith(prefix)] for k, prefix in PREFIXES.items()}
    require({k: len(v) for k, v in columns.items()} == WIDTHS, "Family mapping changed")
    def finite(value):
        return math.isfinite(float(value)) if value.strip() else False
    summaries = []
    for source in sorted({m["source_group"] for m in metadata}):
        rows = [(m, f) for m, f in zip(metadata, features) if m["source_group"] == source]
        groups = collections.Counter(m["group_id"] for m, _ in rows)
        observed = {k: [sum(finite(f[c]) for c in cs) for _, f in rows] for k, cs in columns.items()}
        summaries.append({
            "source": source, "rows": len(rows), "recorded_global_groups": len(groups),
            "largest_group_rows": max(groups.values()),
            "inverse_group_concentration_not_inferential_ess": len(rows) ** 2 / sum(n ** 2 for n in groups.values()),
            "complete_family_rows": {k: sum(n == WIDTHS[k] for n in counts) for k, counts in observed.items()},
            "partial_family_rows": {k: sum(0 < n < WIDTHS[k] for n in counts) for k, counts in observed.items()},
        })
    return {"package_commit_sha256": EXPECTED_COMMIT, "rows": 2207,
            "recorded_global_groups": len({m["group_id"] for m in metadata}),
            "scope": "Input coverage only; no scoring, causal claim, independent-artist claim or inferred effective sample size.",
            "sources": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.package_dir), indent=2, sort_keys=True))
