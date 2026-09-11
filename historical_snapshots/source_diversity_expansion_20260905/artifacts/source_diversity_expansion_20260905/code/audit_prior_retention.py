#!/usr/bin/env python3
"""Read-only existence/size inventory of older natives and retained AIME containers.

This is not a fresh content rehash. Historical content hashes remain in the
original manifests and pinned source audit.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, action="append", required=True)
    parser.add_argument("--aime-manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=Path, required=True)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    issues, model_files, shard_files = [], [], []
    for manifest in args.model_manifest:
        for row in jsonl(manifest):
            path = Path(row["source"])
            size = path.stat().st_size if path.is_file() else 0
            if not size or row.get("status") != "ok":
                issues.append({"path": str(path), "issue": "missing_or_invalid_model_native"})
            model_files.append({"generator": row["generator"], "id": row["id"], "path": str(path),
                                "current_bytes": size, "historical_sha256": row["source_sha256"]})
    selected = jsonl(args.aime_manifest)
    required = sorted({row["source_parquet"] for row in selected})
    with args.shard_index.open(newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    index = {}
    for row in index_rows:
        key = row["shard"]
        if key in index and any(row[field] != index[key][field] for field in ("bytes", "sha256")):
            raise ValueError("Inconsistent shard index: " + key)
        index[key] = row
    for name in required:
        path = args.parquet_root / name
        size = path.stat().st_size if path.is_file() else 0
        reference = index.get(name, {})
        if not size or size != int(reference.get("bytes", -1)):
            issues.append({"path": str(path), "issue": "missing_or_size_mismatched_container"})
        shard_files.append({"path": str(path), "current_bytes": size,
                            "historical_sha256": reference.get("sha256", "")})
    inputs = [*args.model_manifest, args.aime_manifest, args.shard_index]
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not issues else "failed", "issues": issues,
        "scope": "Current file existence/size only; no fresh raw-audio or parquet content rehash.",
        "model_native_files": len(model_files),
        "model_native_bytes": sum(row["current_bytes"] for row in model_files),
        "selected_aime_recordings": len(selected), "required_retained_parquet_files": len(shard_files),
        "required_retained_parquet_bytes": sum(row["current_bytes"] for row in shard_files),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "native_records": model_files, "parquet_records": shard_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"native_records", "parquet_records"}}, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
