#!/usr/bin/env python3
"""Index the completed local experiment archive; audio hashes live in manifests."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EXTENSIONS = {".py", ".sh", ".json", ".jsonl", ".csv", ".tsv", ".md", ".tex", ".pdf",
              ".png", ".log", ".npz", ".npy", ".yaml", ".yml", ".toml", ".txt"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "EXPERIMENT_PROTOCOL.md").is_file():
        raise ValueError("Not the experiment artifact root")
    output_json = root / "audit/artifact_index.json"
    output_md = root / "ARTIFACT_INDEX.md"
    excluded = {output_json, output_md}
    candidates = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in {"qa", "__pycache__", "benchmark"} for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in EXTENSIONS and path.resolve() not in excluded:
            candidates.append(path.resolve())
    for path in args.extra:
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates.append(path.resolve())
    records = []
    for path in sorted(set(candidates)):
        try:
            reference = str(path.relative_to(root))
        except ValueError:
            reference = str(path)
        records.append({"reference": reference, "absolute_path": str(path),
                        "bytes": path.stat().st_size, "sha256": digest(path)})
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
               "indexed_files": len(records), "indexed_bytes": sum(r["bytes"] for r in records),
               "note": "Scientific archive only; QA previews and synthetic runtime benchmark outputs are excluded. "
                       "Audio is retained in its original local/remote locations and identified by item manifests.",
               "files": records}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Experiment artifact index", "", payload["note"], "",
             f"Generated: {payload['generated_utc']}. Indexed files: {len(records):,}.", "",
             "The full SHA-256 index is [artifact_index.json](" + str(output_json) + ").", "",
             "| Reference | Bytes | SHA-256 (prefix; full value in JSON) |", "|---|---:|---|"]
    for row in records:
        lines.append(f"| [{row['reference']}](<{row['absolute_path']}>) | {row['bytes']:,} | `{row['sha256'][:16]}` |")
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}))


if __name__ == "__main__":
    main()
