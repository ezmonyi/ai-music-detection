#!/usr/bin/env python3
"""Audit every expected Beat This, All-In-One, spectrogram and stem output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


STEMS = ("bass", "drums", "other", "vocals")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hashes", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("item_id") or row.get("track") or row.get("id") or "" for row in rows]
    if not rows or "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("Manifest must have unique item ids")

    def find(relative: Path, allow_empty: bool = False) -> Path | None:
        for root in args.inference_root:
            candidate = root / relative
            if candidate.is_file() and (allow_empty or candidate.stat().st_size):
                return candidate
        return None

    states: dict[str, Counter[str]] = defaultdict(Counter)
    details = []
    for row, item_id in zip(rows, ids):
        source = row.get("source_id") or row.get("source") or "unknown"
        beat = find(Path("beats") / f"{item_id}.beats", allow_empty=True)
        structure = find(Path("structure") / f"{item_id}.json")
        spec = find(Path("spec") / f"{item_id}.npy")
        stems = {stem: find(Path("demix") / "htdemucs" / item_id / f"{stem}.wav") for stem in STEMS}
        beat_state = "missing" if beat is None else "empty" if beat.stat().st_size == 0 else "nonempty"
        stem_state = "complete" if all(stems.values()) else "missing" if not any(stems.values()) else "partial"
        state = {
            "beat": beat_state,
            "structure": "present" if structure else "missing",
            "spectrogram": "present" if spec else "missing",
            "stems": stem_state,
        }
        states[source].update(f"{key}_{value}" for key, value in state.items())
        missing_required = structure is None or spec is None or stem_state != "complete" or beat is None
        item = {"item_id": item_id, "source_id": source, **state, "missing_required_output": int(missing_required)}
        if args.hashes:
            paths = {"beat": beat, "structure": structure, "spectrogram": spec, **stems}
            item["sha256"] = {key: sha256(path) for key, path in paths.items() if path is not None}
        details.append(item)
    failures = [row for row in details if row["missing_required_output"]]
    payload = {
        "expected": len(rows), "fully_present": len(rows)-len(failures),
        "missing_required": len(failures),
        "note": "zero-byte .beats is a valid empty detection, distinct from a missing output",
        "per_source": {key: dict(sorted(value.items())) for key, value in sorted(states.items())},
        "details": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, indent=2))
    if args.strict and failures:
        raise RuntimeError(f"{len(failures)} rows missing required outputs; first={failures[0]['item_id']}")


if __name__ == "__main__":
    main()
