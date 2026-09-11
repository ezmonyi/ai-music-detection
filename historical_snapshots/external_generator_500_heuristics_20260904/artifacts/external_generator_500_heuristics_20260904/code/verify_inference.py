#!/usr/bin/env python3
"""Audit per-track Beat This!, All-In-One, and retained Demucs outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


STEMS = ("bass", "drums", "other", "vocals")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7100:
        raise RuntimeError(f"Expected 7100 rows, got {len(rows)}")

    states: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, object]] = []
    for row in rows:
        track = row["track"]
        model = row["model"]
        beat = args.inference_root / "beats" / f"{track}.beats"
        structure = args.inference_root / "structure" / f"{track}.json"
        spectrogram = args.inference_root / "spec" / f"{track}.npy"
        stem_dir = args.inference_root / "demix" / "htdemucs" / track
        stem_paths = [stem_dir / f"{stem}.wav" for stem in STEMS]

        beat_state = "nonempty" if beat.exists() and beat.stat().st_size else "empty" if beat.exists() else "missing"
        structure_state = "present" if structure.exists() and structure.stat().st_size else "missing"
        spectrogram_state = "present" if spectrogram.exists() and spectrogram.stat().st_size else "missing"
        n_stems = sum(path.exists() and path.stat().st_size > 0 for path in stem_paths)
        stems_state = "complete" if n_stems == 4 else "missing" if n_stems == 0 else "partial"
        states[model].update({
            f"beat_{beat_state}": 1,
            f"structure_{structure_state}": 1,
            f"spectrogram_{spectrogram_state}": 1,
            f"stems_{stems_state}": 1,
        })
        if beat_state == "missing" or structure_state == "missing" or spectrogram_state == "missing" or stems_state != "complete":
            failures.append({
                "track": track,
                "model": model,
                "split": row["split"],
                "beat": beat_state,
                "structure": structure_state,
                "spectrogram": spectrogram_state,
                "stems": stems_state,
                "n_stems": n_stems,
            })

    payload = {
        "expected_tracks": len(rows),
        "per_model": {model: dict(sorted(counts.items())) for model, counts in sorted(states.items())},
        "tracks_with_any_missing_output": len(failures),
        "output_failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "output_failures"}, indent=2))


if __name__ == "__main__":
    main()
