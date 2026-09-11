#!/usr/bin/env python3
"""Extract the existing raw and Demucs-bias-corrected vocal features only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import analyze_bias_corrected_vocals as corrected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--bias-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = corrected.read_jsonl(args.manifest.resolve())
    _, bias_db, selected_window = corrected.load_bias(args.bias_npz.resolve())
    metric_rows, vectors = corrected.extract(
        rows,
        args.clips_dir.resolve(),
        args.stems_dir.resolve(),
        bias_db,
        args.output_dir.resolve(),
        args.rebuild,
        args.workers,
    )
    raw_records = [row for row in metric_rows if row["variant"] == "raw"]
    activity_fields = (
        "id", "label", "class_name", "source", "split", "track",
        "raw_vocal_to_mix_rms_db", "raw_vocal_activity_frame_ratio",
        "passes_frozen_activity_threshold",
    )
    with (args.output_dir / "vocal_activity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=activity_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in activity_fields} for row in raw_records])
    metadata = {
        "manifest": str(args.manifest.resolve()),
        "tracks": len(rows),
        "frequency_vector_keys": sorted(vectors),
        "bias": str(args.bias_npz.resolve()),
        "selected_window_bins": selected_window,
        "activity_threshold_db": -18.0,
        "classifier_fitted": False,
        "external_labels_used_for_feature_extraction": False,
    }
    (args.output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
