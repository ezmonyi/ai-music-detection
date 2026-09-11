#!/usr/bin/env python3
"""Aggregate the three frozen Beat This seeds and apply protocol gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "f1_beat",
    "f1_downbeat",
    "cmlt_beat",
    "amlt_beat",
    "cmlt_downbeat",
    "amlt_downbeat",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    rows = []
    for seed in range(3):
        path = args.results_root / f"gtzan_final{seed}_analysis" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(summary)
        row: dict[str, object] = {"model": f"final{seed}"}
        for metric in METRICS:
            row[metric] = summary["bootstrap_metrics"][metric]["estimate"]
        rows.append(row)

    with (args.output_dir / "rhythm_seed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    cross_seed = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        cross_seed[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }

    final0 = summaries[0]
    timing_admission = {
        "beat": bool(
            cross_seed["f1_beat"]["mean"] >= 0.85
            and final0["bootstrap_metrics"]["f1_beat"]["ci_low"] >= 0.83
        ),
        "downbeat": bool(
            cross_seed["f1_downbeat"]["mean"] >= 0.72
            and final0["bootstrap_metrics"]["f1_downbeat"]["ci_low"] >= 0.68
        ),
    }
    feature_rows = []
    for feature in final0["feature_correlations"]:
        row = dict(feature)
        family_gate = timing_admission[
            "beat" if row["feature"] in {"ibi_cv", "tempo_tv", "tempo_entropy"} else "downbeat"
        ]
        row["admitted"] = bool(
            family_gate
            and float(row["estimate"]) >= 0.75
            and float(row["ci_low"]) >= 0.65
            and float(row["sign_stability"]) >= 0.90
        )
        feature_rows.append(row)

    summary = {
        "models": [row["model"] for row in rows],
        "n_tracks_per_model": [item["n_scored_tracks"] for item in summaries],
        "cross_seed_metrics": cross_seed,
        "final0_bootstrap_metrics": final0["bootstrap_metrics"],
        "timing_admission": timing_admission,
        "final0_feature_fidelity": feature_rows,
        "admitted_detector_features": [
            row["feature"] for row in feature_rows if row["admitted"]
        ],
        "feature_exclusions": final0.get("feature_exclusions", {}),
        "missing_prediction_details": final0.get("missing_prediction_details", []),
    }
    (args.output_dir / "rhythm_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
