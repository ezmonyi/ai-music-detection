#!/usr/bin/env python3
"""Quantify controlled-change sensitivity of the three frozen workflows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow_features import (
    DYNAMICS_FEATURES,
    RHYTHM_FEATURES,
    STRUCTURE_FEATURES,
    allinone_boundaries,
    dynamics_features,
    load_beats,
    read_audio,
    rhythm_features,
    structure_features,
)


SEED = 20260903
N_BOOT = 10_000


def paired_stats(delta: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    auc = float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))
    draws = np.empty(N_BOOT)
    for index in range(N_BOOT):
        sample = rng.choice(delta, len(delta), replace=True)
        draws[index] = np.mean(sample > 0) + 0.5 * np.mean(sample == 0)
    return {
        "n": int(len(delta)),
        "median_delta": float(np.median(delta)),
        "probability_superiority": auc,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def paired_table(table: pd.DataFrame, condition_a: str, condition_b: str, features: tuple[str, ...]) -> pd.DataFrame:
    left = table[table.condition == condition_a].set_index("item_id")
    right = table[table.condition == condition_b].set_index("item_id")
    common = left.index.intersection(right.index)
    rows = []
    for item_id in common:
        row = {"item_id": item_id}
        for feature in features:
            row[f"a_{feature}"] = left.loc[item_id, feature]
            row[f"b_{feature}"] = right.loc[item_id, feature]
        rows.append(row)
    return pd.DataFrame(rows)


def composite_delta(pairs: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    deltas = []
    for feature in features:
        a = pairs[f"a_{feature}"].to_numpy(float)
        b = pairs[f"b_{feature}"].to_numpy(float)
        finite_a = a[np.isfinite(a)]
        center = float(np.median(finite_a)) if len(finite_a) else 0.0
        q25, q75 = np.quantile(finite_a, [0.25, 0.75]) if len(finite_a) else (0.0, 1.0)
        scale = max(float(q75 - q25), 1e-8)
        a = np.where(np.isfinite(a), a, center)
        b = np.where(np.isfinite(b), b, center)
        deltas.append((b - a) / scale)
    return np.mean(np.stack(deltas), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--beat-dir", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    rows = []
    for entry in manifest:
        family = entry["family"]
        name = entry["filename"]
        base = {"family": family, "item_id": entry["item_id"], "condition": entry["condition"], "filename": name}
        if family == "dynamics":
            features = dynamics_features(read_audio(args.audio_root / family / name))
        else:
            beat_path = args.beat_dir / f"{Path(name).stem}.beats"
            if beat_path.exists():
                times, numbers = load_beats(beat_path)
            else:
                times, numbers = np.asarray([], dtype=float), np.asarray([], dtype=int)
            if family == "rhythm":
                features = rhythm_features(times)
            else:
                boundaries = allinone_boundaries(args.structure_dir / f"{Path(name).stem}.json")
                features = structure_features(boundaries, times[numbers == 1])
        rows.append({**base, **features})
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "controlled_change_features.csv", index=False)

    comparisons = [
        ("dynamics", "original", "variable_gain", DYNAMICS_FEATURES, "positive"),
        ("dynamics", "original", "constant_gain", DYNAMICS_FEATURES, "negative_control"),
        ("rhythm", "original", "variable_tempo", RHYTHM_FEATURES, "positive"),
        ("rhythm", "original", "constant_tempo", RHYTHM_FEATURES, "negative_control"),
        ("structure", "equal_sections", "variable_sections", STRUCTURE_FEATURES, "positive"),
    ]
    rng = np.random.default_rng(SEED)
    results = []
    family_summary = {}
    for family, condition_a, condition_b, features, role in comparisons:
        pairs = paired_table(table[table.family == family], condition_a, condition_b, features)
        for feature in features:
            delta = pairs[f"b_{feature}"].to_numpy(float) - pairs[f"a_{feature}"].to_numpy(float)
            results.append({
                "family": family, "comparison": f"{condition_b}_minus_{condition_a}",
                "role": role, "feature": feature, **paired_stats(delta, rng),
            })
        composite = paired_stats(composite_delta(pairs, features), rng)
        results.append({
            "family": family, "comparison": f"{condition_b}_minus_{condition_a}",
            "role": role, "feature": "family_composite", **composite,
        })
        if role == "positive":
            family_summary[family] = {
                **composite,
                "sensitivity_pass": bool(
                    composite["probability_superiority"] >= 0.75 and composite["ci_low"] > 0.60
                ),
            }
    result_table = pd.DataFrame(results)
    result_table.to_csv(args.output_dir / "controlled_change_sensitivity.csv", index=False)
    summary = {
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "family_results": family_summary,
        "all_families_pass": bool(all(row["sensitivity_pass"] for row in family_summary.values())),
    }
    (args.output_dir / "controlled_change_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
