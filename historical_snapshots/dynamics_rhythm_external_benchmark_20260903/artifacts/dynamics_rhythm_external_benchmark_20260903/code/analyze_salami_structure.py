#!/usr/bin/env python3
"""Score All-In-One boundaries and section-length summaries on SALAMI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SEED = 20260903
N_BOOT = 10_000
SECOND_FEATURES = ("section_duration_median", "section_duration_cv", "section_duration_entropy")
BAR_FEATURES = ("section_bars_median", "section_bars_cv", "section_bars_offmode_fraction")


def load_annotation(path: Path) -> np.ndarray:
    times = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                times.append(float(line.split("\t", 1)[0]))
            except ValueError:
                continue
    values = np.unique(np.asarray(times, dtype=float))
    return values[np.isfinite(values)]


def normalize_boundaries(times: np.ndarray, duration: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    times = times[(times >= 0.0) & (times <= duration + 1.0)]
    times = np.clip(times, 0.0, duration)
    times = np.unique(np.concatenate(([0.0], times, [duration])))
    return times


def intervals(boundaries: np.ndarray) -> np.ndarray:
    return np.column_stack((boundaries[:-1], boundaries[1:]))


def duration_features(boundaries: np.ndarray) -> dict[str, float]:
    durations = np.diff(boundaries)
    durations = durations[durations > 0.0]
    result = {name: float("nan") for name in SECOND_FEATURES}
    if len(durations) < 3:
        return result
    median = np.median(durations)
    q25, q75 = np.quantile(durations, [0.25, 0.75])
    bins = np.arange(0.0, max(62.0, np.ceil(durations.max() / 2.0) * 2.0 + 2.0), 2.0)
    counts, _ = np.histogram(durations, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    result["section_duration_median"] = float(median)
    result["section_duration_cv"] = float((q75 - q25) / median) if median > 0 else float("nan")
    result["section_duration_entropy"] = float(
        -(probabilities * np.log(probabilities)).sum() / np.log(max(len(counts), 2))
    )
    return result


def bar_features(boundaries: np.ndarray, downbeats: np.ndarray) -> dict[str, float]:
    result = {name: float("nan") for name in BAR_FEATURES}
    if len(downbeats) < 4 or len(boundaries) < 4:
        return result
    snapped_indices = np.asarray(
        [int(np.argmin(np.abs(downbeats - boundary))) for boundary in boundaries], dtype=int
    )
    snapped_indices = np.unique(snapped_indices)
    bar_lengths = np.diff(snapped_indices)
    bar_lengths = bar_lengths[bar_lengths > 0]
    if len(bar_lengths) < 3:
        return result
    median = np.median(bar_lengths)
    q25, q75 = np.quantile(bar_lengths, [0.25, 0.75])
    values, counts = np.unique(bar_lengths, return_counts=True)
    mode = values[np.flatnonzero(counts == counts.max())[0]]
    result["section_bars_median"] = float(median)
    result["section_bars_cv"] = float((q75 - q25) / median) if median > 0 else float("nan")
    result["section_bars_offmode_fraction"] = float(np.mean(bar_lengths != mode))
    return result


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def bootstrap_mean(table: pd.DataFrame, column: str, rng: np.random.Generator) -> dict[str, float | int]:
    groups = {key: group[column].dropna().to_numpy(float) for key, group in table.groupby("song_id")}
    groups = {key: values for key, values in groups.items() if len(values)}
    song_ids = np.array(sorted(groups))
    point = float(np.mean(np.concatenate(list(groups.values()))))
    draws = np.empty(N_BOOT, dtype=float)
    for index in range(N_BOOT):
        sampled = rng.choice(song_ids, len(song_ids), replace=True)
        draws[index] = np.mean(np.concatenate([groups[key] for key in sampled]))
    return {
        "estimate": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_tracks": len(song_ids),
    }


def bootstrap_correlation(
    table: pd.DataFrame, reference_column: str, prediction_column: str, rng: np.random.Generator
) -> dict[str, float | int]:
    clean = table[["song_id", reference_column, prediction_column]].dropna()
    point, pvalue = safe_spearman(
        clean[reference_column].to_numpy(float), clean[prediction_column].to_numpy(float)
    )
    groups = {key: group for key, group in clean.groupby("song_id")}
    song_ids = np.array(sorted(groups))
    draws = []
    for _ in range(N_BOOT):
        sampled = rng.choice(song_ids, len(song_ids), replace=True)
        reference = np.concatenate([groups[key][reference_column].to_numpy(float) for key in sampled])
        prediction = np.concatenate([groups[key][prediction_column].to_numpy(float) for key in sampled])
        rho, _ = safe_spearman(reference, prediction)
        if np.isfinite(rho):
            draws.append(rho)
    draws = np.asarray(draws, dtype=float)
    return {
        "estimate": point,
        "pvalue": pvalue,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sign_stability": float(np.mean(draws > 0) if point >= 0 else np.mean(draws < 0)),
        "n_tracks": len(song_ids),
        "n_annotator_rows": len(clean),
        "n_boot_valid": len(draws),
    }


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        result[order] = np.minimum(ranked, 1.0)
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--allinone-dir", type=Path, required=True)
    parser.add_argument("--beat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.selected_manifest.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))

    rows = []
    missing = []
    for item in selected:
        song_id = item["song_id"]
        structure_path = args.allinone_dir / f"{song_id}.json"
        beat_path = args.beat_dir / f"{song_id}.beats"
        if not structure_path.exists() or not beat_path.exists():
            missing.append(
                {"song_id": song_id, "structure_exists": structure_path.exists(), "beat_exists": beat_path.exists()}
            )
            continue
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        segments = structure["segments"]
        predicted_times = np.asarray(
            [float(segment["start"]) for segment in segments] + [float(segments[-1]["end"])],
            dtype=float,
        )
        beat_data = np.loadtxt(beat_path, ndmin=2)
        downbeats = beat_data[beat_data[:, 1].astype(int) == 1, 0]
        annotation_paths = sorted(
            (args.annotation_root / song_id / "parsed").glob("textfile*_uppercase.txt")
        )
        for annotation_path in annotation_paths:
            reference_raw = load_annotation(annotation_path)
            if len(reference_raw) < 2:
                continue
            duration = min(float(reference_raw[-1]), float(predicted_times[-1]))
            reference = normalize_boundaries(reference_raw, duration)
            prediction = normalize_boundaries(predicted_times, duration)
            ref_intervals = intervals(reference)
            pred_intervals = intervals(prediction)
            p05, r05, f05 = mir_eval.segment.detection(
                ref_intervals, pred_intervals, window=0.5, trim=True
            )
            p30, r30, f30 = mir_eval.segment.detection(
                ref_intervals, pred_intervals, window=3.0, trim=True
            )
            row: dict[str, object] = {
                "song_id": song_id,
                "annotator": annotation_path.stem.split("_")[0],
                "duration_seconds": duration,
                "n_reference_sections": len(ref_intervals),
                "n_predicted_sections": len(pred_intervals),
                "precision_0p5s": p05,
                "recall_0p5s": r05,
                "f1_0p5s": f05,
                "precision_3s": p30,
                "recall_3s": r30,
                "f1_3s": f30,
            }
            for prefix, features in (
                ("ref", duration_features(reference)),
                ("pred", duration_features(prediction)),
                ("ref", bar_features(reference, downbeats)),
                ("pred", bar_features(prediction, downbeats)),
            ):
                row.update({f"{prefix}_{key}": value for key, value in features.items()})
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["song_id", "annotator"])
    table.to_csv(args.output_dir / "per_annotator_metrics.csv", index=False)
    rng = np.random.default_rng(SEED)
    boundary_metrics = {
        column: bootstrap_mean(table, column, rng)
        for column in (
            "precision_0p5s",
            "recall_0p5s",
            "f1_0p5s",
            "precision_3s",
            "recall_3s",
            "f1_3s",
        )
    }
    boundary_admitted = bool(
        boundary_metrics["f1_3s"]["estimate"] >= 0.60
        and boundary_metrics["f1_3s"]["ci_low"] >= 0.55
    )

    fidelity_rows = []
    for feature in SECOND_FEATURES + BAR_FEATURES:
        stats = bootstrap_correlation(table, f"ref_{feature}", f"pred_{feature}", rng)
        fidelity_rows.append(
            {
                "feature": feature,
                **stats,
                "admitted": bool(
                    boundary_admitted
                    and stats["estimate"] >= 0.70
                    and stats["ci_low"] >= 0.55
                    and stats["sign_stability"] >= 0.90
                ),
            }
        )
    for row, qvalue in zip(fidelity_rows, bh_adjust([row["pvalue"] for row in fidelity_rows])):
        row["qvalue"] = qvalue
    pd.DataFrame(fidelity_rows).to_csv(args.output_dir / "feature_fidelity.csv", index=False)
    summary = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_selected_tracks": len(selected),
        "n_scored_tracks": int(table["song_id"].nunique()),
        "n_annotator_rows": len(table),
        "missing": missing,
        "boundary_metrics": boundary_metrics,
        "boundary_admitted": boundary_admitted,
        "feature_fidelity": fidelity_rows,
        "admitted_features": [row["feature"] for row in fidelity_rows if row["admitted"]],
        "functional_labels_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

