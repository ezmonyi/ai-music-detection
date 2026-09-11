#!/usr/bin/env python3
"""Score Beat This! predictions and detector-facing rhythm summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SEED = 20260903
N_BOOT = 10_000
RHYTHM_FEATURES = ("ibi_cv", "tempo_tv", "tempo_entropy")
BAR_FEATURES = ("bar_offmode_fraction", "bar_length_entropy")


def canonical_piece(key: str) -> str:
    path = Path(key)
    if path.name == "track.npy":
        return path.parent.name
    return path.stem


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x[keep], y[keep])
    return float(result.statistic), float(result.pvalue)


def event_features(times: np.ndarray, beat_numbers: np.ndarray) -> dict[str, float]:
    times = np.asarray(times, dtype=float)
    beat_numbers = np.asarray(beat_numbers, dtype=int)
    result = {name: float("nan") for name in RHYTHM_FEATURES + BAR_FEATURES}
    if len(times) >= 16:
        ibi = np.diff(times)
        ibi = ibi[(ibi > 0) & np.isfinite(ibi)]
        if len(ibi) >= 15 and np.median(ibi) > 0:
            q25, q75 = np.quantile(ibi, [0.25, 0.75])
            result["ibi_cv"] = float((q75 - q25) / np.median(ibi))
            bpm = 60.0 / ibi
            bpm = bpm[(bpm >= 30.0) & (bpm <= 300.0)]
            if len(bpm) >= 14 and np.median(bpm) > 0:
                result["tempo_tv"] = float(
                    np.median(np.abs(np.diff(bpm))) / np.median(bpm)
                )
                counts, _ = np.histogram(bpm, bins=np.arange(30.0, 305.0, 5.0))
                probs = counts[counts > 0] / counts.sum()
                result["tempo_entropy"] = float(
                    -(probs * np.log(probs)).sum() / np.log(len(counts))
                )

    downbeat_indices = np.flatnonzero(beat_numbers == 1)
    if len(downbeat_indices) >= 9:
        bar_counts = np.diff(downbeat_indices)
        bar_counts = bar_counts[(bar_counts >= 1) & (bar_counts <= 12)]
        if len(bar_counts) >= 8:
            values, counts = np.unique(bar_counts, return_counts=True)
            mode_count = values[np.flatnonzero(counts == counts.max())[0]]
            result["bar_offmode_fraction"] = float(np.mean(bar_counts != mode_count))
            probs = counts / counts.sum()
            result["bar_length_entropy"] = float(
                -(probs * np.log(probs)).sum() / np.log(12.0)
            )
    return result


def bar_feature_eligible(beat_numbers: np.ndarray) -> bool:
    downbeat_indices = np.flatnonzero(np.asarray(beat_numbers, dtype=int) == 1)
    if len(downbeat_indices) < 9:
        return False
    complete_bar_counts = np.diff(downbeat_indices)
    complete_bar_counts = complete_bar_counts[
        (complete_bar_counts >= 1) & (complete_bar_counts <= 12)
    ]
    return bool(len(complete_bar_counts) >= 8)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = values[np.isfinite(values)]
    draws = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        draws[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n": int(len(values)),
    }


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    estimate, pvalue = safe_spearman(x, y)
    draws = []
    if len(x) >= 3:
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(x), len(x))
            rho, _ = safe_spearman(x[idx], y[idx])
            if np.isfinite(rho):
                draws.append(rho)
    draws_arr = np.asarray(draws, dtype=float)
    if len(draws_arr):
        ci_low, ci_high = np.quantile(draws_arr, [0.025, 0.975])
        sign_stability = np.mean(draws_arr > 0) if estimate >= 0 else np.mean(draws_arr < 0)
    else:
        ci_low = ci_high = sign_stability = float("nan")
    return {
        "estimate": estimate,
        "pvalue": pvalue,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "sign_stability": float(sign_stability),
        "n": int(len(x)),
        "n_boot_valid": int(len(draws_arr)),
    }


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-trim-seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_npz = np.load(args.predictions, allow_pickle=False)
    pred_by_piece = {canonical_piece(key): pred_npz[key] for key in pred_npz.files}
    rows: list[dict[str, float | str | int]] = []
    missing_predictions: list[str] = []
    missing_prediction_details: list[dict[str, str | int]] = []
    rng = np.random.default_rng(SEED)

    for annotation_path in sorted(args.annotations.glob("*.beats")):
        piece = annotation_path.stem
        if piece not in pred_by_piece:
            missing_predictions.append(piece)
            annotation = np.loadtxt(annotation_path, ndmin=2)
            missing_prediction_details.append(
                {
                    "piece": piece,
                    "annotation_columns": int(annotation.shape[1]),
                    "reason": (
                        "annotation_has_no_downbeat_column"
                        if annotation.shape[1] < 2
                        else "prediction_archive_missing"
                    ),
                }
            )
            continue
        reference = np.loadtxt(annotation_path, ndmin=2)
        prediction = np.asarray(pred_by_piece[piece])
        ref_times = reference[:, 0]
        ref_numbers = reference[:, 1].astype(int)
        pred_times = prediction[:, 0]
        pred_numbers = prediction[:, 1].astype(int)
        ref_keep = ref_times >= args.eval_trim_seconds
        pred_keep = pred_times >= args.eval_trim_seconds
        ref_times, ref_numbers = ref_times[ref_keep], ref_numbers[ref_keep]
        pred_times, pred_numbers = pred_times[pred_keep], pred_numbers[pred_keep]
        ref_downbeats = ref_times[ref_numbers == 1]
        pred_downbeats = pred_times[pred_numbers == 1]

        row: dict[str, float | str | int] = {
            "piece": piece,
            "genre": piece.split("_")[1],
            "n_ref_beats": len(ref_times),
            "n_pred_beats": len(pred_times),
            "n_ref_downbeats": len(ref_downbeats),
            "n_pred_downbeats": len(pred_downbeats),
            "ref_rhythm_eligible": len(ref_times) >= 16,
            "pred_rhythm_eligible": len(pred_times) >= 16,
            "ref_bar_eligible": bar_feature_eligible(ref_numbers),
            "pred_bar_eligible": bar_feature_eligible(pred_numbers),
            "f1_beat": mir_eval.beat.f_measure(ref_times, pred_times),
            "f1_downbeat": mir_eval.beat.f_measure(ref_downbeats, pred_downbeats),
        }
        _, row["cmlt_beat"], _, row["amlt_beat"] = mir_eval.beat.continuity(
            ref_times, pred_times
        )
        _, row["cmlt_downbeat"], _, row["amlt_downbeat"] = mir_eval.beat.continuity(
            ref_downbeats, pred_downbeats
        )
        for prefix, features in (
            ("ref", event_features(ref_times, ref_numbers)),
            ("pred", event_features(pred_times, pred_numbers)),
        ):
            row.update({f"{prefix}_{key}": value for key, value in features.items()})
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("piece")
    table.to_csv(args.output_dir / "per_track_metrics.csv", index=False)

    bootstrap_metrics = {}
    for metric in (
        "f1_beat",
        "f1_downbeat",
        "cmlt_beat",
        "amlt_beat",
        "cmlt_downbeat",
        "amlt_downbeat",
    ):
        bootstrap_metrics[metric] = bootstrap_mean(table[metric].to_numpy(float), rng)

    correlations = []
    for feature in RHYTHM_FEATURES + BAR_FEATURES:
        stats = bootstrap_spearman(
            table[f"ref_{feature}"].to_numpy(float),
            table[f"pred_{feature}"].to_numpy(float),
            rng,
        )
        errors = np.abs(
            table[f"ref_{feature}"].to_numpy(float)
            - table[f"pred_{feature}"].to_numpy(float)
        )
        stats["median_absolute_error"] = float(np.nanmedian(errors))
        stats["feature"] = feature
        correlations.append(stats)
    qvalues = bh_adjust([row["pvalue"] for row in correlations])
    for row, qvalue in zip(correlations, qvalues):
        row["qvalue"] = qvalue
        row["admitted"] = bool(
            np.isfinite(row["estimate"])
            and row["estimate"] >= 0.75
            and row["ci_low"] >= 0.65
            and row["sign_stability"] >= 0.90
        )

    timing_admission = {
        "beat": bool(
            bootstrap_metrics["f1_beat"]["estimate"] >= 0.85
            and bootstrap_metrics["f1_beat"]["ci_low"] >= 0.83
        ),
        "downbeat": bool(
            bootstrap_metrics["f1_downbeat"]["estimate"] >= 0.72
            and bootstrap_metrics["f1_downbeat"]["ci_low"] >= 0.68
        ),
    }
    for row in correlations:
        if row["feature"] in RHYTHM_FEATURES:
            row["admitted"] = bool(row["admitted"] and timing_admission["beat"])
        else:
            row["admitted"] = bool(row["admitted"] and timing_admission["downbeat"])

    summary = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "eval_trim_seconds": args.eval_trim_seconds,
        "n_annotation_files": len(list(args.annotations.glob("*.beats"))),
        "n_scored_tracks": len(table),
        "missing_predictions": missing_predictions,
        "missing_prediction_details": missing_prediction_details,
        "feature_exclusions": {
            "rhythm_reference_fewer_than_16_beats": int((~table["ref_rhythm_eligible"]).sum()),
            "rhythm_prediction_fewer_than_16_beats": int((~table["pred_rhythm_eligible"]).sum()),
            "rhythm_pair_excluded": int(
                (~(table["ref_rhythm_eligible"] & table["pred_rhythm_eligible"])).sum()
            ),
            "bar_reference_fewer_than_8_complete_bars": int((~table["ref_bar_eligible"]).sum()),
            "bar_prediction_fewer_than_8_complete_bars": int((~table["pred_bar_eligible"]).sum()),
            "bar_pair_excluded": int(
                (~(table["ref_bar_eligible"] & table["pred_bar_eligible"])).sum()
            ),
        },
        "bootstrap_metrics": bootstrap_metrics,
        "timing_admission": timing_admission,
        "feature_correlations": correlations,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(correlations).to_csv(
        args.output_dir / "feature_fidelity.csv", index=False
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
