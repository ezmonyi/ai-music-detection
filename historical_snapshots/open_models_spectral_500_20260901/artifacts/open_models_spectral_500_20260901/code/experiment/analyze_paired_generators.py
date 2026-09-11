#!/usr/bin/env python3
"""Prompt-paired HeartMuLa vs ACE-Step vocal spectrogram/metric analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import analyze_bias_corrected_vocals as biaslib
import analyze_demucs_artifacts as artifactlib
import stem_spectral_ablation as stemlib


GENERATORS = ("heartmula", "acestep")
VARIANTS = ("raw", "bias_corrected")
COHORTS = ("all", "development", "locked_test")
FREQ_EDGES = np.geomspace(300.0, 20_000.0, 129)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--stems-root", type=Path, required=True)
    parser.add_argument("--bias-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vocal_path(stems_root: Path, generator: str, item_id: str) -> Path:
    track = f"ai_{generator}_{item_id}"
    return stems_root / generator / "htdemucs" / track / "vocals.wav"


def log_band_spectrogram(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, power, _, freqs = stemlib.framed_stft(audio)
    columns = []
    centers = []
    for left, right in zip(FREQ_EDGES[:-1], FREQ_EDGES[1:]):
        mask = (freqs >= left) & (freqs < right)
        columns.append(10.0 * np.log10(power[:, mask].sum(axis=1) + 1e-14))
        centers.append(math.sqrt(left * right))
    spectrogram = np.stack(columns, axis=1)
    times = (np.arange(spectrogram.shape[0]) * stemlib.HOP + stemlib.N_FFT / 2) / stemlib.SR
    return spectrogram, times, np.asarray(centers)


def cohort_match(row: dict[str, object], cohort: str) -> bool:
    return cohort == "all" or str(row["split"]) == cohort


def bh_adjust(p_values: list[float]) -> list[float]:
    return stemlib.bh_adjust(p_values)


def paired_summary(
    metric_rows: list[dict[str, object]], cohort: str, variant: str
) -> list[dict[str, object]]:
    selected = [
        row
        for row in metric_rows
        if str(row["variant"]) == variant and cohort_match(row, cohort)
    ]
    output = []
    for metric in stemlib.ALL_METRICS:
        heart = np.asarray([float(row[f"heartmula__{metric}"]) for row in selected])
        ace = np.asarray([float(row[f"acestep__{metric}"]) for row in selected])
        delta = heart - ace
        try:
            p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        except ValueError:
            p_value = math.nan
        labels = np.concatenate((np.ones(heart.size), np.zeros(ace.size)))
        signed_auc = stemlib.auc(labels, np.concatenate((heart, ace)))
        output.append(
            {
                "cohort": cohort,
                "variant": variant,
                "metric": metric,
                "n_pairs": len(selected),
                "heartmula_median": float(np.median(heart)),
                "acestep_median": float(np.median(ace)),
                "median_paired_difference_heart_minus_ace": float(np.median(delta)),
                "mean_paired_difference_heart_minus_ace": float(np.mean(delta)),
                "paired_wilcoxon_p": p_value,
                "signed_auc_heart_positive": signed_auc,
                "separation_auc": max(signed_auc, 1.0 - signed_auc),
            }
        )
    q_values = bh_adjust([float(row["paired_wilcoxon_p"]) for row in output])
    for row, q_value in zip(output, q_values):
        row["paired_wilcoxon_q"] = q_value
    return output


def plot_heatmap(
    path: Path,
    arrays: dict[str, np.ndarray],
    times: np.ndarray,
    freqs: np.ndarray,
    title_suffix: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    panels = (
        ("raw__absolute_db", "Raw: absolute level difference"),
        ("raw__shape_db", "Raw: frame-normalized spectral shape"),
        ("bias_corrected__absolute_db", "Corrected: absolute level difference"),
        ("bias_corrected__shape_db", "Corrected: frame-normalized spectral shape"),
    )
    for axis, (key, panel_title) in zip(axes.flat, panels):
        values = arrays[key].T
        limit = max(0.25, float(np.nanpercentile(np.abs(values), 98)))
        mesh = axis.pcolormesh(
            times,
            freqs,
            values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_yscale("log")
        axis.set_ylim(300, 20_000)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title(panel_title)
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.01)
        colorbar.set_label("HeartMuLa − ACE-Step (dB)")
    figure.suptitle(f"Prompt-paired average vocal spectrogram difference — {title_suffix}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    summaries: list[dict[str, object]],
    counts: dict[str, int],
    correction_diff: dict[str, float],
) -> None:
    locked_corrected = sorted(
        [
            row
            for row in summaries
            if row["cohort"] == "locked_test" and row["variant"] == "bias_corrected"
        ],
        key=lambda row: (-float(row["separation_auc"]), float(row["paired_wilcoxon_q"])),
    )[:10]
    lines = [
        "# Prompt-paired HeartMuLa vs ACE-Step vocal analysis",
        "",
        "## Design",
        "",
        f"- Prompt pairs: {counts['all']} total; {counts['development']} development; {counts['locked_test']} locked test.",
        "- Positive heatmap values mean HeartMuLa has greater vocal energy than ACE-Step at that time/frequency.",
        "- `absolute` retains level differences; `shape` subtracts each frame's mean across 300 Hz–20 kHz to emphasize spectral balance rather than loudness.",
        "- Generated songs are prompt-paired but not waveform-aligned; time-axis differences describe average structural position, not sample-level cancellation.",
        "",
        "## Effect of shared Demucs correction on generator contrast",
        "",
        f"- All-pair max absolute change in absolute difference map: {correction_diff['all_absolute_max_change_db']:.6f} dB.",
        f"- Locked-pair max absolute change in absolute difference map: {correction_diff['locked_absolute_max_change_db']:.6f} dB.",
        "- The same frozen frequency response is applied to both generators, so their direct dB contrast should largely cancel; the correction is mainly relevant for each generator versus Human.",
        "",
        "## Strongest locked corrected metric differences",
        "",
        "| Metric | HeartMuLa median | ACE-Step median | Paired Δ H−A | Separation AUC | BH q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in locked_corrected:
        lines.append(
            f"| {row['metric']} | {float(row['heartmula_median']):.4g} | "
            f"{float(row['acestep_median']):.4g} | "
            f"{float(row['median_paired_difference_heart_minus_ace']):+.4g} | "
            f"{float(row['separation_auc']):.3f} | {float(row['paired_wilcoxon_q']):.3g} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This paired comparison controls text conditions, language balance, nominal duration and seed number. It does not make the models' random processes equivalent, and Demucs can respond differently to different generated timbres.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.prompt_manifest)
    _, bias_db, selected_window = biaslib.load_bias(args.bias_npz)
    accumulators: dict[str, dict[str, np.ndarray]] = {cohort: {} for cohort in COHORTS}
    counts = {cohort: 0 for cohort in COHORTS}
    metric_rows: list[dict[str, object]] = []
    times = None
    frequencies = None

    for index, row in enumerate(rows, 1):
        item_id = str(row["id"])
        audio = {
            generator: artifactlib.decode(vocal_path(args.stems_root, generator, item_id))
            for generator in GENERATORS
        }
        variants = {
            generator: {
                "raw": signal,
                "bias_corrected": biaslib.apply_bias_filter(signal, bias_db),
            }
            for generator, signal in audio.items()
        }
        current_metrics: dict[str, object] = {
            "id": item_id,
            "split": row["split"],
            "language": row["language"],
        }
        for variant in VARIANTS:
            specs = {}
            for generator in GENERATORS:
                metrics = stemlib.spectral_metrics(variants[generator][variant])
                for metric, value in metrics.items():
                    current_metrics[f"{generator}__{variant}__{metric}"] = value
                spec, current_times, current_freqs = log_band_spectrogram(
                    variants[generator][variant]
                )
                specs[generator] = spec
                if times is None:
                    times, frequencies = current_times, current_freqs
            absolute = specs["heartmula"] - specs["acestep"]
            shape = (
                specs["heartmula"] - specs["heartmula"].mean(axis=1, keepdims=True)
            ) - (specs["acestep"] - specs["acestep"].mean(axis=1, keepdims=True))
            for cohort in COHORTS:
                if not cohort_match(row, cohort):
                    continue
                for suffix, values in (("absolute_db", absolute), ("shape_db", shape)):
                    key = f"{variant}__{suffix}"
                    if key not in accumulators[cohort]:
                        accumulators[cohort][key] = np.zeros_like(values)
                    accumulators[cohort][key] += values
        for variant in VARIANTS:
            metric_rows.append(
                {
                    "id": item_id,
                    "split": row["split"],
                    "language": row["language"],
                    "variant": variant,
                    **{
                        f"{generator}__{metric}": current_metrics[
                            f"{generator}__{variant}__{metric}"
                        ]
                        for generator in GENERATORS
                        for metric in stemlib.ALL_METRICS
                    },
                }
            )
        for cohort in COHORTS:
            if cohort_match(row, cohort):
                counts[cohort] += 1
        if index == 1 or index % 20 == 0 or index == len(rows):
            print(f"paired analysis {index}/{len(rows)}", flush=True)

    assert times is not None and frequencies is not None
    payload: dict[str, np.ndarray] = {"times_s": times, "frequencies_hz": frequencies}
    for cohort in COHORTS:
        for key in accumulators[cohort]:
            accumulators[cohort][key] /= counts[cohort]
            payload[f"{cohort}__{key}"] = accumulators[cohort][key]
        plot_heatmap(
            args.output_dir / f"average_difference_spectrogram_{cohort}.png",
            accumulators[cohort],
            times,
            frequencies,
            f"{cohort} (n={counts[cohort]})",
        )
    np.savez_compressed(args.output_dir / "average_difference_spectrograms.npz", **payload)
    write_csv(args.output_dir / "paired_vocal_metrics.csv", metric_rows)

    summaries = []
    for cohort in COHORTS:
        for variant in VARIANTS:
            summaries += paired_summary(metric_rows, cohort, variant)
    write_csv(args.output_dir / "paired_metric_summary.csv", summaries)

    correction_diff = {
        "all_absolute_max_change_db": float(
            np.max(
                np.abs(
                    accumulators["all"]["bias_corrected__absolute_db"]
                    - accumulators["all"]["raw__absolute_db"]
                )
            )
        ),
        "locked_absolute_max_change_db": float(
            np.max(
                np.abs(
                    accumulators["locked_test"]["bias_corrected__absolute_db"]
                    - accumulators["locked_test"]["raw__absolute_db"]
                )
            )
        ),
    }
    write_report(args.output_dir / "REPORT.md", summaries, counts, correction_diff)
    metadata = {
        "prompts": len(rows),
        "counts": counts,
        "bias_npz": str(args.bias_npz.resolve()),
        "selected_bias_window_bins": selected_window,
        "stft": {
            "sample_rate": stemlib.SR,
            "n_fft": stemlib.N_FFT,
            "hop": stemlib.HOP,
            "frequency_bins": 128,
            "frequency_range_hz": [300, 20_000],
        },
        "correction_difference": correction_diff,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
