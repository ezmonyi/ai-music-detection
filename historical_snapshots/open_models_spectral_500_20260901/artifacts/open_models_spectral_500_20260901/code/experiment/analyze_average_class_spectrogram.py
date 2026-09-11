#!/usr/bin/env python3
"""Average AI-minus-Human vocal spectrogram maps for one generator layout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_bias_corrected_vocals as biaslib
import analyze_demucs_artifacts as artifactlib
import stem_spectral_ablation as stemlib


VARIANTS = ("raw", "bias_corrected")
COHORTS = ("all", "development", "locked_test")
CLASS_NAMES = ("human", "ai")
FREQ_EDGES = np.geomspace(300.0, 20_000.0, 129)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--bias-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-name", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def vocal_path(stems_dir: Path, row: dict[str, object]) -> Path:
    return stems_dir / "htdemucs" / artifactlib.track_name(row) / "vocals.wav"


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


def matches(row: dict[str, object], cohort: str) -> bool:
    return cohort == "all" or str(row["split"]) == cohort


def plot_heatmap(
    path: Path,
    arrays: dict[str, np.ndarray],
    times: np.ndarray,
    freqs: np.ndarray,
    title: str,
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
        colorbar.set_label("AI minus Human (dB)")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    comparison_name: str,
    counts: dict[str, dict[str, int]],
    correction_changes: dict[str, dict[str, float]],
) -> None:
    lines = [
        f"# Average AI-minus-Human vocal spectrogram: {comparison_name}",
        "",
        "## Design",
        "",
        "- Each map is the class mean spectrogram of AI vocals minus the class mean spectrogram of Human vocals.",
        "- Positive values mean more AI vocal energy at that time/frequency; negative values mean more Human vocal energy.",
        "- `absolute` retains level; `shape` subtracts every frame's 300 Hz-20 kHz mean and isolates spectral balance.",
        "- The corrected panels apply the previously frozen MUSDB-derived Demucs frequency-response filter to every vocal stem.",
        "- Songs are not waveform-aligned. Time is normalized only by the common 30 s excerpt position, so maps show population-average structure, not sample cancellation.",
        "",
        "## Cohorts",
        "",
        "| Cohort | Human | AI | Max correction change, absolute | Max correction change, shape |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort in COHORTS:
        lines.append(
            f"| {cohort} | {counts[cohort]['human']} | {counts[cohort]['ai']} | "
            f"{correction_changes[cohort]['absolute_max_change_db']:.6f} dB | "
            f"{correction_changes[cohort]['shape_max_change_db']:.6f} dB |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The map can combine generator acoustics, prompt/domain composition, mastering and class-dependent Demucs behavior. The frozen correction removes only the shared average separator response measured on external oracle vocals.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    if not rows:
        raise SystemExit("empty manifest")
    _, bias_db, selected_window = biaslib.load_bias(args.bias_npz)

    sums: dict[str, dict[str, dict[str, np.ndarray]]] = {
        cohort: {class_name: {} for class_name in CLASS_NAMES} for cohort in COHORTS
    }
    counts: dict[str, dict[str, int]] = {
        cohort: {class_name: 0 for class_name in CLASS_NAMES} for cohort in COHORTS
    }
    times = None
    frequencies = None

    for index, row in enumerate(rows, 1):
        class_name = str(row["class_name"])
        if class_name not in CLASS_NAMES:
            raise RuntimeError(f"unexpected class_name: {class_name}")
        raw = artifactlib.decode(vocal_path(args.stems_dir, row))
        variants = {
            "raw": raw,
            "bias_corrected": biaslib.apply_bias_filter(raw, bias_db),
        }
        current: dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            spec, current_times, current_freqs = log_band_spectrogram(variants[variant])
            current[f"{variant}__absolute_db"] = spec
            current[f"{variant}__shape_db"] = spec - spec.mean(axis=1, keepdims=True)
            if times is None:
                times = current_times
                frequencies = current_freqs
        for cohort in COHORTS:
            if not matches(row, cohort):
                continue
            for key, value in current.items():
                if key not in sums[cohort][class_name]:
                    sums[cohort][class_name][key] = np.zeros_like(value)
                sums[cohort][class_name][key] += value
            counts[cohort][class_name] += 1
        if index == 1 or index % 20 == 0 or index == len(rows):
            print(f"average class spectrogram {index}/{len(rows)}", flush=True)

    assert times is not None and frequencies is not None
    payload: dict[str, np.ndarray] = {"times_s": times, "frequencies_hz": frequencies}
    differences: dict[str, dict[str, np.ndarray]] = {}
    correction_changes: dict[str, dict[str, float]] = {}
    for cohort in COHORTS:
        if min(counts[cohort].values()) < 1:
            raise RuntimeError(f"empty class in {cohort}: {counts[cohort]}")
        differences[cohort] = {}
        for key in sums[cohort]["human"]:
            human_mean = sums[cohort]["human"][key] / counts[cohort]["human"]
            ai_mean = sums[cohort]["ai"][key] / counts[cohort]["ai"]
            difference = ai_mean - human_mean
            differences[cohort][key] = difference
            payload[f"{cohort}__human_mean__{key}"] = human_mean
            payload[f"{cohort}__ai_mean__{key}"] = ai_mean
            payload[f"{cohort}__ai_minus_human__{key}"] = difference
        correction_changes[cohort] = {
            "absolute_max_change_db": float(
                np.max(
                    np.abs(
                        differences[cohort]["bias_corrected__absolute_db"]
                        - differences[cohort]["raw__absolute_db"]
                    )
                )
            ),
            "shape_max_change_db": float(
                np.max(
                    np.abs(
                        differences[cohort]["bias_corrected__shape_db"]
                        - differences[cohort]["raw__shape_db"]
                    )
                )
            ),
        }
        plot_heatmap(
            args.output_dir / f"average_difference_spectrogram_{cohort}.png",
            differences[cohort],
            times,
            frequencies,
            f"Average AI minus Human vocal spectrogram — {args.comparison_name} — {cohort}",
        )

    np.savez_compressed(args.output_dir / "average_difference_spectrograms.npz", **payload)
    metadata = {
        "comparison_name": args.comparison_name,
        "manifest": str(args.manifest.resolve()),
        "bias_npz": str(args.bias_npz.resolve()),
        "selected_bias_window_bins": selected_window,
        "counts": counts,
        "correction_changes": correction_changes,
        "stft": {
            "sample_rate": stemlib.SR,
            "n_fft": stemlib.N_FFT,
            "hop": stemlib.HOP,
            "frequency_bins": len(FREQ_EDGES) - 1,
            "frequency_range_hz": [300, 20_000],
        },
    }
    (args.output_dir / "average_difference_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "AVERAGE_DIFFERENCE_REPORT.md",
        args.comparison_name,
        counts,
        correction_changes,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
