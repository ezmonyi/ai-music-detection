#!/usr/bin/env python3
"""Apply an oracle-derived Demucs frequency-bias filter and rerun vocal analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

import analyze_demucs_artifacts as artifactlib
import stem_spectral_ablation as stemlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--bias-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rebuild", action="store_true")
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


def load_bias(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        frequencies = np.asarray(payload["frequencies_hz"], dtype=np.float64)
        bias = np.asarray(payload["selected_bias_db"], dtype=np.float64)
        window = int(payload["selected_window"])
    expected = np.fft.rfftfreq(stemlib.N_FFT, 1.0 / stemlib.SR)
    if frequencies.shape != expected.shape or not np.allclose(frequencies, expected):
        raise RuntimeError("Bias frequency grid is incompatible with analysis STFT")
    if not np.all(np.isfinite(bias)):
        raise RuntimeError("Bias curve contains non-finite values")
    return frequencies, bias, window


def apply_bias_filter(audio: np.ndarray, bias_db: np.ndarray) -> np.ndarray:
    """Subtract b(f) from STFT magnitudes and retain the original phase."""
    n_fft, hop = stemlib.N_FFT, stemlib.HOP
    frame_count = 1 + (audio.size - n_fft) // hop
    stride = audio.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_count, n_fft),
        strides=(hop * stride, stride),
        writeable=False,
    )
    window = np.hanning(n_fft)
    spectrum = np.fft.rfft(frames * window, axis=1)
    spectrum *= np.power(10.0, -bias_db / 20.0)[None, :]
    corrected_frames = np.fft.irfft(spectrum, n=n_fft, axis=1)
    output = np.zeros(audio.size, dtype=np.float64)
    denominator = np.zeros(audio.size, dtype=np.float64)
    window_squared = window * window
    for frame_index in range(frame_count):
        left = frame_index * hop
        right = left + n_fft
        output[left:right] += corrected_frames[frame_index] * window
        denominator[left:right] += window_squared
    supported = denominator > 1e-10
    output[supported] /= denominator[supported]
    output[~supported] = audio[~supported]
    output -= output.mean()
    return output


def clip_path(clips_dir: Path, row: dict[str, object]) -> Path:
    return clips_dir / f"{artifactlib.track_name(row)}.flac"


def vocal_path(stems_dir: Path, row: dict[str, object]) -> Path:
    return stems_dir / "htdemucs" / artifactlib.track_name(row) / "vocals.wav"


def vocal_activity(mix: np.ndarray, vocals: np.ndarray) -> tuple[float, float]:
    mix_rms = float(np.sqrt(np.mean(mix * mix)))
    vocal_rms = float(np.sqrt(np.mean(vocals * vocals)))
    rms_ratio = 20.0 * math.log10((vocal_rms + 1e-20) / (mix_rms + 1e-20))
    _, mix_power, _, _ = stemlib.framed_stft(mix)
    _, vocal_power, _, _ = stemlib.framed_stft(vocals)
    frame_ratio = 10.0 * np.log10(
        (vocal_power.sum(axis=1) + 1e-20) / (mix_power.sum(axis=1) + 1e-20)
    )
    return rms_ratio, float(np.mean(frame_ratio >= artifactlib.VOCAL_ACTIVITY_THRESHOLD_DB))


def extract(
    rows: list[dict[str, object]],
    clips_dir: Path,
    stems_dir: Path,
    bias_db: np.ndarray,
    output_dir: Path,
    rebuild: bool,
    workers: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    metrics_path = output_dir / "vocal_metrics_raw_and_corrected.csv"
    vectors_path = output_dir / "frequency_vectors_raw_and_corrected.npz"
    if not rebuild and metrics_path.exists() and vectors_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        with np.load(vectors_path, allow_pickle=False) as payload:
            vectors = {name: payload[name] for name in payload.files}
        return metric_rows, vectors

    if workers < 1:
        raise ValueError("workers must be positive")

    first_raw = artifactlib.decode(vocal_path(stems_dir, rows[0]))
    identity = apply_bias_filter(first_raw, np.zeros_like(bias_db))
    relative_error = float(
        np.sqrt(np.mean((identity - first_raw) ** 2))
        / (np.sqrt(np.mean(first_raw * first_raw)) + 1e-20)
    )
    if relative_error > 1e-8:
        raise RuntimeError(f"STFT/ISTFT identity check failed: {relative_error}")

    def process(row: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        mix = artifactlib.decode(clip_path(clips_dir, row))
        raw = artifactlib.decode(vocal_path(stems_dir, row))
        corrected = apply_bias_filter(raw, bias_db)
        rms_ratio, frame_ratio = vocal_activity(mix, raw)
        base = {
            "id": row["id"],
            "label": row["label"],
            "class_name": row["class_name"],
            "source": row["source"],
            "split": row["split"],
            "track": artifactlib.track_name(row),
            "raw_vocal_to_mix_rms_db": rms_ratio,
            "raw_vocal_activity_frame_ratio": frame_ratio,
            "passes_frozen_activity_threshold": int(
                rms_ratio >= artifactlib.VOCAL_ACTIVITY_THRESHOLD_DB
            ),
        }
        current_rows: list[dict[str, object]] = []
        current_vectors: dict[str, np.ndarray] = {}
        for variant, audio in (("raw", raw), ("bias_corrected", corrected)):
            metrics = stemlib.spectral_metrics(audio)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Non-finite metrics: {base['track']}/{variant}")
            current_rows.append({**base, "variant": variant, **metrics})
            for name, vector in artifactlib.frequency_vectors(audio).items():
                current_vectors[f"{variant}__{name}"] = vector
        return current_rows, current_vectors

    metric_rows: list[dict[str, object]] = []
    vector_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    started = time.time()
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = executor.map(process, rows) if executor else map(process, rows)
    try:
        for index, (current_rows, current_vectors) in enumerate(iterator, 1):
            metric_rows.extend(current_rows)
            for name, vector in current_vectors.items():
                vector_lists[name].append(vector)
            if index == 1 or index % 20 == 0 or index == len(rows):
                print(f"corrected vocal analysis {index}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    write_csv(metrics_path, metric_rows)
    vectors = {name: np.stack(items) for name, items in vector_lists.items()}
    np.savez_compressed(vectors_path, **vectors)
    return metric_rows, vectors


def effect_rows(
    rows: list[dict[str, object]], variant: str
) -> list[dict[str, object]]:
    selected = [row for row in rows if str(row["variant"]) == variant]
    labels = np.asarray([int(row["label"]) for row in selected])
    output = []
    for metric in stemlib.ALL_METRICS:
        values = np.asarray([float(row[metric]) for row in selected])
        human, ai = values[labels == 0], values[labels == 1]
        signed_auc = stemlib.auc(labels, values)
        output.append(
            {
                "variant": variant,
                "metric": metric,
                "human_median": float(np.median(human)),
                "ai_median": float(np.median(ai)),
                "cohen_d": stemlib.cohen_d(human, ai),
                "signed_auc": signed_auc,
                "separation_auc": max(signed_auc, 1.0 - signed_auc),
                "p_value": float(mannwhitneyu(human, ai, alternative="two-sided").pvalue),
            }
        )
    return output


def paired_changes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {(str(row["track"]), str(row["variant"])): row for row in rows}
    tracks = sorted({str(row["track"]) for row in rows})
    labels = {str(row["track"]): int(row["label"]) for row in rows}
    output = []
    for metric in stemlib.ALL_METRICS:
        raw = np.asarray([float(lookup[(track, "raw")][metric]) for track in tracks])
        corrected = np.asarray([float(lookup[(track, "bias_corrected")][metric]) for track in tracks])
        delta = corrected - raw
        try:
            p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        except ValueError:
            p_value = math.nan
        output.append(
            {
                "metric": metric,
                "all_median_change": float(np.median(delta)),
                "human_median_change": float(np.median(delta[[labels[track] == 0 for track in tracks]])),
                "ai_median_change": float(np.median(delta[[labels[track] == 1 for track in tracks]])),
                "paired_wilcoxon_p": p_value,
            }
        )
    return output


def matrix_for_variant(
    rows: list[dict[str, object]], ordered_rows: list[dict[str, object]], variant: str
) -> np.ndarray:
    lookup = {(str(row["track"]), str(row["variant"])): row for row in rows}
    return np.asarray(
        [
            [float(lookup[(artifactlib.track_name(row), variant)][metric]) for metric in stemlib.ALL_METRICS]
            for row in ordered_rows
        ]
    )


def write_report(
    path: Path,
    manifest: Path,
    rows: list[dict[str, object]],
    evaluations: list[dict[str, object]],
    effects: list[dict[str, object]],
    selected_window: int,
) -> None:
    locked = {
        str(row["representation"]): row
        for row in evaluations
        if row["evaluation"] == "locked_test" and "vocal_active_sensitivity" not in str(row["representation"])
    }
    lines = [
        "# Raw vs oracle-bias-corrected Demucs vocals",
        "",
        "## Design",
        "",
        f"- Manifest: `{manifest}` ({len(rows)//2} tracks).",
        f"- Correction: subtract the MUSDB-oracle frequency bias using the selected Savitzky-Golay window `{selected_window}` bins.",
        "- The raw development/locked split is preserved exactly; the correction was frozen before seeing AI/Human labels.",
        "- The vocal-active sensitivity set is frozen from raw vocal/mix RMS >= -18 dB so correction cannot change cohort membership.",
        "",
        "## Locked-test classifier comparison",
        "",
        "| Representation | N | Balanced accuracy | ROC-AUC | 95% AUC CI |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [
        "raw__vocal_spectral",
        "bias_corrected__vocal_spectral",
        *[f"raw__{name}" for name in artifactlib.FREQUENCY_RANGES],
        *[f"bias_corrected__{name}" for name in artifactlib.FREQUENCY_RANGES],
    ]
    for name in order:
        if name not in locked:
            continue
        row = locked[name]
        lines.append(
            f"| {name} | {int(row['n'])} | {float(row['balanced_accuracy']):.3f} | "
            f"{float(row['roc_auc']):.3f} | {float(row['roc_auc_ci_low']):.3f}-{float(row['roc_auc_ci_high']):.3f} |"
        )
    lines += ["", "## Strongest corrected univariate vocal metrics", "", "| Metric | Human median | AI median | Cohen d | Separation AUC | BH q |", "|---|---:|---:|---:|---:|---:|"]
    corrected_effects = sorted(
        [row for row in effects if row["variant"] == "bias_corrected"],
        key=lambda row: -float(row["separation_auc"]),
    )[:10]
    for row in corrected_effects:
        lines.append(
            f"| {row['metric']} | {float(row['human_median']):.4g} | {float(row['ai_median']):.4g} | "
            f"{float(row['cohen_d']):+.3f} | {float(row['separation_auc']):.3f} | {float(row['bh_q_value']):.3g} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The filter removes a shared, average human-oracle separator response. Remaining discrimination can still combine intrinsic source differences, dataset provenance, mastering/codec effects, and class-dependent separation behavior.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    rows = read_jsonl(manifest)
    _, bias_db, selected_window = load_bias(args.bias_npz.resolve())
    metric_rows, vectors = extract(
        rows,
        args.clips_dir.resolve(),
        args.stems_dir.resolve(),
        bias_db,
        output_dir,
        args.rebuild,
        args.workers,
    )

    effects = effect_rows(metric_rows, "raw") + effect_rows(metric_rows, "bias_corrected")
    q_values = stemlib.bh_adjust([float(row["p_value"]) for row in effects])
    for row, q_value in zip(effects, q_values):
        row["bh_q_value"] = q_value
    write_csv(output_dir / "univariate_effects.csv", effects)
    changes = paired_changes(metric_rows)
    q_values = stemlib.bh_adjust([float(row["paired_wilcoxon_p"]) for row in changes])
    for row, q_value in zip(changes, q_values):
        row["paired_wilcoxon_q"] = q_value
    write_csv(output_dir / "paired_metric_changes.csv", changes)

    matrices: dict[str, np.ndarray] = {}
    for variant in ("raw", "bias_corrected"):
        matrices[f"{variant}__vocal_spectral"] = matrix_for_variant(metric_rows, rows, variant)
        for name in artifactlib.FREQUENCY_RANGES:
            matrices[f"{variant}__{name}"] = vectors[f"{variant}__{name}"]

    evaluations: list[dict[str, object]] = []
    source_evaluations: list[dict[str, object]] = []
    for name, matrix in matrices.items():
        evaluations += artifactlib.evaluate_matrix(name, matrix, rows, args.seed)
        source_evaluations += artifactlib.evaluate_locked_sources(name, matrix, rows, args.seed)

    raw_records = [row for row in metric_rows if row["variant"] == "raw"]
    activity_mask = np.asarray([int(row["passes_frozen_activity_threshold"]) == 1 for row in raw_records])
    active_rows = [row for row, active in zip(rows, activity_mask) if active]
    for name, matrix in matrices.items():
        sensitivity_name = f"{name}_vocal_active_sensitivity"
        evaluations += artifactlib.evaluate_matrix(
            sensitivity_name, matrix[activity_mask], active_rows, args.seed + 1_000
        )
        source_evaluations += artifactlib.evaluate_locked_sources(
            sensitivity_name, matrix[activity_mask], active_rows, args.seed + 1_000
        )
    write_csv(output_dir / "classification_ablation.csv", evaluations)
    write_csv(output_dir / "locked_source_breakdown.csv", source_evaluations)
    activity_fields = {
        "id", "label", "class_name", "source", "split", "track",
        "raw_vocal_to_mix_rms_db", "raw_vocal_activity_frame_ratio",
        "passes_frozen_activity_threshold",
    }
    write_csv(
        output_dir / "vocal_activity.csv",
        [{key: row[key] for key in row if key in activity_fields} for row in raw_records],
    )
    write_report(output_dir / "REPORT.md", manifest, metric_rows, evaluations, effects, selected_window)
    metadata = {
        "manifest": str(manifest),
        "tracks": len(rows),
        "bias": str(args.bias_npz.resolve()),
        "selected_window_bins": selected_window,
        "correction": "STFT magnitude multiplied by 10**(-b(f)/20), phase retained, Hann overlap-add",
        "activity_subset_frozen_from_raw": True,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
