#!/usr/bin/env python3
"""Compare Human-vs-Suno/HeartMuLa/ACE-Step spectral results on one protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

import stem_spectral_ablation as stemlib


GENERATORS = ("Suno", "HeartMuLa", "ACE-Step")
BAND_REPRESENTATIONS = (
    "vocals_full_0_3_20k",
    "vocals_only_5_10k",
    "vocals_without_5_10k",
    "vocals_only_5_16k",
    "vocals_low_mid_0_3_5k",
)
BANDS = (
    ("0.3-5 kHz", 300.0, 5_000.0),
    ("5-10 kHz", 5_000.0, 10_000.0),
    ("10-16 kHz", 10_000.0, 16_000.0),
    ("16-20 kHz", 16_000.0, 20_000.1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for slug in ("suno", "heartmula", "acestep"):
        parser.add_argument(f"--{slug}-metrics-dir", type=Path, required=True)
        parser.add_argument(f"--{slug}-average-npz", type=Path, required=True)
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classification_rows(generator: str, directory: Path) -> list[dict[str, object]]:
    output = []
    for row in read_csv(directory / "classification_ablation.csv"):
        representation = row["representation"]
        if row["evaluation"] != "locked_test" or representation.endswith(
            "_vocal_active_sensitivity"
        ):
            continue
        output.append(
            {
                "generator": generator,
                "representation": representation,
                "dimension": int(row["dimension"]),
                "n": int(row["n"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "balanced_accuracy_ci_low": float(row["balanced_accuracy_ci_low"]),
                "balanced_accuracy_ci_high": float(row["balanced_accuracy_ci_high"]),
                "roc_auc": float(row["roc_auc"]),
                "roc_auc_ci_low": float(row["roc_auc_ci_low"]),
                "roc_auc_ci_high": float(row["roc_auc_ci_high"]),
            }
        )
    return output


def metric_effect_rows(generator: str, directory: Path, cohort: str) -> list[dict[str, object]]:
    rows = read_csv(directory / "vocal_metrics_raw_and_corrected.csv")
    selected = [
        row
        for row in rows
        if row["variant"] == "bias_corrected"
        and (cohort == "all" or row["split"] == cohort)
    ]
    labels = np.asarray([int(row["label"]) for row in selected])
    output = []
    p_values = []
    for metric in stemlib.ALL_METRICS:
        values = np.asarray([float(row[metric]) for row in selected])
        human = values[labels == 0]
        ai = values[labels == 1]
        signed_auc = stemlib.auc(labels, values)
        p_value = float(mannwhitneyu(human, ai, alternative="two-sided").pvalue)
        p_values.append(p_value)
        output.append(
            {
                "generator": generator,
                "cohort": cohort,
                "variant": "bias_corrected",
                "metric": metric,
                "n_human": human.size,
                "n_ai": ai.size,
                "human_median": float(np.median(human)),
                "ai_median": float(np.median(ai)),
                "cohen_d": stemlib.cohen_d(human, ai),
                "signed_auc": signed_auc,
                "separation_auc": max(signed_auc, 1.0 - signed_auc),
                "p_value": p_value,
            }
        )
    for row, q_value in zip(output, stemlib.bh_adjust(p_values)):
        row["bh_q_value"] = q_value
    return output


def average_band_rows(generator: str, path: Path) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    rows = []
    with np.load(path, allow_pickle=False) as payload:
        frequencies = np.asarray(payload["frequencies_hz"])
        locked_profile = np.asarray(
            payload["locked_test__ai_minus_human__bias_corrected__shape_db"]
        ).mean(axis=0)
        for cohort in ("all", "development", "locked_test"):
            for variant in ("raw", "bias_corrected"):
                for kind in ("absolute_db", "shape_db"):
                    values = np.asarray(
                        payload[f"{cohort}__ai_minus_human__{variant}__{kind}"]
                    )
                    for band, low, high in BANDS:
                        mask = (frequencies >= low) & (frequencies < high)
                        selected = values[:, mask]
                        rows.append(
                            {
                                "generator": generator,
                                "cohort": cohort,
                                "variant": variant,
                                "map_kind": kind,
                                "band": band,
                                "mean_ai_minus_human_db": float(np.mean(selected)),
                                "median_ai_minus_human_db": float(np.median(selected)),
                                "mean_absolute_difference_db": float(np.mean(np.abs(selected))),
                                "positive_fraction": float(np.mean(selected > 0)),
                            }
                        )
    return rows, frequencies, locked_profile


def plot_auc(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {(row["generator"], row["representation"]): row for row in rows}
    labels = [
        "Full 0.3-20k",
        "Only 5-10k",
        "Without 5-10k",
        "Only 5-16k",
        "Low-mid 0.3-5k",
    ]
    x = np.arange(len(BAND_REPRESENTATIONS))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for offset, generator in enumerate(GENERATORS):
        values = [
            float(lookup[(generator, f"bias_corrected__{representation}")]["roc_auc"])
            for representation in BAND_REPRESENTATIONS
        ]
        axis.bar(x + (offset - 1) * width, values, width, label=generator)
    axis.axhline(0.5, color="black", linewidth=1, linestyle="--")
    axis.set_ylim(0.45, 1.0)
    axis.set_ylabel("Locked-test ROC-AUC")
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_title("Human-vs-generator vocal frequency ablation (bias-corrected)")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_metric_effects(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["cohort"] == "locked_test"]
    lookup = {(row["generator"], row["metric"]): float(row["cohen_d"]) for row in selected}
    order = sorted(
        stemlib.ALL_METRICS,
        key=lambda metric: -max(abs(lookup[(generator, metric)]) for generator in GENERATORS),
    )
    matrix = np.asarray([[lookup[(generator, metric)] for generator in GENERATORS] for metric in order])
    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(7.5, 11), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(GENERATORS)), GENERATORS)
    axis.set_yticks(np.arange(len(order)), order)
    axis.set_title("Locked corrected metric effects (Cohen d: AI minus Human)")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:+.2f}", ha="center", va="center", fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Cohen d")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_profiles(path: Path, profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for generator, (frequencies, profile) in profiles.items():
        axis.plot(frequencies, profile, linewidth=2, label=generator)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlim(300, 20_000)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("AI minus Human, frame-normalized dB")
    axis.set_title("Locked-test corrected average vocal spectral-shape profile")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report(
    path: Path,
    classification: list[dict[str, object]],
    metrics: list[dict[str, object]],
    bands: list[dict[str, object]],
    paired_dir: Path,
) -> None:
    locked_metrics = [row for row in metrics if row["cohort"] == "locked_test"]
    lines = [
        "# Suno, HeartMuLa and ACE-Step spectral comparison",
        "",
        "## Confirmatory frequency-ablation result",
        "",
        "The classifier is trained on each development split and evaluated once on the corresponding locked test split. AUCs below are therefore generator-specific Human-vs-AI discrimination, not a single universal detector score.",
        "",
        "| Generator | Best corrected band representation | Locked AUC | 95% CI | Balanced accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for generator in GENERATORS:
        candidates = [
            row
            for row in classification
            if row["generator"] == generator
            and str(row["representation"]).startswith("bias_corrected__vocals_")
        ]
        best = max(candidates, key=lambda row: float(row["roc_auc"]))
        lines.append(
            f"| {generator} | {str(best['representation']).removeprefix('bias_corrected__')} | "
            f"{float(best['roc_auc']):.3f} | {float(best['roc_auc_ci_low']):.3f}-{float(best['roc_auc_ci_high']):.3f} | "
            f"{float(best['balanced_accuracy']):.3f} |"
        )

    lines += ["", "## Strongest locked corrected scalar effects", ""]
    for generator in GENERATORS:
        strongest = sorted(
            [row for row in locked_metrics if row["generator"] == generator],
            key=lambda row: (-float(row["separation_auc"]), float(row["bh_q_value"])),
        )[:5]
        lines += [f"### {generator}", "", "| Metric | Human median | AI median | Cohen d | Separation AUC | BH q |", "|---|---:|---:|---:|---:|---:|"]
        for row in strongest:
            lines.append(
                f"| {row['metric']} | {float(row['human_median']):.4g} | {float(row['ai_median']):.4g} | "
                f"{float(row['cohen_d']):+.3f} | {float(row['separation_auc']):.3f} | {float(row['bh_q_value']):.3g} |"
            )
        lines.append("")

    lines += [
        "## High-frequency average-map summary",
        "",
        "Values are locked-test, bias-corrected, frame-normalized AI-minus-Human dB. They summarize the heatmaps without replacing them.",
        "",
        "| Generator | Band | Mean shape difference | Mean absolute difference | Positive fraction |",
        "|---|---|---:|---:|---:|",
    ]
    selected_bands = [
        row
        for row in bands
        if row["cohort"] == "locked_test"
        and row["variant"] == "bias_corrected"
        and row["map_kind"] == "shape_db"
    ]
    for generator in GENERATORS:
        for row in [item for item in selected_bands if item["generator"] == generator]:
            lines.append(
                f"| {generator} | {row['band']} | {float(row['mean_ai_minus_human_db']):+.3f} dB | "
                f"{float(row['mean_absolute_difference_db']):.3f} dB | {float(row['positive_fraction']):.3f} |"
            )

    paired_report = paired_dir / "REPORT.md"
    lines += [
        "",
        "## Reading the result",
        "",
        "- `locked_auc_by_generator.png` asks which frequency regions generalize within each generator-vs-Human task.",
        "- `locked_corrected_metric_effects.png` asks whether the direction and size of scalar spectral cues agree across generators.",
        "- `locked_corrected_spectral_shape_profiles.png` and the per-generator four-panel maps locate population-average excess or deficit over time and frequency.",
        "- Agreement across generators is stronger evidence for a model-family-independent heuristic; disagreement means a detector may be learning a generator fingerprint.",
        "",
        "## Boundaries",
        "",
        "The frozen correction removes a shared average Demucs frequency response, not class-dependent separation error. Muse contributes text prompts only; its Suno-produced waveforms were never used. Prompt style and lyric structure can still bias what both open models generate. Human and generated excerpts are balanced in count and duration but not paired by composition or mastering.",
        "",
        f"Prompt-paired HeartMuLa-vs-ACE-Step details: `{paired_report}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_dirs = {
        "Suno": args.suno_metrics_dir,
        "HeartMuLa": args.heartmula_metrics_dir,
        "ACE-Step": args.acestep_metrics_dir,
    }
    average_paths = {
        "Suno": args.suno_average_npz,
        "HeartMuLa": args.heartmula_average_npz,
        "ACE-Step": args.acestep_average_npz,
    }
    classification = [
        row
        for generator, directory in metric_dirs.items()
        for row in classification_rows(generator, directory)
    ]
    metrics = [
        row
        for generator, directory in metric_dirs.items()
        for cohort in ("all", "locked_test")
        for row in metric_effect_rows(generator, directory, cohort)
    ]
    bands = []
    profiles = {}
    for generator, average_path in average_paths.items():
        current_rows, frequencies, profile = average_band_rows(generator, average_path)
        bands += current_rows
        profiles[generator] = (frequencies, profile)

    write_csv(args.output_dir / "generator_classification_comparison.csv", classification)
    write_csv(args.output_dir / "generator_metric_effect_comparison.csv", metrics)
    write_csv(args.output_dir / "average_spectrogram_band_summary.csv", bands)
    plot_auc(args.output_dir / "locked_auc_by_generator.png", classification)
    plot_metric_effects(args.output_dir / "locked_corrected_metric_effects.png", metrics)
    plot_profiles(args.output_dir / "locked_corrected_spectral_shape_profiles.png", profiles)
    report(args.output_dir / "REPORT.md", classification, metrics, bands, args.paired_dir)
    metadata = {
        "generators": list(GENERATORS),
        "metric_directories": {key: str(value.resolve()) for key, value in metric_dirs.items()},
        "average_spectrograms": {key: str(value.resolve()) for key, value in average_paths.items()},
        "paired_directory": str(args.paired_dir.resolve()),
        "confirmatory_cohort": "locked_test",
        "correction": "frozen external MUSDB-derived Demucs frequency response",
    }
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
