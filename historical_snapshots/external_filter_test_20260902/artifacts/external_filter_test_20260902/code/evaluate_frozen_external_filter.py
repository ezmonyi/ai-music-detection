#!/usr/bin/env python3
"""Evaluate the already-selected pooled vocal filter on a frozen external set.

No external label is used to fit, calibrate, select a representation, or set a
threshold. The primary representation, ridge alpha, score threshold, Demucs
bias, and vocal-activity gate are all inherited from the earlier experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_demucs_artifacts as artifactlib
import cross_generator_generalization as crosslib
import stem_spectral_ablation as stemlib


PRIMARY_REPRESENTATION = "bias_corrected__vocals_without_5_10k"
DIAGNOSTIC_REPRESENTATIONS = (
    "bias_corrected__vocal_spectral",
    "bias_corrected__vocals_full_0_3_20k",
    "bias_corrected__vocals_only_5_10k",
    "bias_corrected__vocals_without_5_10k",
    "bias_corrected__vocals_only_5_16k",
    "bias_corrected__vocals_low_mid_0_3_5k",
)
RIDGE_ALPHA = 10.0
SCORE_THRESHOLD = 0.0
VOCAL_ACTIVITY_THRESHOLD_DB = -18.0
SEED = 20_260_902


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for slug in ("suno", "heartmula", "acestep"):
        parser.add_argument(f"--{slug}-manifest", type=Path, required=True)
        parser.add_argument(f"--{slug}-metrics-dir", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-metrics-dir", type=Path, required=True)
    parser.add_argument("--bias-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_external(manifest_path: Path, metrics_dir: Path) -> dict[str, object]:
    manifest = read_jsonl(manifest_path)
    metric_rows = read_csv(metrics_dir / "vocal_metrics_raw_and_corrected.csv")
    metric_lookup = {
        (str(row["track"]), str(row["variant"])): row for row in metric_rows
    }
    activity_lookup = {
        str(row["track"]): str(row["passes_frozen_activity_threshold"]).lower()
        in {"1", "true", "yes"}
        for row in read_csv(metrics_dir / "vocal_activity.csv")
    }
    with np.load(metrics_dir / "frequency_vectors_raw_and_corrected.npz", allow_pickle=False) as payload:
        vectors = {key: np.asarray(payload[key]) for key in payload.files}
    matrices: dict[str, np.ndarray] = {}
    for variant in ("raw", "bias_corrected"):
        matrices[f"{variant}__vocal_spectral"] = np.asarray(
            [
                [
                    float(metric_lookup[(artifactlib.track_name(row), variant)][metric])
                    for metric in stemlib.ALL_METRICS
                ]
                for row in manifest
            ]
        )
        for name in artifactlib.FREQUENCY_RANGES:
            matrices[f"{variant}__{name}"] = vectors[f"{variant}__{name}"]
    return {
        "manifest": manifest,
        "labels": np.asarray([int(row["label"]) for row in manifest]),
        "tracks": np.asarray([artifactlib.track_name(row) for row in manifest]),
        "active": np.asarray([activity_lookup[artifactlib.track_name(row)] for row in manifest]),
        "full_band": np.asarray([int(row["full_band_eligible"]) == 1 for row in manifest]),
        "matrices": matrices,
    }


def frozen_training_matrix(
    domains: dict[str, dict[str, object]], representation: str, active_only: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    canonical = domains["Suno"]
    human_index = crosslib.indices(canonical, "development", 0, active_only=active_only)
    human = np.asarray(canonical["matrices"][representation])[human_index]
    ai_parts = []
    counts = {"human_unique": int(human.shape[0])}
    for name in ("Suno", "HeartMuLa", "ACE-Step"):
        domain = domains[name]
        current_index = crosslib.indices(domain, "development", 1, active_only=active_only)
        current = np.asarray(domain["matrices"][representation])[current_index]
        ai_parts.append(current)
        counts[f"ai_{name}"] = int(current.shape[0])
    ai = np.concatenate(ai_parts, axis=0)
    repeats = int(np.ceil(ai.shape[0] / human.shape[0]))
    balanced_human = np.tile(human, (repeats, 1))[: ai.shape[0]]
    train_x = np.concatenate((balanced_human, ai), axis=0)
    train_y = np.concatenate(
        (np.zeros(balanced_human.shape[0], dtype=int), np.ones(ai.shape[0], dtype=int))
    )
    counts["human_effective"] = int(balanced_human.shape[0])
    counts["ai_total"] = int(ai.shape[0])
    return train_x, train_y, counts


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    predictions = scores >= SCORE_THRESHOLD
    tn = int(np.sum((labels == 0) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tp = int(np.sum((labels == 1) & predictions))
    if not np.any(labels == 0) or not np.any(labels == 1):
        roc_auc = math.nan
        balanced_accuracy = math.nan
    else:
        balanced_accuracy, roc_auc = artifactlib.metrics(labels, scores)
    return {
        "n": int(labels.size),
        "n_human": int(np.sum(labels == 0)),
        "n_ai": int(np.sum(labels == 1)),
        "accuracy": float(np.mean(predictions == labels)) if labels.size else math.nan,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "sensitivity_ai": safe_div(tp, tp + fn),
        "specificity_human": safe_div(tn, tn + fp),
        "precision_ai": safe_div(tp, tp + fp),
        "f1_ai": safe_div(2 * tp, 2 * tp + fp + fn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def add_intervals(
    metrics: dict[str, object], labels: np.ndarray, scores: np.ndarray, seed: int
) -> dict[str, object]:
    if np.any(labels == 0) and np.any(labels == 1):
        ba_low, ba_high, auc_low, auc_high = artifactlib.bootstrap_intervals(
            labels, scores, seed, repeats=2_000
        )
    else:
        ba_low = ba_high = auc_low = auc_high = math.nan
    return {
        **metrics,
        "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high,
        "roc_auc_ci_low": auc_low,
        "roc_auc_ci_high": auc_high,
    }


def evaluate_slice(
    name: str,
    model_variant: str,
    cohort: str,
    mask: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> dict[str, object]:
    selected_labels = labels[mask]
    selected_scores = scores[mask]
    offset = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(name + model_variant + cohort)
    )
    result = add_intervals(
        classification_metrics(selected_labels, selected_scores),
        selected_labels,
        selected_scores,
        seed + offset,
    )
    return {"slice": name, "model_variant": model_variant, "cohort": cohort, **result}


def plot_group_results(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [
        row for row in rows
        if str(row["slice"]).startswith("group_")
        and row["model_variant"] == "frozen_all_track"
        and row["cohort"] == "full_band_all"
    ]
    selected.sort(key=lambda row: str(row["slice"]))
    names = [str(row["slice"]).replace("group_", "G") for row in selected]
    values = [float(row["balanced_accuracy"]) for row in selected]
    lows = [value - float(row["balanced_accuracy_ci_low"]) for value, row in zip(values, selected)]
    highs = [float(row["balanced_accuracy_ci_high"]) - value for value, row in zip(values, selected)]
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.bar(names, values, color="#4677b4")
    axis.errorbar(names, values, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4)
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Balanced accuracy")
    axis.set_xlabel("Frozen random mixed test group")
    axis.set_title("External AIME accuracy: frozen all-track filter")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_score_distribution(
    path: Path, labels: np.ndarray, scores: np.ndarray
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(float(scores.min()), float(scores.max()), 22)
    axis.hist(scores[labels == 0], bins=bins, alpha=0.62, label="Human", color="#3b78b5")
    axis.hist(scores[labels == 1], bins=bins, alpha=0.62, label="AI", color="#d0614b")
    axis.axvline(SCORE_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="Frozen threshold")
    axis.set_xlabel("Frozen ridge score (AI direction is positive)")
    axis.set_ylabel("Tracks")
    axis.set_title("External AIME score distribution")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(
    path: Path, labels: np.ndarray, scores: np.ndarray
) -> None:
    predictions = (scores >= SCORE_THRESHOLD).astype(int)
    matrix = np.asarray(
        [
            [np.sum((labels == true) & (predictions == predicted)) for predicted in (0, 1)]
            for true in (0, 1)
        ],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    figure.subplots_adjust(left=0.18, right=0.86, bottom=0.15, top=0.88)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks((0, 1), ("Human", "AI"))
    axis.set_yticks((0, 1), ("Human", "AI"))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Frozen all-track filter confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    metric_rows: list[dict[str, object]],
    representation_rows: list[dict[str, object]],
    training_counts: dict[str, dict[str, int]],
    active: np.ndarray,
    full_band: np.ndarray,
) -> None:
    lookup = {
        (str(row["slice"]), str(row["model_variant"]), str(row["cohort"])): row
        for row in metric_rows
    }
    primary = lookup[("all_external", "frozen_all_track", "full_band_all")]
    vocal = lookup[("all_external", "frozen_vocal_active", "full_band_vocal_active")]
    lines = [
        "# Frozen external AIME evaluation",
        "",
        "## Locked protocol",
        "",
        f"- Primary representation: `{PRIMARY_REPRESENTATION}`; it was selected in the earlier development experiment, not on AIME.",
        f"- Ridge alpha `{RIDGE_ALPHA:g}`, score threshold `{SCORE_THRESHOLD:g}`, and raw vocal/mix activity gate `{VOCAL_ACTIVITY_THRESHOLD_DB:g} dB` are unchanged.",
        "- Training is FMA Human plus Suno, HeartMuLa, and ACE-Step development data. External data are MTG-Jamendo Human plus Udio, Riffusion, and Stable Audio v1/v2.",
        "- AIME Human/AI pairs share exact three-tag prompt descriptions; five seeded groups have disjoint descriptions and are balanced 10+10 before the frozen activity gate.",
        "- External labels were not used for representation choice, fitting, threshold calibration, sample selection, or feature extraction.",
        f"- Frozen training counts: `{training_counts}`.",
        "",
        "## Primary result",
        "",
        f"The all-track frozen filter scores all `{int(np.sum(full_band))}` native-full-band tracks. "
        f"Accuracy is `{float(primary['accuracy']):.3f}`, balanced accuracy `{float(primary['balanced_accuracy']):.3f}` "
        f"(95% CI `{float(primary['balanced_accuracy_ci_low']):.3f}-{float(primary['balanced_accuracy_ci_high']):.3f}`), and ROC-AUC `{float(primary['roc_auc']):.3f}` "
        f"(95% CI `{float(primary['roc_auc_ci_low']):.3f}-{float(primary['roc_auc_ci_high']):.3f}`).",
        "",
        f"The separately frozen vocal-active sensitivity model covers `{int(np.sum(active & full_band))}/{int(np.sum(full_band))}` tracks. On that covered subset, accuracy is `{float(vocal['accuracy']):.3f}`, balanced accuracy `{float(vocal['balanced_accuracy']):.3f}`, and AUC `{float(vocal['roc_auc']):.3f}`. Coverage must be reported with this conditional result.",
        "",
        "## Random mixed groups",
        "",
        "| Group | H/AI | Accuracy | Balanced accuracy | ROC-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for index in range(1, 6):
        row = lookup[(f"group_{index:02d}", "frozen_all_track", "full_band_all")]
        lines.append(
            f"| G{index} | {int(row['n_human'])}/{int(row['n_ai'])} | {float(row['accuracy']):.3f} | "
            f"{float(row['balanced_accuracy']):.3f} | {float(row['roc_auc']):.3f} |"
        )
    lines += [
        "",
        "## AI-source breakdown (paired Human controls)",
        "",
        "| AI source | H/AI | Balanced accuracy | ROC-AUC | AI sensitivity | Human specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in ("aime_udio", "aime_riffusion", "aime_stable_audio_v1", "aime_stable_audio_v2"):
        row = lookup[(source, "frozen_all_track", "full_band_all")]
        lines.append(
            f"| {source} | {int(row['n_human'])}/{int(row['n_ai'])} | {float(row['balanced_accuracy']):.3f} | "
            f"{float(row['roc_auc']):.3f} | {float(row['sensitivity_ai']):.3f} | {float(row['specificity_human']):.3f} |"
        )
    lines += [
        "",
        "## Representation diagnostic (not model selection)",
        "",
        "| Representation | Covered BA | Covered AUC |",
        "|---|---:|---:|",
    ]
    for row in representation_rows:
        lines.append(
            f"| {row['representation']} | {float(row['balanced_accuracy']):.3f} | {float(row['roc_auc']):.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is external generator and corpus generalization, not a universal proof of AI origin. Errors can still reflect mastering, codec, provider, or source-separation behavior. The full-band restriction prevents low native sample rate from becoming a trivial class cue. The vocal-active result is conditional and must not be presented without its low coverage.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains = {
        "Suno": crosslib.load_domain(args.suno_manifest, args.suno_metrics_dir),
        "HeartMuLa": crosslib.load_domain(args.heartmula_manifest, args.heartmula_metrics_dir),
        "ACE-Step": crosslib.load_domain(args.acestep_manifest, args.acestep_metrics_dir),
    }
    external = load_external(args.external_manifest, args.external_metrics_dir)
    manifest = list(external["manifest"])
    labels = np.asarray(external["labels"])
    active = np.asarray(external["active"])
    full_band = np.asarray(external["full_band"])

    representation_scores: dict[str, np.ndarray] = {}
    representation_rows = []
    all_track_training_counts: dict[str, int] | None = None
    for representation in DIAGNOSTIC_REPRESENTATIONS:
        train_x, train_y, counts = frozen_training_matrix(
            domains, representation, active_only=False
        )
        if all_track_training_counts is None:
            all_track_training_counts = counts
        elif counts != all_track_training_counts:
            raise RuntimeError("training counts vary across representations")
        scores = artifactlib.fit_predict(
            train_x, train_y, np.asarray(external["matrices"][representation])
        )
        representation_scores[representation] = scores
        mask = full_band
        row = evaluate_slice(
            "all_external", "frozen_all_track", "full_band_all", mask, labels, scores,
            args.seed + len(representation_rows) * 10_000,
        )
        representation_rows.append(
            {
                "representation": representation,
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"slice", "model_variant", "cohort"}
                },
            }
        )
    assert all_track_training_counts is not None
    all_scores = representation_scores[PRIMARY_REPRESENTATION]
    active_train_x, active_train_y, active_training_counts = frozen_training_matrix(
        domains, PRIMARY_REPRESENTATION, active_only=True
    )
    active_scores = artifactlib.fit_predict(
        active_train_x,
        active_train_y,
        np.asarray(external["matrices"][PRIMARY_REPRESENTATION]),
    )
    training_counts = {
        "frozen_all_track": all_track_training_counts,
        "frozen_vocal_active": active_training_counts,
    }

    score_rows = []
    for row, is_active, is_full_band, all_score, active_score in zip(
        manifest, active, full_band, all_scores, active_scores
    ):
        all_decision = "ai" if all_score >= SCORE_THRESHOLD else "human"
        all_correct = int((all_score >= SCORE_THRESHOLD) == bool(int(row["label"])))
        active_decision = "abstain"
        active_correct: int | str = ""
        if is_active and is_full_band:
            active_decision = "ai" if active_score >= SCORE_THRESHOLD else "human"
            active_correct = int((active_score >= SCORE_THRESHOLD) == bool(int(row["label"])))
        score_rows.append(
            {
                "track": artifactlib.track_name(row),
                "group_id": row["group_id"],
                "pair_id": row["pair_id"],
                "label": row["label"],
                "class_name": row["class_name"],
                "source": row["source"],
                "model": row["model"],
                "condition_id": row["condition_id"],
                "prompt_index": row["prompt_index"],
                "original_sample_rate": row["original_sample_rate"],
                "full_band_eligible": int(is_full_band),
                "passes_frozen_activity_threshold": int(is_active),
                "all_track_score": float(all_score),
                "all_track_decision": all_decision,
                "all_track_correct": all_correct,
                "vocal_active_score": float(active_score),
                "vocal_active_decision": active_decision,
                "vocal_active_correct_if_covered": active_correct,
            }
        )

    metric_rows = []
    evaluations = (
        ("frozen_all_track", "full_band_all", full_band, all_scores),
        ("frozen_vocal_active", "full_band_vocal_active", full_band & active, active_scores),
    )
    for model_variant, cohort, cohort_mask, current_scores in evaluations:
        metric_rows.append(
            evaluate_slice(
                "all_external", model_variant, cohort, cohort_mask,
                labels, current_scores, args.seed,
            )
        )
        for group_index in range(1, 6):
            group_id = f"group_{group_index:02d}"
            group_mask = np.asarray([str(row["group_id"]) == group_id for row in manifest])
            metric_rows.append(
                evaluate_slice(
                    group_id, model_variant, cohort, cohort_mask & group_mask,
                    labels, current_scores, args.seed,
                )
            )
        for source in ("aime_udio", "aime_riffusion", "aime_stable_audio_v1", "aime_stable_audio_v2"):
            ai_mask = np.asarray([int(row["label"]) == 1 and str(row["source"]) == source for row in manifest])
            pair_ids = {str(row["pair_id"]) for row, keep in zip(manifest, ai_mask) if keep}
            paired_human_mask = np.asarray(
                [int(row["label"]) == 0 and str(row["pair_id"]) in pair_ids for row in manifest]
            )
            metric_rows.append(
                evaluate_slice(
                    source, model_variant, cohort,
                    cohort_mask & (ai_mask | paired_human_mask),
                    labels, current_scores, args.seed,
                )
            )

    write_csv(args.output_dir / "scores.csv", score_rows)
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "representation_diagnostics.csv", representation_rows)
    plot_group_results(args.output_dir / "random_group_balanced_accuracy.png", metric_rows)
    plot_score_distribution(args.output_dir / "score_distribution.png", labels, all_scores)
    plot_confusion_matrix(args.output_dir / "confusion_matrix.png", labels, all_scores)
    write_report(
        args.output_dir / "REPORT.md", metric_rows, representation_rows,
        training_counts, active, full_band,
    )
    metadata = {
        "external_manifest": str(args.external_manifest.resolve()),
        "external_manifest_sha256": sha256(args.external_manifest),
        "bias_npz": str(args.bias_npz.resolve()),
        "bias_npz_sha256": sha256(args.bias_npz),
        "primary_representation": PRIMARY_REPRESENTATION,
        "primary_selected_before_external_evaluation": True,
        "ridge_alpha": RIDGE_ALPHA,
        "score_threshold": SCORE_THRESHOLD,
        "vocal_activity_threshold_db": VOCAL_ACTIVITY_THRESHOLD_DB,
        "training_counts": training_counts,
        "external_tracks": len(manifest),
        "external_class_counts": dict(sorted(Counter(str(row["class_name"]) for row in manifest).items())),
        "external_labels_used_for_fit_or_threshold": False,
        "full_band_tracks": int(np.sum(full_band)),
        "covered_full_band_tracks": int(np.sum(full_band & active)),
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
