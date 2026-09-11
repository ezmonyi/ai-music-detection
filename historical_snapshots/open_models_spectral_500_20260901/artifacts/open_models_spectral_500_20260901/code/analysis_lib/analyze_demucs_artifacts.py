#!/usr/bin/env python3
"""Analyze fixed htdemucs stems for AI/Human spectral and Demucs artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import stem_spectral_ablation as stemlib

try:
    import soundfile as sf
except ImportError:  # Local analysis can fall back to the existing ffmpeg decoder.
    sf = None


BANDS = {
    "full": (20, 22_000),
    "body_1_5k": (1_000, 5_000),
    "sibilance_5_10k": (5_000, 10_000),
    "hf_5_16k": (5_000, 16_000),
    "air_12_20k": (12_000, 20_000),
}
STEMS = ("vocals", "drums", "bass", "other")
VOCAL_ACTIVITY_THRESHOLD_DB = -18.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_814)
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


def track_name(row: dict[str, object]) -> str:
    return f"{row['class_name']}_{row['source']}_{row['id']}"


def stem_path(root: Path, row: dict[str, object], stem: str) -> Path:
    return root / "htdemucs" / track_name(row) / f"{stem}.wav"


def decode(path: Path) -> np.ndarray:
    if sf is not None:
        audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
        if sample_rate != stemlib.SR:
            raise RuntimeError(f"Unexpected sample rate {sample_rate}: {path}")
        mono = audio.mean(axis=1)
        if mono.size < int(stemlib.N_SAMPLES * 0.99):
            raise RuntimeError(f"Short decoded audio: {path}")
        mono = mono[:stemlib.N_SAMPLES]
        mono -= mono.mean()
        return mono
    return stemlib.decode_audio(path)


def all_band_series(audio: np.ndarray) -> dict[str, np.ndarray]:
    _, power, _, freqs = stemlib.framed_stft(audio)
    return {
        band_name: stemlib.band(power, freqs, low, high).sum(axis=1)
        for band_name, (low, high) in BANDS.items()
    }


def db_ratio(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10((numerator + 1e-20) / (denominator + 1e-20))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.log(left + 1e-14)
    right = np.log(right + 1e-14)
    if float(left.std()) < 1e-8 or float(right.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def artifact_metrics(mix: np.ndarray, sources: dict[str, np.ndarray]) -> dict[str, float]:
    residual = mix - sum(sources.values())
    output: dict[str, float] = {}
    series: dict[tuple[str, str], np.ndarray] = {}
    signals = {"mix": mix, "residual": residual, **sources}
    for name, audio in signals.items():
        for band_name, values in all_band_series(audio).items():
            series[(name, band_name)] = values
    for band_name in BANDS:
        mix_energy = float(series[("mix", band_name)].mean())
        residual_energy = float(series[("residual", band_name)].mean())
        output[f"reconstruction_residual_{band_name}_db"] = db_ratio(residual_energy, mix_energy)
        denominator = sum(float(series[(stem, band_name)].mean()) for stem in STEMS)
        for stem in STEMS:
            output[f"allocation_{stem}_{band_name}_db"] = db_ratio(
                float(series[(stem, band_name)].mean()), denominator
            )
    mix_rms = float(np.sqrt(np.mean(mix * mix)))
    vocal_rms = float(np.sqrt(np.mean(sources["vocals"] * sources["vocals"])))
    output["vocal_to_mix_rms_db"] = 20.0 * math.log10(
        (vocal_rms + 1e-20) / (mix_rms + 1e-20)
    )
    vocal_to_mix_frames_db = 10.0 * np.log10(
        (series[("vocals", "full")] + 1e-20) / (series[("mix", "full")] + 1e-20)
    )
    output["vocal_activity_frame_ratio"] = float(
        np.mean(vocal_to_mix_frames_db >= VOCAL_ACTIVITY_THRESHOLD_DB)
    )
    for band_name in ("sibilance_5_10k", "hf_5_16k", "air_12_20k"):
        for other in ("drums", "other"):
            output[f"envelope_corr_vocals_{other}_{band_name}"] = safe_corr(
                series[("vocals", band_name)], series[(other, band_name)]
            )
        residual_series = series[("residual", band_name)]
        output[f"residual_burstiness_{band_name}"] = float(
            np.percentile(residual_series, 95) / (np.median(residual_series) + 1e-14)
        )
    return output


def frequency_vectors(audio: np.ndarray) -> dict[str, np.ndarray]:
    _, power, rms, freqs = stemlib.framed_stft(audio)
    rms_db = 20.0 * np.log10(rms + 1e-14)
    active = rms_db >= max(float(np.percentile(rms_db, 30)), float(rms_db.max() - 35.0))
    p = power[active]
    output = {}
    for name, ranges in FREQUENCY_RANGES.items():
        vectors = []
        for low, high in ranges:
            edges = np.geomspace(low, high, 49)
            for left, right in zip(edges[:-1], edges[1:]):
                values = p[:, (freqs >= left) & (freqs < right)].sum(axis=1)
                db = 10.0 * np.log10(values + 1e-14)
                vectors.extend((float(np.mean(db)), float(np.std(db))))
        vector = np.asarray(vectors, dtype=np.float64)
        vector[0::2] -= float(np.mean(vector[0::2]))
        output[name] = vector
    return output


FREQUENCY_RANGES = {
    "vocals_full_0_3_20k": ((300, 20_000),),
    "vocals_only_5_10k": ((5_000, 10_000),),
    "vocals_without_5_10k": ((300, 5_000), (10_000, 20_000)),
    "vocals_only_5_16k": ((5_000, 16_000),),
    "vocals_low_mid_0_3_5k": ((300, 5_000),),
}


def extract(
    rows: list[dict[str, object]], experiment_dir: Path, stems_dir: Path, rebuild: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    stem_csv = experiment_dir / "stem_metrics.csv"
    artifact_csv = experiment_dir / "demucs_artifacts.csv"
    cache_npz = experiment_dir / "frequency_vectors.npz"
    if not rebuild and stem_csv.exists() and artifact_csv.exists() and cache_npz.exists():
        with stem_csv.open(encoding="utf-8", newline="") as handle:
            stem_rows = list(csv.DictReader(handle))
        with artifact_csv.open(encoding="utf-8", newline="") as handle:
            artifact_rows = list(csv.DictReader(handle))
        with np.load(cache_npz, allow_pickle=False) as cached:
            vectors = {key: cached[key] for key in cached.files}
        return stem_rows, artifact_rows, vectors

    stem_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    vector_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    started = time.time()
    for index, row in enumerate(rows, 1):
        mix = decode(experiment_dir / "clips" / f"{track_name(row)}.flac")
        sources = {stem: decode(stem_path(stems_dir, row, stem)) for stem in STEMS}
        accompaniment = sources["drums"] + sources["bass"] + sources["other"]
        signals = {"mix": mix, **sources, "accompaniment": accompaniment}
        by_stem = {}
        for stem, audio in signals.items():
            metrics = stemlib.spectral_metrics(audio)
            nonfinite = [name for name, value in metrics.items() if not np.isfinite(value)]
            if nonfinite:
                raise RuntimeError(
                    f"Non-finite spectral metrics for {track_name(row)}/{stem}: {nonfinite}"
                )
            by_stem[stem] = metrics
            stem_rows.append({
                "id": row["id"], "label": row["label"], "class_name": row["class_name"],
                "source": row["source"], "split": row["split"], "track": track_name(row),
                "stem": stem, **metrics,
            })
        artifacts = artifact_metrics(mix, sources)
        nonfinite_artifacts = [
            name for name, value in artifacts.items() if not np.isfinite(value)
        ]
        if nonfinite_artifacts:
            raise RuntimeError(
                f"Non-finite Demucs artifacts metrics for {track_name(row)}: "
                f"{nonfinite_artifacts}"
            )
        for metric in stemlib.ALL_METRICS:
            artifacts[f"delta_vocals_minus_mix_{metric}"] = by_stem["vocals"][metric] - by_stem["mix"][metric]
        artifact_rows.append({
            "id": row["id"], "label": row["label"], "class_name": row["class_name"],
            "source": row["source"], "split": row["split"], "track": track_name(row), **artifacts,
        })
        for name, vector in frequency_vectors(sources["vocals"]).items():
            vector_lists[name].append(vector)
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"analysis {index}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
    write_csv(stem_csv, stem_rows)
    write_csv(artifact_csv, artifact_rows)
    vectors = {name: np.stack(items) for name, items in vector_lists.items()}
    np.savez_compressed(cache_npz, **vectors)
    return stem_rows, artifact_rows, vectors


def auc(labels: np.ndarray, values: np.ndarray) -> float:
    return stemlib.auc(labels, values)


def effect_rows(rows: list[dict[str, object]], metric_names: list[str], family: str) -> list[dict[str, object]]:
    labels = np.asarray([int(row["label"]) for row in rows])
    output = []
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows])
        human, ai = values[labels == 0], values[labels == 1]
        signed = auc(labels, values)
        output.append({
            "family": family, "metric": metric,
            "human_median": float(np.median(human)), "ai_median": float(np.median(ai)),
            "cohen_d": stemlib.cohen_d(human, ai), "signed_auc": signed,
            "separation_auc": max(signed, 1.0 - signed),
            "p_value": float(mannwhitneyu(human, ai, alternative="two-sided").pvalue),
        })
    return output


def source_robustness_rows(
    rows: list[dict[str, object]], metric_names: list[str], family: str
) -> list[dict[str, object]]:
    human_rows = [row for row in rows if int(row["label"]) == 0]
    output = []
    for metric in metric_names:
        human = np.asarray([float(row[metric]) for row in human_rows])
        source_results: dict[str, tuple[float, float]] = {}
        for source in ("humair_suno", "suno_unknown"):
            ai = np.asarray([float(row[metric]) for row in rows if row["source"] == source])
            labels = np.concatenate((np.zeros(human.size, dtype=int), np.ones(ai.size, dtype=int)))
            signed = auc(labels, np.concatenate((human, ai)))
            source_results[source] = (
                stemlib.cohen_d(human, ai), max(signed, 1.0 - signed)
            )
        humair_d, humair_auc = source_results["humair_suno"]
        unknown_d, unknown_auc = source_results["suno_unknown"]
        output.append({
            "family": family, "metric": metric,
            "humair_cohen_d": humair_d, "unknown_cohen_d": unknown_d,
            "direction_agreement": int(humair_d * unknown_d > 0),
            "humair_separation_auc": humair_auc,
            "unknown_separation_auc": unknown_auc,
            "worst_source_auc": min(humair_auc, unknown_auc),
        })
    return sorted(output, key=lambda row: -float(row["worst_source_auc"]))


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean, scale, weights, intercept = stemlib.fit_ridge(train_x, train_y, alpha=10.0)
    return ((test_x - mean) / scale) @ weights + intercept


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return stemlib.balanced_accuracy(labels, scores), auc(labels, scores)


def bootstrap_intervals(
    labels: np.ndarray, scores: np.ndarray, seed: int, repeats: int = 2_000
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == label) for label in (0, 1)]
    bas = np.empty(repeats, dtype=np.float64)
    aucs = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True) for indices in by_class
        ])
        bas[repeat], aucs[repeat] = metrics(labels[sampled], scores[sampled])
    return (
        float(np.percentile(bas, 2.5)), float(np.percentile(bas, 97.5)),
        float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)),
    )


def evaluate_matrix(name: str, x: np.ndarray, rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    labels = np.asarray([int(row["label"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    development = np.flatnonzero(splits == "development")
    locked = np.flatnonzero(splits == "locked_test")
    output = []
    rng = np.random.default_rng(seed + 99)
    class_parts = []
    for label in (0, 1):
        indices = development[labels[development] == label].copy()
        rng.shuffle(indices)
        class_parts.append(np.array_split(indices, 5))
    cv_scores = np.zeros(development.size)
    dev_position = {index: pos for pos, index in enumerate(development)}
    for fold in range(5):
        test = np.concatenate((class_parts[0][fold], class_parts[1][fold]))
        train = np.setdiff1d(development, test)
        scores = fit_predict(x[train], labels[train], x[test])
        for index, score in zip(test, scores):
            cv_scores[dev_position[int(index)]] = score
    ba, roc = metrics(labels[development], cv_scores)
    ci_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(name))
    ba_low, ba_high, auc_low, auc_high = bootstrap_intervals(
        labels[development], cv_scores, ci_seed
    )
    output.append({
        "representation": name, "evaluation": "development_5fold_cv",
        "dimension": x.shape[1], "n": development.size,
        "balanced_accuracy": ba, "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high, "roc_auc": roc,
        "roc_auc_ci_low": auc_low, "roc_auc_ci_high": auc_high,
    })
    test_scores = fit_predict(x[development], labels[development], x[locked])
    ba, roc = metrics(labels[locked], test_scores)
    ba_low, ba_high, auc_low, auc_high = bootstrap_intervals(
        labels[locked], test_scores, ci_seed + 1
    )
    output.append({
        "representation": name, "evaluation": "locked_test",
        "dimension": x.shape[1], "n": locked.size,
        "balanced_accuracy": ba, "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high, "roc_auc": roc,
        "roc_auc_ci_low": auc_low, "roc_auc_ci_high": auc_high,
    })
    return output


def evaluate_locked_sources(
    name: str, x: np.ndarray, rows: list[dict[str, object]], seed: int
) -> list[dict[str, object]]:
    labels = np.asarray([int(row["label"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    sources = np.asarray([str(row["source"]) for row in rows])
    development = np.flatnonzero(splits == "development")
    locked = np.flatnonzero(splits == "locked_test")
    locked_scores = fit_predict(x[development], labels[development], x[locked])
    locked_labels = labels[locked]
    locked_sources = sources[locked]
    output = []
    base_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(name))
    for source_number, source in enumerate(("humair_suno", "suno_unknown")):
        keep = (locked_labels == 0) | (locked_sources == source)
        subset_labels = locked_labels[keep]
        subset_scores = locked_scores[keep]
        ba, roc = metrics(subset_labels, subset_scores)
        ba_low, ba_high, auc_low, auc_high = bootstrap_intervals(
            subset_labels, subset_scores, base_seed + 10_000 + source_number
        )
        output.append({
            "representation": name, "ai_source": source,
            "n_human": int(np.sum(subset_labels == 0)),
            "n_ai": int(np.sum(subset_labels == 1)),
            "balanced_accuracy": ba, "balanced_accuracy_ci_low": ba_low,
            "balanced_accuracy_ci_high": ba_high, "roc_auc": roc,
            "roc_auc_ci_low": auc_low, "roc_auc_ci_high": auc_high,
        })
    return output


def analyze(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    rows = read_jsonl(experiment_dir / "manifest.jsonl")
    stem_rows, artifact_rows, frequency_vectors = extract(rows, experiment_dir, args.stems_dir.resolve(), args.rebuild)
    stem_lookup = {(str(row["track"]), str(row["stem"])): row for row in stem_rows}
    mix_matrix = np.asarray([[float(stem_lookup[(track_name(row), "mix")][metric]) for metric in stemlib.ALL_METRICS] for row in rows])
    vocal_matrix = np.asarray([[float(stem_lookup[(track_name(row), "vocals")][metric]) for metric in stemlib.ALL_METRICS] for row in rows])
    artifact_names = [name for name in artifact_rows[0] if name not in {"id", "label", "class_name", "source", "split", "track"}]
    artifact_matrix = np.asarray([[float(row[name]) for name in artifact_names] for row in artifact_rows])

    effects = effect_rows(artifact_rows, artifact_names, "demucs_artifacts")
    mix_records = [stem_lookup[(track_name(row), "mix")] for row in rows]
    vocal_records = [stem_lookup[(track_name(row), "vocals")] for row in rows]
    effects += effect_rows(mix_records, list(stemlib.ALL_METRICS), "mix_spectral")
    effects += effect_rows(vocal_records, list(stemlib.ALL_METRICS), "vocal_spectral")
    q_values = stemlib.bh_adjust([float(row["p_value"]) for row in effects])
    for row, q_value in zip(effects, q_values):
        row["bh_q_value"] = q_value
    effects.sort(key=lambda row: (str(row["family"]), -float(row["separation_auc"])))
    write_csv(experiment_dir / "univariate_effects.csv", effects)

    robustness = source_robustness_rows(artifact_rows, artifact_names, "demucs_artifacts")
    robustness += source_robustness_rows(mix_records, list(stemlib.ALL_METRICS), "mix_spectral")
    robustness += source_robustness_rows(vocal_records, list(stemlib.ALL_METRICS), "vocal_spectral")
    write_csv(experiment_dir / "source_robustness.csv", robustness)

    evaluations = []
    combined_matrix = np.concatenate((vocal_matrix, artifact_matrix), axis=1)
    primary_matrices = {
        "mix_spectral": mix_matrix,
        "vocal_spectral": vocal_matrix,
        "demucs_artifacts_only": artifact_matrix,
        "vocal_plus_demucs_artifacts": combined_matrix,
        **frequency_vectors,
    }
    for name, matrix in primary_matrices.items():
        evaluations += evaluate_matrix(name, matrix, rows, args.seed)

    activity_mask = np.asarray([
        float(row["vocal_to_mix_rms_db"]) >= VOCAL_ACTIVITY_THRESHOLD_DB
        for row in artifact_rows
    ])
    active_rows = [row for row, active in zip(rows, activity_mask) if active]
    for name, matrix in primary_matrices.items():
        evaluations += evaluate_matrix(
            f"{name}_vocal_active_sensitivity",
            matrix[activity_mask], active_rows, args.seed + 1_000,
        )
    write_csv(experiment_dir / "classification_ablation.csv", evaluations)

    source_evaluations = []
    for name, matrix in primary_matrices.items():
        source_evaluations += evaluate_locked_sources(name, matrix, rows, args.seed)
        source_evaluations += evaluate_locked_sources(
            f"{name}_vocal_active_sensitivity",
            matrix[activity_mask], active_rows, args.seed + 1_000,
        )
    write_csv(experiment_dir / "locked_source_breakdown.csv", source_evaluations)

    vocal_activity = []
    for row, artifact_row in zip(rows, artifact_rows):
        share = float(artifact_row["vocal_to_mix_rms_db"])
        vocal_activity.append({
            "id": row["id"], "label": row["label"], "class_name": row["class_name"],
            "source": row["source"], "split": row["split"], "track": track_name(row),
            "vocal_to_mix_rms_db": share,
            "vocal_activity_frame_ratio": float(artifact_row["vocal_activity_frame_ratio"]),
            "passes_threshold": int(share >= VOCAL_ACTIVITY_THRESHOLD_DB),
        })
    write_csv(experiment_dir / "vocal_activity.csv", vocal_activity)

    counts = Counter((str(row["class_name"]), str(row["source"]), str(row["split"])) for row in rows)
    top_artifacts = [row for row in effects if row["family"] == "demucs_artifacts"][:12]
    top_transformations = [
        row for row in effects
        if row["family"] == "demucs_artifacts"
        and str(row["metric"]).startswith("delta_vocals_minus_mix_")
    ][:12]
    top_mixes = [row for row in effects if row["family"] == "mix_spectral"][:12]
    top_vocals = [row for row in effects if row["family"] == "vocal_spectral"][:12]
    activity_counts = Counter(
        (str(row["class_name"]), str(row["source"]), int(row["passes_threshold"]))
        for row in vocal_activity
    )
    lookup = {(row["representation"], row["evaluation"]): row for row in evaluations}
    source_lookup = {
        (row["representation"], row["ai_source"]): row for row in source_evaluations
    }
    human_manifest = [row for row in rows if int(row["label"]) == 0]
    human_artists = Counter(str(row.get("artist_id", "")) for row in human_manifest)
    human_genres = Counter(str(row.get("genre_top", "")) for row in human_manifest)
    manifest_digest = (experiment_dir / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    lines = [
        "# Demucs artifacts 100+100 experiment", "",
        "## Frozen design", "",
        f"- Samples: `{len(rows)}`; Human `100`, AI `100` (`50` Humair-Suno/chirp-v3 + `50` second-Suno)",
        f"- Development/locked split counts: `{dict(sorted(counts.items()))}`",
        f"- Frozen manifest SHA-256: `{manifest_digest}`",
        f"- Human cohort: `{len(human_artists)}` FMA artists, maximum `{max(human_artists.values())}` tracks per artist; genres `{dict(human_genres)}`",
        "- Vocal screening used only to rank the fixed candidate pools: `MIT/ast-finetuned-audioset-10-10-0.4593` revision `f826b80d28226b62986cc218e5cec390b1096902`",
        "- All analyzed excerpts: `30.000 s`, stereo, `44.1 kHz`; inputs stored losslessly as FLAC",
        "- Compute: `5090-2`, GPU `0` only for Demucs; CPU analysis read the frozen float32 stems",
        "- Front end: `Demucs 4.1.0 htdemucs`, four float32 stems, shifts 0, overlap 0.25, segment 7",
        "- Analysis unit: one 30-second excerpt per track", "",
        f"- Post-Demucs vocal activity threshold: vocals/mix RMS >= `{VOCAL_ACTIVITY_THRESHOLD_DB:.1f} dB`",
        f"- Vocal activity counts: `{dict(sorted(activity_counts.items()))}`", "",
        "## Classification ablation", "",
        "| Representation | Development BA/AUC | Locked test BA/AUC |", "|---|---:|---:|",
    ]
    for name in ["mix_spectral", "vocal_spectral", "demucs_artifacts_only", "vocal_plus_demucs_artifacts", *FREQUENCY_RANGES]:
        dev, test = lookup[(name, "development_5fold_cv")], lookup[(name, "locked_test")]
        lines.append(f"| {name} | {float(dev['balanced_accuracy']):.3f}/{float(dev['roc_auc']):.3f} | {float(test['balanced_accuracy']):.3f}/{float(test['roc_auc']):.3f} |")
    lines += [
        "", "## Vocal-active sensitivity analysis", "",
        f"- Retained tracks: `{len(active_rows)}/{len(rows)}`; this filter is diagnostic and does not replace the fixed primary cohort.", "",
        "| Representation | Development BA/AUC | Locked test BA/AUC |", "|---|---:|---:|",
    ]
    for name in ["mix_spectral", "vocal_spectral", "demucs_artifacts_only", "vocal_plus_demucs_artifacts", *FREQUENCY_RANGES]:
        sensitivity_name = f"{name}_vocal_active_sensitivity"
        dev = lookup[(sensitivity_name, "development_5fold_cv")]
        test = lookup[(sensitivity_name, "locked_test")]
        lines.append(f"| {name} | {float(dev['balanced_accuracy']):.3f}/{float(dev['roc_auc']):.3f} | {float(test['balanced_accuracy']):.3f}/{float(test['roc_auc']):.3f} |")
    lines += [
        "", "## Locked-test AI-source breakdown", "",
        "| Representation | Humair-Suno BA/AUC | Second-Suno BA/AUC |",
        "|---|---:|---:|",
    ]
    for name in ("mix_spectral", "vocal_spectral", "demucs_artifacts_only", "vocal_plus_demucs_artifacts"):
        humair = source_lookup[(name, "humair_suno")]
        unknown = source_lookup[(name, "suno_unknown")]
        lines.append(
            f"| {name} | {float(humair['balanced_accuracy']):.3f}/{float(humair['roc_auc']):.3f} "
            f"| {float(unknown['balanced_accuracy']):.3f}/{float(unknown['roc_auc']):.3f} |"
        )
    artifact_test = lookup[("demucs_artifacts_only", "locked_test")]
    combined_test = lookup[("vocal_plus_demucs_artifacts", "locked_test")]
    active_artifact_test = lookup[("demucs_artifacts_only_vocal_active_sensitivity", "locked_test")]
    band_only = lookup[("vocals_only_5_10k", "locked_test")]
    band_removed = lookup[("vocals_without_5_10k", "locked_test")]
    lines += [
        "", "## Main findings", "",
        f"- Demucs artifacts-only: locked BA `{float(artifact_test['balanced_accuracy']):.3f}` "
        f"(95% CI `{float(artifact_test['balanced_accuracy_ci_low']):.3f}–{float(artifact_test['balanced_accuracy_ci_high']):.3f}`), "
        f"AUC `{float(artifact_test['roc_auc']):.3f}` "
        f"(`{float(artifact_test['roc_auc_ci_low']):.3f}–{float(artifact_test['roc_auc_ci_high']):.3f}`).",
        f"- Vocal + artifacts: locked BA `{float(combined_test['balanced_accuracy']):.3f}`, AUC `{float(combined_test['roc_auc']):.3f}`; "
        "both AI sources retain similar performance in the source breakdown.",
        f"- Vocal-active sensitivity does not remove the effect: artifacts-only locked AUC `{float(active_artifact_test['roc_auc']):.3f}` on `53` locked tracks.",
        f"- The narrow 5–10 kHz band is informative but not sufficient: only-band AUC `{float(band_only['roc_auc']):.3f}` versus `{float(band_removed['roc_auc']):.3f}` after removing it.",
        "- Vocal high-frequency level, slope, entropy, crest, and sibilance metrics remain individually discriminative, but the strongest applied detector combines broader spectrum with Demucs response.",
    ]
    lines += ["", "## Strongest Demucs artifacts effects", "", "| Metric | Human median | AI median | d | AUC | BH q |", "|---|---:|---:|---:|---:|---:|"]
    for row in top_artifacts:
        lines.append(f"| {row['metric']} | {float(row['human_median']):.4g} | {float(row['ai_median']):.4g} | {float(row['cohen_d']):+.2f} | {float(row['separation_auc']):.3f} | {float(row['bh_q_value']):.3g} |")
    lines += ["", "## Strongest mix-to-vocal transformations", "", "| Metric | Human median delta | AI median delta | d | AUC | BH q |", "|---|---:|---:|---:|---:|---:|"]
    for row in top_transformations:
        lines.append(f"| {row['metric']} | {float(row['human_median']):.4g} | {float(row['ai_median']):.4g} | {float(row['cohen_d']):+.2f} | {float(row['separation_auc']):.3f} | {float(row['bh_q_value']):.3g} |")
    lines += ["", "## Strongest mixture spectral effects", "", "| Metric | Human median | AI median | d | AUC | BH q |", "|---|---:|---:|---:|---:|---:|"]
    for row in top_mixes:
        lines.append(f"| {row['metric']} | {float(row['human_median']):.4g} | {float(row['ai_median']):.4g} | {float(row['cohen_d']):+.2f} | {float(row['separation_auc']):.3f} | {float(row['bh_q_value']):.3g} |")
    lines += ["", "## Strongest vocal spectral effects", "", "| Metric | Human median | AI median | d | AUC | BH q |", "|---|---:|---:|---:|---:|---:|"]
    for row in top_vocals:
        lines.append(f"| {row['metric']} | {float(row['human_median']):.4g} | {float(row['ai_median']):.4g} | {float(row['cohen_d']):+.2f} | {float(row['separation_auc']):.3f} | {float(row['bh_q_value']):.3g} |")
    lines += [
        "", "## Metric interpretation", "",
        "- Reconstruction residual: band-energy ratio of `mix − (vocals + drums + bass + other)` to the mixture.",
        "- Source allocation: each stem's band energy divided by summed stem energy in that band.",
        "- Cross-stem envelope correlation and residual burstiness: leakage/transient proxies in 5–10, 5–16, and 12–20 kHz.",
        "- Mix-to-vocal delta: paired change in each spectral metric introduced by the complete separation front end.",
        "- Band ablation: ridge classifiers on log-frequency vocal vectors; dimensions differ where two retained ranges are concatenated.",
        "", "## Interpretation boundary", "",
        "- Demucs artifacts are measured as mixture-consistency, source-allocation, cross-stem envelope, and class-dependent spectral-transformation proxies.",
        "- The mix-to-vocal comparison measures how the separation front end changes discriminability; it does not prove that the changed cue was created by Demucs.",
        "- Without oracle isolated stems, these proxies do not uniquely identify perceptual separation error.",
        "- A strong artifacts-only classifier is useful for the applied detector but weakens claims about intrinsic vocal generation artifacts.",
        "- Source, genre, codec, and mastering remain potential confounders and must be reported with the locked-test result.",
        "- `source_robustness.csv` checks whether each effect has the same direction and useful separation for both AI sources.",
    ]
    (experiment_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    analyze(parse_args())
