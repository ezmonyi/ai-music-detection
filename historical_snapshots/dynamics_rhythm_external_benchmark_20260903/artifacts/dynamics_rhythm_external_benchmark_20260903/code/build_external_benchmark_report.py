#!/usr/bin/env python3
"""Build aggregate admissions, figures, and the English external benchmark report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(row: dict) -> str:
    return f"{row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rhythm_path = RESULTS / "rhythm_benchmark" / "rhythm_benchmark_summary.json"
    oracle_path = RESULTS / "maestro_dynamics" / "summary.json"
    waveform_path = RESULTS / "maestro_waveform_dynamics" / "summary.json"
    structure_path = RESULTS / "salami_structure" / "summary.json"
    rhythm = load(rhythm_path)
    oracle = load(oracle_path)
    waveform = load(waveform_path)
    structure = load(structure_path)

    decisions: list[dict[str, object]] = []
    for row in rhythm["final0_feature_fidelity"]:
        decisions.append(
            {
                "family": "rhythm_or_bar_richness",
                "estimator": "Beat This final0",
                "feature": row["feature"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "point_threshold": 0.75,
                "lower_ci_threshold": 0.65,
                "base_task_gate": True,
                "detector_admitted": bool(row["admitted"]),
                "decision": "reject: feature fidelity below threshold",
            }
        )
    for row in oracle["window_feature_fidelity"]:
        if row["estimator"] != "harmonic_attack_level_db":
            continue
        decisions.append(
            {
                "family": "dynamics_oracle_onset_pitch",
                "estimator": row["estimator"],
                "feature": row["feature"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "point_threshold": 0.50,
                "lower_ci_threshold": 0.35,
                "base_task_gate": True,
                "detector_admitted": False,
                "decision": "oracle pass only: not deployable without MIDI onset/pitch",
            }
        )
    for row in waveform["feature_fidelity"]:
        decisions.append(
            {
                "family": "dynamics_waveform_only",
                "estimator": "spectral-flux plus harmonic attack peaks",
                "feature": row["feature"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "point_threshold": 0.50,
                "lower_ci_threshold": 0.35,
                "base_task_gate": bool(waveform["event_admitted"]),
                "detector_admitted": bool(row["admitted"]),
                "decision": "admit" if row["admitted"] else "reject: window fidelity below threshold",
            }
        )
    for row in structure["feature_fidelity"]:
        decisions.append(
            {
                "family": "section_or_bar_length",
                "estimator": "All-In-One plus Beat This",
                "feature": row["feature"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "point_threshold": 0.70,
                "lower_ci_threshold": 0.55,
                "base_task_gate": bool(structure["boundary_admitted"]),
                "detector_admitted": bool(row["admitted"]),
                "decision": "reject: boundary gate and feature fidelity failed",
            }
        )
    decision_path = RESULTS / "admission_decisions.csv"
    with decision_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)

    aggregate = {
        "benchmark_date": "2026-09-03",
        "gpu_inference_host": "5090-5",
        "gpu": "NVIDIA GeForce RTX 5090 cuda:0",
        "rhythm_timing_admitted": rhythm["timing_admission"],
        "rhythm_detector_features_admitted": rhythm["admitted_detector_features"],
        "oracle_dynamics_features_passing": oracle["admitted_window_features"],
        "waveform_dynamics_event_gate": waveform["event_admitted"],
        "waveform_dynamics_features_admitted": waveform["admitted_features"],
        "structure_boundary_admitted": structure["boundary_admitted"],
        "structure_features_admitted": structure["admitted_features"],
        "features_admitted_to_ai_human_ablation": [],
        "ai_human_ablation_executed": False,
        "ai_human_locked_test_touched": False,
        "reason": "No deployable detector-facing feature passed its external admission gate.",
        "source_summary_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (rhythm_path, oracle_path, waveform_path, structure_path)
        },
    }
    aggregate_path = RESULTS / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    color_pass, color_fail, color_oracle = "#287D5A", "#B6534B", "#4E79A7"

    axis = axes[0, 0]
    labels = ["Beat F1", "Downbeat F1"]
    keys = ["f1_beat", "f1_downbeat"]
    thresholds = [0.85, 0.72]
    values = [rhythm["final0_bootstrap_metrics"][key]["estimate"] for key in keys]
    lows = [rhythm["final0_bootstrap_metrics"][key]["ci_low"] for key in keys]
    highs = [rhythm["final0_bootstrap_metrics"][key]["ci_high"] for key in keys]
    x = np.arange(len(labels))
    axis.errorbar(x, values, yerr=[np.array(values) - lows, np.array(highs) - values], fmt="o", color=color_pass, capsize=6, markersize=8)
    for i, threshold in enumerate(thresholds):
        axis.hlines(threshold, i - 0.30, i + 0.30, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.60, 0.95)
    axis.set_ylabel("F1 (95% track-bootstrap CI)")
    axis.set_title("A. Base rhythm localization passes")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[0, 1]
    rows = rhythm["final0_feature_fidelity"]
    y = np.arange(len(rows))
    values = np.array([row["estimate"] for row in rows])
    lows = np.array([row["ci_low"] for row in rows])
    highs = np.array([row["ci_high"] for row in rows])
    axis.errorbar(values, y, xerr=[values - lows, highs - values], fmt="o", color=color_fail, capsize=4)
    axis.axvline(0.75, color="black", linestyle="--", label="point threshold")
    axis.set_yticks(y, [row["feature"].replace("_", " ") for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0.0, 0.85)
    axis.set_xlabel("Reference-prediction Spearman rho")
    axis.set_title("B. Rhythm/bar richness features fail")
    axis.grid(axis="x", alpha=0.25)

    axis = axes[1, 0]
    oracle_rows = [row for row in oracle["window_feature_fidelity"] if row["estimator"] == "harmonic_attack_level_db"]
    wave_rows = waveform["feature_fidelity"]
    x = np.arange(len(FEATURE_LABELS := ["Span", "IQR", "Adjacent change"]))
    width = 0.12
    for offset, current_rows, label, color in (
        (-width, oracle_rows, "Oracle MIDI onset/pitch", color_oracle),
        (width, wave_rows, "Waveform-only", color_fail),
    ):
        values = np.array([row["estimate"] for row in current_rows])
        lows = np.array([row["ci_low"] for row in current_rows])
        highs = np.array([row["ci_high"] for row in current_rows])
        axis.errorbar(x + offset, values, yerr=[values - lows, highs - values], fmt="o", label=label, color=color, capsize=4)
    axis.axhline(0.50, color="black", linestyle="--", label="point threshold")
    axis.set_xticks(x, FEATURE_LABELS)
    axis.set_ylim(-0.30, 1.0)
    axis.set_ylabel("MIDI-audio Spearman rho")
    axis.set_title("C. Dynamics: oracle success does not deploy")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 1]
    boundary = structure["boundary_metrics"]["f1_3s"]
    struct_rows = structure["feature_fidelity"]
    labels = ["Boundary F1@3s"] + [row["feature"].replace("section_", "").replace("_", " ") for row in struct_rows]
    values = np.array([boundary["estimate"]] + [row["estimate"] for row in struct_rows])
    lows = np.array([boundary["ci_low"]] + [row["ci_low"] for row in struct_rows])
    highs = np.array([boundary["ci_high"]] + [row["ci_high"] for row in struct_rows])
    y = np.arange(len(labels))
    axis.errorbar(values, y, xerr=[values - lows, highs - values], fmt="o", color=color_fail, capsize=4)
    axis.axvline(0.60, color="black", linestyle="--", label="boundary threshold")
    axis.axvline(0.70, color="black", linestyle=":", label="feature threshold")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.35, 0.85)
    axis.set_xlabel("F1 or Spearman rho")
    axis.set_title("D. Structure and length features fail")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(axis="x", alpha=0.25)

    figure.suptitle("External validity gates for dynamics, rhythm, and structure", fontsize=16)
    figure_path = FIGURES / "external_benchmark_dashboard.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    rmetrics = rhythm["final0_bootstrap_metrics"]
    lines = [
        "# External Benchmark of Dynamics, Rhythm, and Structure Features",
        "",
        "**Frozen benchmark date:** 2026-09-03  ",
        "**Execution:** learned inference only on RTX 5090 GPU 0 of `5090-5`; deterministic scoring on the same host  ",
        "**Decision:** **no detector-facing feature passed the complete external-to-deployment admission path, so no AI/Human detector ablation was run.**",
        "",
        "![External benchmark dashboard](../figures/external_benchmark_dashboard.png)",
        "",
        "## What the benchmark establishes",
        "",
        "The experiment separates base-task competence from fidelity of the proposed heuristic. Beat/downbeat localization can be accurate while rhythm-richness summaries are unreliable; individual note dynamics can be recovered while 30-second dynamics-variety summaries are unreliable; and a structure model can emit plausible sections while failing boundary and section-length fidelity. Passing a base task is therefore necessary but not sufficient for admission to the applied filter.",
        "",
        "## Benchmark A: rhythm and bar-pattern recovery",
        "",
        "Beat This! `final0`, `final1`, and `final2` were evaluated on 993 GTZAN tracks using the author annotation release and native non-DBN postprocessor. Six additional jazz annotations contained no downbeat column and were retained as explicit exclusions.",
        "",
        "| Metric | final0 estimate [95% CI] | Three-seed mean (SD) | Gate |",
        "|---|---:|---:|---|",
        f"| Beat F1 | {interval(rmetrics['f1_beat'])} | {rhythm['cross_seed_metrics']['f1_beat']['mean']:.3f} ({rhythm['cross_seed_metrics']['f1_beat']['sample_std']:.3f}) | pass |",
        f"| Downbeat F1 | {interval(rmetrics['f1_downbeat'])} | {rhythm['cross_seed_metrics']['f1_downbeat']['mean']:.3f} ({rhythm['cross_seed_metrics']['f1_downbeat']['sample_std']:.3f}) | pass |",
        f"| Beat CMLt / AMLt | {rmetrics['cmlt_beat']['estimate']:.3f} / {rmetrics['amlt_beat']['estimate']:.3f} | {rhythm['cross_seed_metrics']['cmlt_beat']['mean']:.3f} / {rhythm['cross_seed_metrics']['amlt_beat']['mean']:.3f} | descriptive |",
        f"| Downbeat CMLt / AMLt | {rmetrics['cmlt_downbeat']['estimate']:.3f} / {rmetrics['amlt_downbeat']['estimate']:.3f} | {rhythm['cross_seed_metrics']['cmlt_downbeat']['mean']:.3f} / {rhythm['cross_seed_metrics']['amlt_downbeat']['mean']:.3f} | descriptive |",
        "",
        "The timing gate passes, but all five detector summaries fail the preregistered rho >= 0.75 and lower-CI >= 0.65 rule:",
        "",
        "| Candidate summary | rho [95% CI] | Decision |",
        "|---|---:|---|",
    ]
    for row in rhythm["final0_feature_fidelity"]:
        lines.append(f"| `{row['feature']}` | {interval(row)} | reject |")
    lines += [
        "",
        "This is the central rhythm result: the beat tracker is accurate at event localization, but small timing errors and missing downbeats distort second-order variability and entropy summaries too strongly for the intended filter.",
        "",
        "## Benchmark B: acoustic recovery of MIDI dynamics",
        "",
        "The oracle-onset/pitch analysis used 50 composition-disjoint MAESTRO test recordings, four frozen 30-second windows per recording, 65,049 event measurements, and 10,000 recording-cluster bootstrap replicates. The harmonic attack-level estimator passes the event gate (median within-group rho 0.815, 95% CI 0.800--0.834; quartile-ranking accuracy 0.948, 95% CI 0.941--0.956). Its three window features also pass when MIDI onset and pitch are supplied:",
        "",
        "| Oracle feature | rho [95% CI] | Gain drift | Oracle result |",
        "|---|---:|---:|---|",
    ]
    for row in oracle_rows:
        lines.append(f"| `{row['feature']}` | {interval(row)} | {row['max_abs_gain6_drift']:.2e} | pass |")
    lines += [
        "",
        "A real detector has no oracle MIDI. The separately frozen deployment addendum therefore detected onsets and harmonic attack peaks from waveform alone. On the same 200 windows it achieved onset F1 0.751, matched-event rho 0.626 [0.594, 0.659], and quartile-ranking accuracy 0.905 [0.890, 0.919]. The event gate passes, but no 30-second detector feature does:",
        "",
        "| Waveform-only feature | rho [95% CI] | Gain drift | Decision |",
        "|---|---:|---:|---|",
    ]
    for row in wave_rows:
        lines.append(f"| `{row['feature']}` | {interval(row)} | {row['max_abs_gain6_drift']:.2e} | reject |")
    lines += [
        "",
        "The result supports a narrow claim: isolated piano attack strength is measurable from audio. It does not support the broader claim that dynamics variety over a 30-second polyphonic passage is currently estimated faithfully enough for AI-music detection.",
        "",
        "## Benchmark C: section and bar-length recovery",
        "",
        "Forty-four SALAMI tracks with 68 independent annotation files were selected from the official Internet Archive mapping before inference. All-In-One used the eight released Harmonix checkpoints through a recorded Blackwell-compatible pure-PyTorch attention implementation; Beat This! `final0` supplied downbeats. No functional section labels were scored.",
        "",
        f"Boundary F1@0.5s is {structure['boundary_metrics']['f1_0p5s']['estimate']:.3f} [{structure['boundary_metrics']['f1_0p5s']['ci_low']:.3f}, {structure['boundary_metrics']['f1_0p5s']['ci_high']:.3f}]. Boundary F1@3s is {interval(boundary)}, below both the 0.60 point threshold and 0.55 lower-bound threshold. Because the boundary gate fails, no downstream section-length feature can be admitted; their direct correlations also remain well below rho 0.70.",
        "",
        "| Candidate summary | rho [95% CI] | Decision |",
        "|---|---:|---|",
    ]
    for row in struct_rows:
        lines.append(f"| `{row['feature']}` | {interval(row)} | reject |")
    lines += [
        "",
        "## Applied-filter decision",
        "",
        "The preregistered gate admits zero waveform-deployable features. The existing AI/Human development set was therefore not used to choose among these candidates, and the already examined external AI/Human test was not touched. Running an AI/Human ablation after seeing these failures would convert the detector data into a rescue-tuning set and inflate the evidence. The appropriate next experiment is to redesign one estimator family, freeze it, and test it on a new external benchmark before any detector application.",
        "",
        "The most promising redesign is the dynamics family because its event-level waveform result is strong. Candidate changes should target aggregation rather than event estimation: instrument-aware clustering, source-separated non-vocal sub-stems, and bar-synchronous robust summaries. These are new hypotheses and require a new held-out benchmark.",
        "",
        "## Reproducibility and artifact map",
        "",
        "- Protocol: `EXPERIMENT_PROTOCOL.md`; post-protocol deployment validation: `DEPLOYMENT_VALIDATION_ADDENDUM.md`.",
        "- Dataset/model provenance: `DATASET_MODEL_AUDIT.md`.",
        "- Exact commands and recovery history: `REPRODUCE_COMMANDS.md` and `RUN_LOG.md`.",
        "- Selection and transport verification: `manifests/`.",
        "- Executed analysis code: `code/`.",
        "- Direct numerical outputs: `results/rhythm_benchmark/`, `results/maestro_dynamics/`, `results/maestro_waveform_dynamics/`, and `results/salami_structure/`.",
        "- Aggregate machine-readable decision: `results/aggregate_summary.json` and `results/admission_decisions.csv`.",
        "- Full remote audio/checkpoint/runtime archive: `/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903`.",
        "",
        "Primary upstream resources: [Beat This!](https://github.com/CPJKU/beat_this), [Beat This! annotations](https://github.com/CPJKU/beat_this_annotations), [MAESTRO](https://magenta.withgoogle.com/datasets/maestro), [SALAMI](https://github.com/DDMAL/salami-data-public), and [All-In-One](https://github.com/mir-aidj/all-in-one).",
        "",
        "## Direct result hashes",
        "",
    ]
    for path in (rhythm_path, oracle_path, waveform_path, structure_path, decision_path, aggregate_path, figure_path):
        lines.append(f"- `{path.relative_to(ROOT)}`: `{sha256(path)}`")
    (REPORTS / "REPORT_EN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
