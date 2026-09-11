#!/usr/bin/env python3
"""Independently audit frozen external breath-event result artifacts.

No feature extraction, model fitting, threshold selection, or prediction is
performed. Event matches and aggregate metrics are recomputed from the stored
reference/predicted intervals using an independent bipartite matcher.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys


AMBIGUOUS = "m9_caro_vibrato.wav"
EXPECTED = {"clips": 113, "singers": 20, "events": 141, "zero_event_clips": 38}


def load_json(path):
    def reject(value):
        raise ValueError(f"Non-standard/nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_event_matches(reference, prediction):
    edges = []
    for ref_start, ref_end in reference:
        compatible = []
        for index, (pred_start, pred_end) in enumerate(prediction):
            if (
                pred_end - pred_start <= 2.0 + 1e-8
                and abs(ref_start - pred_start) <= .2 + 1e-9
                and min(ref_end, pred_end) > max(ref_start, pred_start)
            ):
                compatible.append(index)
        edges.append(compatible)
    assigned_reference = {}

    def augment(reference_index, seen):
        for prediction_index in edges[reference_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            if prediction_index not in assigned_reference or augment(assigned_reference[prediction_index], seen):
                assigned_reference[prediction_index] = reference_index
                return True
        return False

    matches = sum(augment(index, set()) for index in range(len(reference)))
    return matches, len(prediction) - matches, len(reference) - matches


def event_metrics(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def same_number(left, right, tolerance=1e-12):
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def all_finite(value):
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit(results_dir, project_root):
    results_dir = Path(results_dir).resolve()
    project_root = Path(project_root).resolve()
    config = load_json(results_dir / "frozen_config.json")
    summary = load_json(results_dir / "summary.json")
    records = load_json(results_dir / "event_predictions.json")
    models = load_json(results_dir / "fold_models.json")
    declared_hashes = load_json(results_dir / "artifact_sha256.json")
    errors = []

    def check(condition, name, detail=""):
        if not condition:
            errors.append({"check": name, "detail": str(detail)})

    actual_files = {
        str(path.relative_to(results_dir))
        for path in results_dir.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.json"
    }
    check(set(declared_hashes) == actual_files, "artifact_manifest_file_set", {
        "missing_from_manifest": sorted(actual_files - set(declared_hashes)),
        "missing_on_disk": sorted(set(declared_hashes) - actual_files),
    })
    hash_mismatches = {
        relative: {"declared": expected_hash, "actual": sha256(results_dir / relative)}
        for relative, expected_hash in declared_hashes.items()
        if (results_dir / relative).is_file() and sha256(results_dir / relative) != expected_hash
    }
    check(not hash_mismatches, "artifact_sha256", hash_mismatches)

    source_hashes = {
        "code_sha256": project_root / "code" / "benchmark_breath_events_v2.py",
        "frozen_v1_dependency_sha256": project_root / "code" / "benchmark_breath_events.py",
        "protocol_sha256": project_root / "BREATH_BENCHMARK_PROTOCOL_EN.md",
        "acquisition_summary_sha256": project_root / "external_validation" / "vocalset_breath_original" / "acquisition_summary_v2.json",
        "independent_audit_sha256": project_root / "external_validation" / "vocalset_breath_original" / "independent_audit.json",
    }
    source_hash_results = {}
    for field, path in source_hashes.items():
        actual = sha256(path)
        declared = config.get(field)
        source_hash_results[field] = {"path": str(path), "declared": declared, "actual": actual, "matches": declared == actual}
        check(declared == actual, "source_hash_" + field, source_hash_results[field])

    names = [record["filename"] for record in records]
    selected = config.get("selected_primary_files", [])
    frame_inputs = {
        path.name[:-4]
        for path in (results_dir / "frames").glob("*.wav.npz")
        if not path.name.endswith(".predictions.npz")
    }
    frame_predictions = {
        path.name[:-len(".predictions.npz")]
        for path in (results_dir / "frames").glob("*.wav.predictions.npz")
    }
    check(len(records) == EXPECTED["clips"], "record_count", len(records))
    check(len(set(names)) == len(names), "unique_record_names")
    check(set(names) == set(selected) == frame_inputs == frame_predictions, "selected_record_frame_identity_sets")
    check(AMBIGUOUS not in names and AMBIGUOUS not in selected and AMBIGUOUS not in frame_inputs, "ambiguous_clip_absent_all_folds")
    conflict_exclusions = [item for item in config.get("excluded", []) if item.get("filename") == AMBIGUOUS]
    check(len(conflict_exclusions) == 1 and "conflicts" in conflict_exclusions[0].get("reason", ""), "ambiguous_clip_explicit_exclusion", conflict_exclusions)

    total_tp = total_fp = total_fn = 0
    event_count = zero_event_clips = 0
    zero_event_fp = zero_event_clips_with_fp = 0
    hard_negative_frames = hard_negative_false_positive_frames = 0
    valid_frames = positive_frames = tp_frames = tn_frames = 0
    recomputed_per_singer = collections.defaultdict(lambda: [0, 0, 0])
    row_metric_mismatches = []
    for record in records:
        tp, fp, fn = maximum_event_matches(record["reference_events"], record["predicted_events"])
        for key, actual in zip(("tp", "fp", "fn"), (tp, fp, fn)):
            if record.get(key) != actual:
                row_metric_mismatches.append({"filename": record["filename"], "field": key, "stored": record.get(key), "recomputed": actual})
        metrics = event_metrics(tp, fp, fn)
        for key in ("precision", "recall", "f1"):
            if not same_number(record.get(key), metrics[key]):
                row_metric_mismatches.append({"filename": record["filename"], "field": key, "stored": record.get(key), "recomputed": metrics[key]})
        total_tp += tp
        total_fp += fp
        total_fn += fn
        recomputed_per_singer[record["singer"]][0] += tp
        recomputed_per_singer[record["singer"]][1] += fp
        recomputed_per_singer[record["singer"]][2] += fn
        event_count += len(record["reference_events"])
        if not record["reference_events"]:
            zero_event_clips += 1
            zero_event_fp += fp
            zero_event_clips_with_fp += int(fp > 0)
        hard_negative_frames += record["hard_negative_frames"]
        hard_negative_false_positive_frames += record["false_positive_hard_negative_frames"]
        valid_frames += record["valid_frames"]
        positive_frames += record["positive_valid_frames"]
        tp_frames += record["tp_frames"]
        tn_frames += record["tn_frames"]
    check(not row_metric_mismatches, "per_clip_event_recomputation", row_metric_mismatches)
    overall = event_metrics(total_tp, total_fp, total_fn)
    check(event_count == EXPECTED["events"], "reference_event_count", event_count)
    check(zero_event_clips == EXPECTED["zero_event_clips"], "zero_event_clip_count", zero_event_clips)
    singers = sorted(recomputed_per_singer)
    check(len(singers) == EXPECTED["singers"], "singer_count", len(singers))
    for key, value in overall.items():
        check(same_number(summary["overall"][key], value), "summary_overall_" + key, {"stored": summary["overall"][key], "recomputed": value})
    aggregate_frame_ba = .5 * (
        tp_frames / positive_frames + tn_frames / (valid_frames - positive_frames)
    )
    check(same_number(summary["aggregate_frame_balanced_accuracy"], aggregate_frame_ba), "aggregate_frame_balanced_accuracy")
    check(summary["hard_negative_frames"] == hard_negative_frames, "hard_negative_frames")
    check(summary["false_positive_hard_negative_frames"] == hard_negative_false_positive_frames, "hard_negative_false_positive_frames")
    stored_per_singer = summary.get("per_singer", {})
    per_singer_mismatches = []
    for singer, counts in recomputed_per_singer.items():
        metrics = event_metrics(*counts)
        for key, value in metrics.items():
            if not same_number(stored_per_singer.get(singer, {}).get(key), value):
                per_singer_mismatches.append({"singer": singer, "field": key, "stored": stored_per_singer.get(singer, {}).get(key), "recomputed": value})
    check(not per_singer_mismatches and set(stored_per_singer) == set(singers), "per_singer_recomputation", per_singer_mismatches)

    feature_count = len(config.get("features", []))
    heldout = [model.get("heldout_singer") for model in models]
    model_errors = []
    for model in models:
        singer = model.get("heldout_singer")
        if not (
            len(model.get("mean", [])) == feature_count
            and len(model.get("scale", [])) == feature_count
            and len(model.get("coef", [])) == 1
            and len(model.get("coef", [[]])[0]) == feature_count
            and len(model.get("intercept", [])) == 1
            and len(model.get("iterations", [])) == 1
            and all_finite(model["mean"])
            and all_finite(model["scale"])
            and all(value > 0 for value in model["scale"])
            and all_finite(model["coef"])
            and all_finite(model["intercept"])
            and 0 < model["iterations"][0] <= config["classifier"]["max_iter"]
            and 0 < model["positive_training_frames"] < model["training_frames"]
        ):
            model_errors.append(singer)
    check(len(models) == EXPECTED["singers"] and len(set(heldout)) == len(models) and set(heldout) == set(singers), "fold_model_identity", heldout)
    check(not model_errors, "fold_model_coefficients", model_errors)

    gate = config["gate"]
    recomputed_gate = (
        not errors
        and overall["f1"] >= gate["event_f1"]
        and overall["precision"] >= gate["precision"]
    )
    # Gate consistency excludes this final stored-value check itself.
    check(summary.get("gate_passed") == recomputed_gate, "stored_gate_decision", {"stored": summary.get("gate_passed"), "recomputed": recomputed_gate})

    return {
        "schema": "external-breath-events-v2-independent-result-audit-v1",
        "status": "passed_artifact_audit" if not errors else "failed_artifact_audit",
        "artifact_audit_passed": not errors,
        "measurement_admission_gate_passed": recomputed_gate,
        "no_refit_or_rescoring": True,
        "expected_cohort": EXPECTED,
        "observed_cohort": {
            "clips": len(records),
            "singers": len(singers),
            "reference_events": event_count,
            "zero_event_clips": zero_event_clips,
            "ambiguous_clip_absent": AMBIGUOUS not in names,
        },
        "recomputed_overall": overall,
        "recomputed_event_predictions": total_tp + total_fp,
        "zero_event_error": {
            "clips": zero_event_clips,
            "clips_with_false_positive": zero_event_clips_with_fp,
            "false_positive_events": zero_event_fp,
            "clip_false_positive_rate": zero_event_clips_with_fp / zero_event_clips,
        },
        "hard_negative_error": {
            "frames": hard_negative_frames,
            "false_positive_frames": hard_negative_false_positive_frames,
            "false_positive_fraction": hard_negative_false_positive_frames / hard_negative_frames,
        },
        "recomputed_aggregate_frame_balanced_accuracy": aggregate_frame_ba,
        "fold_models": {
            "count": len(models),
            "feature_count": feature_count,
            "heldout_singers": sorted(heldout),
            "coefficient_shapes_valid": not model_errors,
            "training_frames_min": min(model["training_frames"] for model in models),
            "training_frames_max": max(model["training_frames"] for model in models),
            "positive_training_frames_min": min(model["positive_training_frames"] for model in models),
            "positive_training_frames_max": max(model["positive_training_frames"] for model in models),
            "iterations_min": min(model["iterations"][0] for model in models),
            "iterations_max": max(model["iterations"][0] for model in models),
        },
        "runtime": config["runtime"],
        "source_hashes": source_hash_results,
        "declared_artifact_hash_count": len(declared_hashes),
        "artifact_hash_mismatches": hash_mismatches,
        "gate": gate,
        "errors": errors,
        "audit_code_sha256": sha256(__file__),
        "audit_runtime": {"python": sys.version.split()[0]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    result = audit(args.results_dir, args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "artifact_audit_passed": result["artifact_audit_passed"],
        "measurement_admission_gate_passed": result["measurement_admission_gate_passed"],
        "output": str(args.output),
    }))
    if not result["artifact_audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
