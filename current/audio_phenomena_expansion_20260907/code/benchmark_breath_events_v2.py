#!/usr/bin/env python3
"""Pre-fit v2 breath benchmark with conservative archive-identity exclusion.

The signal features, targets, event matching, classifier and thresholds are
imported unchanged from frozen v1.  V2 changes only acquisition admission and
the primary cohort: a filename/annotation singer prefix must agree with the
original ZIP singer directory.  The sole audited conflict is retained in the
acquisition/quarantine artifacts but excluded before any model fitting.
"""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

EXPECTED_CONFLICTS = {"m9_caro_vibrato.wav"}
EXPECTED_COUNTS = {
    "pre_identity_primary_clips": 114,
    "pre_identity_raw_high_medium_events": 146,
    "pre_identity_merged_high_medium_events": 144,
    "pre_identity_zero_event_clips": 38,
    "eligible_primary_clips": 113,
    "eligible_raw_high_medium_events": 143,
    "eligible_merged_high_medium_events": 141,
    "eligible_zero_event_clips": 38,
    "eligible_singers": 20,
}
SINGER_PREFIX = re.compile(r"^([fm])(\d+)_")
ARCHIVE_SINGER_DIR = re.compile(r"^(female|male)(\d+)$")


def load_json(path):
    def reject(value):
        raise ValueError(f"Non-standard/nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def filename_singer(name):
    match = SINGER_PREFIX.match(name)
    if not match:
        raise ValueError(f"Missing unambiguous singer prefix: {name}")
    return match.group(1) + match.group(2)


def expected_archive_singer(name):
    singer = filename_singer(name)
    return ("female" if singer.startswith("f") else "male") + singer[1:]


def physical_archive_singer(archive_member):
    matches = [part for part in PurePosixPath(archive_member).parts if ARCHIVE_SINGER_DIR.fullmatch(part)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one singer directory in archive member: {archive_member}")
    return matches[0]


def merge_high_medium(label):
    intervals = sorted(
        (float(event["start_sec"]), float(event["end_sec"]))
        for event in label.get("breath_events", [])
        if event.get("confidence") in ("high", "medium")
    )
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return intervals, merged


def cohort_counts(rows_and_labels, prefix):
    raw = merged = zero = 0
    singers = set()
    for row, label in rows_and_labels:
        intervals, merged_intervals = merge_high_medium(label)
        raw += len(intervals)
        merged += len(merged_intervals)
        zero += int(not merged_intervals)
        singers.add(row["singer"])
    return {
        f"{prefix}_clips": len(rows_and_labels),
        f"{prefix}_raw_high_medium_events": raw,
        f"{prefix}_merged_high_medium_events": merged,
        f"{prefix}_zero_event_clips": zero,
        f"{prefix}_singers": len(singers),
    }


def select_primary_cohort(
    manifest,
    labels_dir,
    expected_conflicts=EXPECTED_CONFLICTS,
    expected_counts=EXPECTED_COUNTS,
):
    labels_dir = Path(labels_dir)
    names = [row.get("filename") for row in manifest]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate acquisition filenames")
    primary_before_identity = []
    eligible = []
    excluded = []
    conflicts = []
    for row in manifest:
        name = row["filename"]
        singer = filename_singer(name)
        if row.get("singer") != singer:
            raise ValueError(f"Manifest singer does not match filename prefix: {name}")
        expected_physical = expected_archive_singer(name)
        actual_physical = physical_archive_singer(row["archive_member"])
        conflict = actual_physical != expected_physical
        if conflict:
            conflicts.append({
                "filename": name,
                "filename_singer": singer,
                "expected_archive_singer_dir": expected_physical,
                "actual_archive_singer_dir": actual_physical,
                "archive_member": row["archive_member"],
                "rule": "exclude_before_fit_filename_annotation_prefix_vs_original_zip_directory_conflict",
            })
        label_path = labels_dir / "labels" / "vocalset" / (Path(name).stem + ".breath.json")
        label = load_json(label_path)
        if label.get("audio_file") != name:
            raise ValueError(f"Annotation identity mismatch: {name}")
        labeler = label.get("labeler")
        labeler = labeler.strip() if isinstance(labeler, str) else ""
        review_time = label.get("review_time_sec")
        named_primary = (
            bool(labeler)
            and isinstance(review_time, (int, float))
            and not isinstance(review_time, bool)
            and math.isfinite(review_time)
            and review_time > 0
        )
        if not named_primary:
            excluded.append({"filename": name, "reason": "No named reviewer or positive finite review time"})
            continue
        primary_before_identity.append((row, label))
        if conflict:
            excluded.append({
                "filename": name,
                "reason": "Filename/annotation singer prefix conflicts with original ZIP singer directory",
                "expected_archive_singer_dir": expected_physical,
                "actual_archive_singer_dir": actual_physical,
                "archive_member": row["archive_member"],
            })
        else:
            eligible.append((row, label))

    conflict_names = {record["filename"] for record in conflicts}
    if conflict_names != set(expected_conflicts):
        raise ValueError(
            f"Physical identity conflict set changed: observed={sorted(conflict_names)}, "
            f"expected={sorted(expected_conflicts)}"
        )
    observed_counts = {
        **cohort_counts(primary_before_identity, "pre_identity_primary"),
        **cohort_counts(eligible, "eligible_primary"),
    }
    # Normalize keys to the prospective contract names.
    observed_counts = {
        "pre_identity_primary_clips": observed_counts["pre_identity_primary_clips"],
        "pre_identity_raw_high_medium_events": observed_counts["pre_identity_primary_raw_high_medium_events"],
        "pre_identity_merged_high_medium_events": observed_counts["pre_identity_primary_merged_high_medium_events"],
        "pre_identity_zero_event_clips": observed_counts["pre_identity_primary_zero_event_clips"],
        "eligible_primary_clips": observed_counts["eligible_primary_clips"],
        "eligible_raw_high_medium_events": observed_counts["eligible_primary_raw_high_medium_events"],
        "eligible_merged_high_medium_events": observed_counts["eligible_primary_merged_high_medium_events"],
        "eligible_zero_event_clips": observed_counts["eligible_primary_zero_event_clips"],
        "eligible_singers": observed_counts["eligible_primary_singers"],
    }
    if observed_counts != dict(expected_counts):
        raise ValueError(f"Primary cohort counts changed: observed={observed_counts}, expected={expected_counts}")
    return [row for row, _ in eligible], excluded, {
        "rule": "named reviewer + positive finite review time + filename/annotation singer prefix agrees with original ZIP singer directory",
        "conflicts": conflicts,
        "expected_conflict_names": sorted(expected_conflicts),
        "counts": observed_counts,
        "conflict_audio_retained_in_acquisition": True,
        "conflict_excluded_before_any_feature_fit": True,
        "singer_identity_verified_for_excluded_clip": False,
    }


def main():
    # Keep the cohort-admission audit importable in lightweight local test
    # environments; signal/model dependencies are required only for a real run.
    import numpy as np
    import scipy
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    import benchmark_breath_events as v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary-name", default="acquisition_summary_v2.json")
    parser.add_argument("--independent-audit", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Refusing existing output; preserve all receipts")

    acquisition_path = args.acquisition_dir / args.summary_name
    acquisition = load_json(acquisition_path)
    if not (
        acquisition.get("status") == "complete_accounting"
        and acquisition.get("selected") == 244
        and acquisition.get("downloaded_verified") == 244
        and acquisition.get("errors") == []
        and len(acquisition.get("manifest", [])) == 244
        and acquisition.get("whole_archive_hash_verified") is False
    ):
        raise ValueError("V2 benchmark requires clean 244/244 acquisition accounting")
    independent_audit_path = args.independent_audit or args.acquisition_dir / "independent_audit.json"
    independent_audit = load_json(independent_audit_path)
    if not (
        independent_audit.get("gate_passed") is True
        and independent_audit.get("errors") == []
        and independent_audit.get("acquisition_summary_sha256") == v1.sha(acquisition_path)
    ):
        raise ValueError("Independent acquisition audit is absent, failed, or targets another summary")

    selected, excluded, cohort_audit = select_primary_cohort(acquisition["manifest"], args.labels_dir)
    args.output_dir.mkdir(parents=True)
    config = {
        "benchmark_version": "breath-events-v2-physical-identity-conservative",
        "seed": v1.SEED,
        "code_sha256": v1.sha(__file__),
        "frozen_v1_dependency_sha256": v1.sha(Path(v1.__file__)),
        "protocol_sha256": v1.sha(args.protocol),
        "acquisition_summary_path": str(acquisition_path.resolve()),
        "acquisition_summary_sha256": v1.sha(acquisition_path),
        "independent_audit_path": str(independent_audit_path.resolve()),
        "independent_audit_sha256": v1.sha(independent_audit_path),
        "selected_primary_files": [row["filename"] for row in selected],
        "excluded": excluded,
        "primary_subset_audit": cohort_audit,
        "frame_length": v1.NFFT,
        "hop_length": v1.HOP,
        "sample_rate": v1.SR,
        "features": v1.FEATURE_NAMES,
        "classifier": {"name": "LogisticRegression", "C": 1, "class_weight": "balanced", "max_iter": 1000, "threshold": .5},
        "gate": {"event_f1": .70, "precision": .70},
        "no_ai_human_labels": True,
        "runtime": {
            "numpy": np.__version__,
            "librosa": v1.librosa.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "soundfile": v1.sf.__version__,
        },
    }
    v1.write_json(args.output_dir / "frozen_config.json", config)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        clips = list(pool.map(v1.extract, [(row, args.labels_dir, args.acquisition_dir / "audio") for row in selected]))
    (args.output_dir / "frames").mkdir()
    for clip in clips:
        np.savez_compressed(args.output_dir / "frames" / (clip["name"] + ".npz"), **{
            key: clip[key] for key in ("features", "times", "truth", "valid", "hard_negative")
        })

    records, models = [], []
    for singer in sorted({clip["singer"] for clip in clips}):
        train_x, train_y = [], []
        for clip in clips:
            if clip["singer"] == singer:
                continue
            positive = np.flatnonzero(clip["valid"] & (clip["truth"] == 1))
            negative = np.flatnonzero(clip["valid"] & (clip["truth"] == 0))
            if len(negative) > 2000:
                rng = np.random.default_rng(
                    v1.SEED + int(hashlib.sha256(clip["name"].encode()).hexdigest()[:8], 16)
                )
                negative = rng.choice(negative, 2000, replace=False)
            use = np.r_[positive, negative]
            train_x.append(clip["features"][use])
            train_y.append(clip["truth"][use])
        x, y = np.concatenate(train_x), np.concatenate(train_y)
        if len(np.unique(y)) != 2:
            raise ValueError("Training fold lacks both external event classes")
        scale = StandardScaler().fit(x)
        model = LogisticRegression(
            C=1, class_weight="balanced", max_iter=1000, random_state=v1.SEED
        ).fit(scale.transform(x), y)
        models.append({
            "heldout_singer": singer,
            "mean": scale.mean_.tolist(),
            "scale": scale.scale_.tolist(),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "iterations": model.n_iter_.tolist(),
            "training_frames": len(y),
            "positive_training_frames": int(y.sum()),
        })
        for clip in clips:
            if clip["singer"] != singer:
                continue
            score = model.predict_proba(scale.transform(clip["features"]))[:, 1]
            predicted = v1.predicted_events(score, clip["valid"], clip["times"])
            tp, fp, fn = v1.match_events(clip["events"], predicted)
            valid = clip["valid"]
            ba = balanced_accuracy_score(clip["truth"][valid], score[valid] >= .5) if len(np.unique(clip["truth"][valid])) == 2 else None
            records.append({
                "filename": clip["name"],
                "singer": singer,
                **v1.event_metrics(tp, fp, fn),
                "frame_ba": ba,
                "duration": clip["duration"],
                "ignored_seconds": float((~valid).sum() * .01),
                "valid_frames": int(valid.sum()),
                "positive_valid_frames": int(clip["truth"][valid].sum()),
                "tp_frames": int(((score >= .5) & valid & (clip["truth"] == 1)).sum()),
                "tn_frames": int(((score < .5) & valid & (clip["truth"] == 0)).sum()),
                "hard_negative_frames": int(clip["hard_negative"].sum()),
                "false_positive_hard_negative_frames": int(((score >= .5) & clip["hard_negative"]).sum()),
                "reference_events": clip["events"],
                "predicted_events": predicted,
            })
            np.savez_compressed(args.output_dir / "frames" / (clip["name"] + ".predictions.npz"), score=score)
        print(json.dumps({"singer_finished": singer, "clips_finished": len(records)}), flush=True)

    totals = v1.event_metrics(*[sum(row[key] for row in records) for key in ("tp", "fp", "fn")])
    positives = sum(row["positive_valid_frames"] for row in records)
    negatives = sum(row["valid_frames"] - row["positive_valid_frames"] for row in records)
    frame_ba = .5 * (
        sum(row["tp_frames"] for row in records) / positives
        + sum(row["tn_frames"] for row in records) / negatives
    )
    summary = {
        "status": "completed",
        "benchmark_version": config["benchmark_version"],
        "primary_clips": len(records),
        "singers": len(models),
        "overall": totals,
        "strict_positive_events_after_overlap_merge": sum(len(row["reference_events"]) for row in records),
        "no_event_primary_clips": sum(not row["reference_events"] for row in records),
        "ignored_primary_seconds": sum(row["ignored_seconds"] for row in records),
        "primary_duration_seconds": sum(row["duration"] for row in records),
        "aggregate_frame_balanced_accuracy": frame_ba,
        "excluded_acquisition_clip_count": len(excluded),
        "physical_identity_conflict_exclusions": cohort_audit["conflicts"],
        "hard_negative_frames": sum(row["hard_negative_frames"] for row in records),
        "false_positive_hard_negative_frames": sum(row["false_positive_hard_negative_frames"] for row in records),
        "gate_passed": (
            len(records) == EXPECTED_COUNTS["eligible_primary_clips"]
            and (totals["f1"] or 0) >= .7
            and (totals["precision"] or 0) >= .7
        ),
        "validates_ai_human_classification": False,
        "excluded": excluded,
        "acquisition_errors": acquisition["errors"],
        "per_singer": {
            singer: v1.event_metrics(*[
                sum(row[key] for row in records if row["singer"] == singer)
                for key in ("tp", "fp", "fn")
            ])
            for singer in sorted({clip["singer"] for clip in clips})
        },
    }
    v1.write_json(args.output_dir / "event_predictions.json", records)
    v1.write_json(args.output_dir / "fold_models.json", models)
    v1.write_json(args.output_dir / "summary.json", summary)
    v1.write_json(args.output_dir / "artifact_sha256.json", {
        str(path.relative_to(args.output_dir)): v1.sha(path)
        for path in sorted(args.output_dir.rglob("*")) if path.is_file()
    })
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
