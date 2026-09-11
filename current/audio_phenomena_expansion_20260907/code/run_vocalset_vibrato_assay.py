#!/usr/bin/env python3
"""Resumable CPU pYIN runner for the preregistered VocalSet V assay.

The development and evaluation roles are gated separately.  The current
development workflow maps and opens only development files.  Evaluation needs
an explicit held-out authorization and a distinct frozen receipt.
"""

from __future__ import annotations

import argparse
import collections
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Iterable, Mapping


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np

from benchmark_pitch_tracking import PYIN_CONFIG, estimate_pyin
from vocalset_vibrato_features import CONFIG as PRESENCE_CONFIG
from vocalset_vibrato_features import PRIMARY_FEATURE, extract_vibrato_presence


VERSION = "v1-development-before-any-heldout-scoring"
MAX_WORKERS = 4
EXPECTED_PAIRS = {"development": 43, "evaluation": 47}
EXPECTED_CLIPS = {"development": 86, "evaluation": 94}
REQUIRED_LIBROSA = "0.11.0"
EVALUATION_AUTHORIZATION = "root_approved_heldout_evaluation"

PREPROCESS_CONFIG: dict[str, Any] = {
    "source_sample_rate_hz": 44100,
    "source_channels": 1,
    "source_read_dtype": "float32",
    "channel_policy": "require original mono; no folding",
    "dc_removal": False,
    "amplitude_normalization": False,
    "target_sample_rate_hz": 16000,
    "resampler": "scipy.signal.resample_poly",
    "resample_up": 160,
    "resample_down": 441,
    "resample_window": ["kaiser", 5.0],
    "resample_padtype": "constant",
    "analysis_dtype": "float32",
    "audio_extent": "entire verified source file; no crop and no padding",
}

DEVELOPMENT_GATE = {
    "required_role": "development",
    "required_pairs": 43,
    "required_clips": 86,
    "required_failures": 0,
    "minimum_numeric_coverage_each_technique": 0.75,
    "maximum_absolute_coverage_difference": 0.10,
    "minimum_paired_coverage_each_context": 0.60,
    "require_positive_median_paired_difference": True,
    "minimum_strictly_positive_pair_fraction": 0.65,
    "ties_count_as_nonpositive": True,
}


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("librosa", "numpy", "scipy", "soundfile", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def frozen_config() -> dict[str, Any]:
    return {
        "version": VERSION,
        "preprocessing": PREPROCESS_CONFIG,
        "pyin": PYIN_CONFIG,
        "presence": PRESENCE_CONFIG,
        "primary_feature": PRIMARY_FEATURE,
        "development_gate": DEVELOPMENT_GATE,
        "heldout_singer_gate": {
            "minimum_contributing_singers": 8,
            "minimum_valid_pairs_per_contributing_singer": 2,
            "required_positive_singers": "ceil(0.9 * contributing_singer_count)",
            "not_executed_in_development_run": True,
        },
    }


def load_pair_rows(pair_manifest: Path) -> list[dict[str, str]]:
    with pair_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pair_id",
        "split",
        "singer",
        "context",
        "straight_filename",
        "straight_audio_sha256",
        "straight_sample_rate_hz",
        "straight_frames",
        "vibrato_filename",
        "vibrato_audio_sha256",
        "vibrato_sample_rate_hz",
        "vibrato_frames",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"pair manifest lacks required fields: {sorted(required)}")
    pair_ids = [row["pair_id"] for row in rows]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair manifest pair_id values must be nonempty and unique")
    return rows


def select_role_rows(rows: list[dict[str, str]], role: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == role]
    if len(selected) != EXPECTED_PAIRS[role]:
        raise ValueError(
            f"{role} manifest has {len(selected)} pairs, expected {EXPECTED_PAIRS[role]}"
        )
    other_role = "evaluation" if role == "development" else "development"
    if {row["singer"] for row in selected} & {
        row["singer"] for row in rows if row["split"] == other_role
    }:
        raise ValueError("singer leakage across manifest roles")
    return selected


def clip_specs(role_rows: list[dict[str, str]], source_root: Path) -> list[dict[str, Any]]:
    """Map only the already-selected role to physical files."""

    specs = []
    for row in role_rows:
        for technique in ("straight", "vibrato"):
            filename = row[f"{technique}_filename"]
            if Path(filename).name != filename:
                raise ValueError(f"unsafe/non-basename filename: {filename!r}")
            specs.append(
                {
                    "clip_id": f"{row['pair_id']}::{technique}",
                    "pair_id": row["pair_id"],
                    "role": row["split"],
                    "singer": row["singer"],
                    "context": row["context"],
                    "content_id": row["content_id"],
                    "technique": technique,
                    "filename": filename,
                    "source_path": str((source_root / filename).resolve()),
                    "expected_source_sha256": row[f"{technique}_audio_sha256"],
                    "expected_sample_rate_hz": int(row[f"{technique}_sample_rate_hz"]),
                    "expected_frames": int(row[f"{technique}_frames"]),
                }
            )
    if len(specs) != EXPECTED_CLIPS[role_rows[0]["split"]]:
        raise ValueError("unexpected role clip count")
    if len({spec["clip_id"] for spec in specs}) != len(specs):
        raise ValueError("duplicate clip IDs in selected role")
    if len({spec["filename"] for spec in specs}) != len(specs):
        raise ValueError("source file reused within selected role")
    return specs


def item_key(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()


def item_receipt_path(output_dir: Path, clip_id: str) -> Path:
    return output_dir / "receipts" / f"{item_key(clip_id)}.json"


def contour_path(output_dir: Path, clip_id: str) -> Path:
    return output_dir / "contours" / f"{item_key(clip_id)}.npz"


def _source_inventory(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    import soundfile as sf

    inventory = []
    for spec in specs:
        source_path = Path(spec["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        actual_hash = file_sha256(source_path)
        if actual_hash != spec["expected_source_sha256"]:
            raise ValueError(f"source SHA mismatch for {spec['filename']}")
        info = sf.info(source_path)
        if (
            info.samplerate != spec["expected_sample_rate_hz"]
            or info.frames != spec["expected_frames"]
            or info.channels != PREPROCESS_CONFIG["source_channels"]
            or info.samplerate != PREPROCESS_CONFIG["source_sample_rate_hz"]
        ):
            raise ValueError(f"source physical metadata mismatch for {spec['filename']}")
        inventory.append(
            {
                "clip_id": spec["clip_id"],
                "filename": spec["filename"],
                "source_path": str(source_path),
                "source_sha256": actual_hash,
                "sample_rate_hz": int(info.samplerate),
                "frames": int(info.frames),
                "channels": int(info.channels),
            }
        )
    return inventory


def prepare_review_receipt(args: argparse.Namespace) -> int:
    if args.role == "evaluation" and args.review_decision != EVALUATION_AUTHORIZATION:
        raise SystemExit("held-out preparation requires explicit root held-out authorization")
    versions = package_versions()
    if versions["librosa"] != REQUIRED_LIBROSA:
        raise SystemExit(
            f"requires librosa {REQUIRED_LIBROSA}, observed {versions['librosa']}"
        )
    rows = load_pair_rows(args.pair_manifest)
    role_rows = select_role_rows(rows, args.role)
    specs = clip_specs(role_rows, args.source_root)
    inventory = _source_inventory(specs)
    code_path = Path(__file__).resolve()
    feature_path = code_path.with_name("vocalset_vibrato_features.py")
    benchmark_path = code_path.with_name("benchmark_pitch_tracking.py")
    config = frozen_config()
    receipt = {
        "schema": "vocalset-vibrato-root-review-receipt-v1",
        "status": f"frozen_before_{args.role}",
        "review_decision": args.review_decision,
        "role": args.role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
        "pair_manifest_path": str(args.pair_manifest.resolve()),
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "runner_code_path": str(code_path),
        "runner_code_sha256": file_sha256(code_path),
        "presence_code_path": str(feature_path),
        "presence_code_sha256": file_sha256(feature_path),
        "prior_pyin_code_path": str(benchmark_path),
        "prior_pyin_code_sha256": file_sha256(benchmark_path),
        "frozen_config": config,
        "frozen_config_sha256": canonical_hash(config),
        "expected_pairs": EXPECTED_PAIRS[args.role],
        "expected_clips": EXPECTED_CLIPS[args.role],
        "selected_pair_id_set_sha256": hashlib.sha256(
            ("\n".join(sorted(row["pair_id"] for row in role_rows)) + "\n").encode()
        ).hexdigest(),
        "selected_source_inventory": inventory,
        "selected_source_inventory_sha256": canonical_hash(inventory),
        "heldout_guard": {
            "other_role_pair_count_in_manifest": len(rows) - len(role_rows),
            "other_role_audio_paths_mapped_or_opened": False,
            "evaluation_requires_distinct_receipt_and_explicit_authorization": True,
        },
        "no_ai_human_labels": True,
        "no_model_fit": True,
        "no_scores_computed_during_prepare": True,
    }
    atomic_json(args.review_receipt, receipt)
    print(json.dumps(clean({key: receipt[key] for key in (
        "status", "role", "expected_pairs", "expected_clips", "hostname",
        "pair_manifest_sha256", "runner_code_sha256", "presence_code_sha256",
        "prior_pyin_code_sha256", "frozen_config_sha256",
        "selected_source_inventory_sha256", "heldout_guard",
    )}), indent=2, sort_keys=True))
    return 0


def verify_review_receipt(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = json.loads(args.review_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != f"frozen_before_{args.role}" or receipt.get("role") != args.role:
        raise ValueError("review receipt role/status mismatch")
    if receipt.get("review_decision") != args.review_decision:
        raise ValueError("review decision token does not match frozen receipt")
    if args.role == "evaluation" and args.review_decision != EVALUATION_AUTHORIZATION:
        raise ValueError("held-out execution requires explicit root held-out authorization")
    code_path = Path(__file__).resolve()
    expected = {
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "runner_code_sha256": file_sha256(code_path),
        "presence_code_sha256": file_sha256(code_path.with_name("vocalset_vibrato_features.py")),
        "prior_pyin_code_sha256": file_sha256(code_path.with_name("benchmark_pitch_tracking.py")),
        "frozen_config_sha256": canonical_hash(frozen_config()),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"review receipt {key} mismatch")
    if receipt.get("packages") != package_versions():
        raise ValueError("runtime package versions changed since review receipt")
    rows = load_pair_rows(args.pair_manifest)
    specs = clip_specs(select_role_rows(rows, args.role), args.source_root)
    if canonical_hash(receipt.get("selected_source_inventory")) != receipt.get(
        "selected_source_inventory_sha256"
    ):
        raise ValueError("review receipt source inventory hash is invalid")
    by_clip = {row["clip_id"]: row for row in receipt["selected_source_inventory"]}
    if set(by_clip) != {spec["clip_id"] for spec in specs}:
        raise ValueError("review receipt source inventory clip set mismatch")
    for spec in specs:
        recorded = by_clip[spec["clip_id"]]
        if (
            recorded["source_path"] != spec["source_path"]
            or recorded["source_sha256"] != spec["expected_source_sha256"]
        ):
            raise ValueError(f"review receipt source mapping mismatch: {spec['clip_id']}")
    return receipt, specs


def preprocess_audio(spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import soundfile as sf
    from scipy.signal import resample_poly

    path = Path(spec["source_path"])
    source_hash = file_sha256(path)
    if source_hash != spec["expected_source_sha256"]:
        raise ValueError(f"source SHA mismatch for {spec['filename']}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != PREPROCESS_CONFIG["source_sample_rate_hz"]:
        raise ValueError(f"unexpected source sample rate for {spec['filename']}")
    if audio.shape != (spec["expected_frames"], PREPROCESS_CONFIG["source_channels"]):
        raise ValueError(f"decoded source shape mismatch for {spec['filename']}: {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"nonfinite source samples for {spec['filename']}")
    mono = audio[:, 0]
    resampled = resample_poly(
        mono,
        PREPROCESS_CONFIG["resample_up"],
        PREPROCESS_CONFIG["resample_down"],
        window=("kaiser", 5.0),
        padtype="constant",
    ).astype(np.float32, copy=False)
    expected_length = math.ceil(
        len(mono) * PREPROCESS_CONFIG["resample_up"] / PREPROCESS_CONFIG["resample_down"]
    )
    if len(resampled) != expected_length or not np.isfinite(resampled).all():
        raise ValueError(f"resampled waveform length/content invalid for {spec['filename']}")
    resampled = np.ascontiguousarray(resampled, dtype="<f4")
    return resampled, {
        "source_sha256": source_hash,
        "source_sample_rate_hz": int(sample_rate),
        "source_frames": int(len(mono)),
        "analysis_sample_rate_hz": PREPROCESS_CONFIG["target_sample_rate_hz"],
        "analysis_frames": int(len(resampled)),
        "analysis_waveform_sha256": hashlib.sha256(resampled.tobytes()).hexdigest(),
        "analysis_rms": float(np.sqrt(np.mean(np.square(resampled, dtype=np.float64)))),
    }


def process_clip(spec: dict[str, Any], contract_hash: str) -> dict[str, Any]:
    try:
        waveform, audio_audit = preprocess_audio(spec)
        time_sec, f0_hz, voiced, voiced_probability = estimate_pyin(waveform)
        features = extract_vibrato_presence(time_sec, f0_hz, voiced)
        window_details = features.pop("V_window_details")
        return {
            "status": "ok",
            **spec,
            "contract_hash": contract_hash,
            **audio_audit,
            **features,
            "window_details": window_details,
            "arrays": {
                "time_sec": time_sec,
                "f0_hz": f0_hz,
                "voiced": voiced,
                "voiced_probability": voiced_probability,
            },
        }
    except Exception as error:
        return {
            "status": "error",
            **spec,
            "contract_hash": contract_hash,
            "exception_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def _valid_cached_receipt(
    path: Path, output_dir: Path, spec: dict[str, Any], contract_hash: str
) -> bool:
    if not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        contour = contour_path(output_dir, spec["clip_id"])
        return bool(
            receipt.get("status") == "ok"
            and receipt.get("contract_hash") == contract_hash
            and receipt.get("clip_id") == spec["clip_id"]
            and receipt.get("source_sha256") == spec["expected_source_sha256"]
            and contour.is_file()
            and receipt.get("contour_sha256") == file_sha256(contour)
        )
    except Exception:
        return False


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field)) for field in fields})
    os.replace(temporary, path)


def consolidate(output_dir: Path, specs: list[dict[str, Any]], contract_hash: str, role: str) -> dict[str, Any]:
    receipts = []
    for spec in specs:
        path = item_receipt_path(output_dir, spec["clip_id"])
        if not path.is_file():
            receipts.append({"status": "missing", **spec, "contract_hash": contract_hash})
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("contract_hash") != contract_hash:
            raise ValueError(f"cached contract mismatch: {spec['clip_id']}")
        receipts.append(receipt)
    receipts.sort(key=lambda row: row["clip_id"])
    clip_fields = [
        "status", "clip_id", "pair_id", "role", "singer", "context", "content_id",
        "technique", "filename", "source_path", "source_sha256", "contour_path",
        "contour_sha256", "analysis_waveform_sha256", "source_sample_rate_hz",
        "source_frames", "analysis_sample_rate_hz", "analysis_frames", "analysis_rms",
        PRIMARY_FEATURE, "V_status", "V_missing_reason", "V_pitch_frame_coverage",
        "V_candidate_window_count", "V_eligible_stable_pitch_window_count",
        "V_transition_rejected_window_count", "V_periodic_window_count",
        "V_detected_rate_hz_median_conditional", "V_detected_extent_cents_median_conditional",
        "exception_type", "error",
    ]
    write_csv(output_dir / "per_clip_features.csv", receipts, clip_fields)

    by_pair: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for receipt in receipts:
        by_pair[receipt["pair_id"]][receipt["technique"]] = receipt
    pair_rows = []
    for pair_id, techniques in sorted(by_pair.items()):
        straight = techniques["straight"]
        vibrato = techniques["vibrato"]
        straight_value = straight.get(PRIMARY_FEATURE)
        vibrato_value = vibrato.get(PRIMARY_FEATURE)
        paired_valid = bool(
            straight["status"] == "ok"
            and vibrato["status"] == "ok"
            and straight.get("V_status") == "ok"
            and vibrato.get("V_status") == "ok"
            and isinstance(straight_value, (int, float))
            and isinstance(vibrato_value, (int, float))
        )
        difference = float(vibrato_value - straight_value) if paired_valid else math.nan
        pair_rows.append(
            {
                "pair_id": pair_id,
                "singer": straight["singer"],
                "context": straight["context"],
                "content_id": straight["content_id"],
                "straight_value": straight_value,
                "vibrato_value": vibrato_value,
                "paired_valid": int(paired_valid),
                "difference_vibrato_minus_straight": difference,
                "direction": (
                    "positive" if paired_valid and difference > 0
                    else "negative" if paired_valid and difference < 0
                    else "tie" if paired_valid
                    else "missing"
                ),
                "straight_status": straight.get("V_status", straight["status"]),
                "vibrato_status": vibrato.get("V_status", vibrato["status"]),
            }
        )
    pair_fields = [
        "pair_id", "singer", "context", "content_id", "straight_value", "vibrato_value",
        "paired_valid", "difference_vibrato_minus_straight", "direction",
        "straight_status", "vibrato_status",
    ]
    write_csv(output_dir / "per_pair_differences.csv", pair_rows, pair_fields)

    singer_rows = []
    for singer in sorted({row["singer"] for row in pair_rows}):
        singer_pairs = [row for row in pair_rows if row["singer"] == singer]
        valid = [row for row in singer_pairs if row["paired_valid"]]
        differences = [row["difference_vibrato_minus_straight"] for row in valid]
        singer_rows.append(
            {
                "singer": singer,
                "requested_pairs": len(singer_pairs),
                "paired_valid_pairs": len(valid),
                "positive_pairs": sum(value > 0 for value in differences),
                "tie_pairs": sum(value == 0 for value in differences),
                "negative_pairs": sum(value < 0 for value in differences),
                "mean_difference": float(np.mean(differences)) if differences else math.nan,
                "median_difference": float(np.median(differences)) if differences else math.nan,
            }
        )
    write_csv(
        output_dir / "per_singer_summary.csv",
        singer_rows,
        [
            "singer", "requested_pairs", "paired_valid_pairs", "positive_pairs", "tie_pairs",
            "negative_pairs", "mean_difference", "median_difference",
        ],
    )

    coverage = {}
    for technique in ("straight", "vibrato"):
        subset = [row for row in receipts if row["technique"] == technique]
        numeric = sum(row.get("V_status") == "ok" for row in subset)
        coverage[technique] = {"numeric": numeric, "total": len(subset), "fraction": numeric / len(subset)}
    context_coverage = {}
    for context in sorted({row["context"] for row in pair_rows}):
        subset = [row for row in pair_rows if row["context"] == context]
        valid = sum(row["paired_valid"] for row in subset)
        context_coverage[context] = {"paired_valid": valid, "total": len(subset), "fraction": valid / len(subset)}
    valid_pairs = [row for row in pair_rows if row["paired_valid"]]
    differences = np.asarray(
        [row["difference_vibrato_minus_straight"] for row in valid_pairs], dtype=np.float64
    )
    directions = collections.Counter(row["direction"] for row in pair_rows)
    checks = {
        "role_is_development": role == "development",
        "requested_pairs_eq_43": len(pair_rows) == DEVELOPMENT_GATE["required_pairs"],
        "requested_clips_eq_86": len(receipts) == DEVELOPMENT_GATE["required_clips"],
        "failures_eq_0": all(row["status"] == "ok" for row in receipts),
        "straight_numeric_coverage_ge_0_75": coverage["straight"]["fraction"] >= 0.75,
        "vibrato_numeric_coverage_ge_0_75": coverage["vibrato"]["fraction"] >= 0.75,
        "coverage_difference_le_0_10": abs(
            coverage["straight"]["fraction"] - coverage["vibrato"]["fraction"]
        ) <= 0.10,
        "each_context_paired_coverage_ge_0_60": all(
            value["fraction"] >= 0.60 for value in context_coverage.values()
        ),
        "median_paired_difference_gt_0": bool(len(differences) and np.median(differences) > 0),
        "strictly_positive_pair_fraction_ge_0_65": bool(
            len(differences) and np.sum(differences > 0) / len(differences) >= 0.65
        ),
    }
    gate_passed = role == "development" and all(checks.values())
    summary = {
        "schema": "vocalset-vibrato-development-summary-v1",
        "status": "complete" if all(row["status"] == "ok" for row in receipts) else "incomplete",
        "role": role,
        "contract_hash": contract_hash,
        "requested_pairs": len(pair_rows),
        "requested_clips": len(receipts),
        "status_counts": dict(collections.Counter(row["status"] for row in receipts)),
        "numeric_coverage_by_technique": coverage,
        "paired_coverage_by_context": context_coverage,
        "pair_direction_counts_including_missing": dict(directions),
        "paired_valid_pairs": len(valid_pairs),
        "paired_difference_median": float(np.median(differences)) if len(differences) else None,
        "strictly_positive_pair_fraction": float(np.sum(differences > 0) / len(differences)) if len(differences) else None,
        "per_singer_contributing_count_ge_2_pairs": sum(
            row["paired_valid_pairs"] >= 2 for row in singer_rows
        ),
        "development_gate": DEVELOPMENT_GATE,
        "development_gate_checks": checks,
        "development_gate_passed": gate_passed,
        "heldout_evaluation_opened_or_scored": False,
        "continuous_rate_or_extent_validated": False,
        "ai_detector_fitted": False,
        "output_sha256": {},
    }
    for path in (
        output_dir / "per_clip_features.csv",
        output_dir / "per_pair_differences.csv",
        output_dir / "per_singer_summary.csv",
    ):
        summary["output_sha256"][path.name] = file_sha256(path)
    atomic_json(output_dir / "summary.json", summary)
    return summary


def run_assay(args: argparse.Namespace) -> int:
    if not (1 <= args.workers <= MAX_WORKERS):
        raise SystemExit(f"workers must be 1..{MAX_WORKERS}")
    receipt, specs = verify_review_receipt(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "review_receipt_sha256": file_sha256(args.review_receipt),
        "role": args.role,
        "frozen_config_sha256": receipt["frozen_config_sha256"],
        "runner_code_sha256": receipt["runner_code_sha256"],
        "presence_code_sha256": receipt["presence_code_sha256"],
        "prior_pyin_code_sha256": receipt["prior_pyin_code_sha256"],
        "pair_manifest_sha256": receipt["pair_manifest_sha256"],
        "selected_source_inventory_sha256": receipt["selected_source_inventory_sha256"],
    }
    contract_hash = canonical_hash(contract)
    pending = []
    for spec in specs:
        receipt_path = item_receipt_path(args.output_dir, spec["clip_id"])
        if _valid_cached_receipt(receipt_path, args.output_dir, spec, contract_hash):
            continue
        if receipt_path.exists() and not args.retry_errors:
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
            if cached.get("status") == "error" and cached.get("contract_hash") == contract_hash:
                continue
        pending.append(spec)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_clip, spec, contract_hash): spec for spec in pending}
        for future in as_completed(futures):
            result = future.result()
            spec = futures[future]
            receipt_path = item_receipt_path(args.output_dir, spec["clip_id"])
            if result["status"] == "ok":
                arrays = result.pop("arrays")
                contour = contour_path(args.output_dir, spec["clip_id"])
                atomic_npz(contour, arrays)
                result["contour_path"] = str(contour.resolve())
                result["contour_sha256"] = file_sha256(contour)
            atomic_json(receipt_path, result)
            print(f"{result['status']} {result['clip_id']}", flush=True)

    summary = consolidate(args.output_dir, specs, contract_hash, args.role)
    run_manifest = {
        "schema": "vocalset-vibrato-run-manifest-v1",
        "status": summary["status"],
        "role": args.role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "packages": package_versions(),
        "workers": args.workers,
        "contract": contract,
        "contract_hash": contract_hash,
        "review_receipt_path": str(args.review_receipt.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "summary_sha256": file_sha256(args.output_dir / "summary.json"),
        "receipt_count": len(list((args.output_dir / "receipts").glob("*.json"))),
        "contour_count": len(list((args.output_dir / "contours").glob("*.npz"))),
        "output_file_sha256": {},
    }
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            run_manifest["output_file_sha256"][str(path.relative_to(args.output_dir))] = file_sha256(path)
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(clean(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--pair-manifest", required=True, type=Path)
        subparser.add_argument("--source-root", required=True, type=Path)
        subparser.add_argument("--review-receipt", required=True, type=Path)
        subparser.add_argument("--role", required=True, choices=("development", "evaluation"))
        subparser.add_argument("--review-decision", required=True)
        if command == "run":
            subparser.add_argument("--output-dir", required=True, type=Path)
            subparser.add_argument("--workers", type=int, default=4)
            subparser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("pair_manifest", "source_root", "review_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == "prepare":
        return prepare_review_receipt(args)
    args.output_dir = args.output_dir.resolve()
    return run_assay(args)


if __name__ == "__main__":
    raise SystemExit(main())
