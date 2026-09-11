#!/usr/bin/env python3
"""Preflight and run the frozen 10 s/30 s expanded evaluation protocol.

The default mode is preflight-only.  Formal scoring is reachable only with
``--mode run --confirm-formal-ready`` after both duration views satisfy the
frozen inventory, role, source-unit, identifier, and extraction-status gates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_expanded as evaluator


EXPECTED_ROLE_PERMISSION = {
    "development": "yes",
    "locked": "locked_only",
    "stress": "stress_only",
    "provisional": "no_until_verified",
    "pilot": "pilot_only",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    artifact = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--confirm-formal-ready", action="store_true")
    parser.add_argument(
        "--metadata-10s", type=Path,
        default=artifact / "manifests/final/metadata_10s.csv",
    )
    parser.add_argument(
        "--metadata-30s", type=Path,
        default=artifact / "manifests/final/metadata_30s.csv",
    )
    parser.add_argument("--features-10s", type=Path, required=True)
    parser.add_argument("--features-30s", type=Path, required=True)
    parser.add_argument("--verification-10s", type=Path, required=True)
    parser.add_argument("--verification-30s", type=Path, required=True)
    parser.add_argument("--materialization-audit", type=Path, required=True)
    parser.add_argument("--legacy-audit", type=Path, required=True)
    parser.add_argument("--families-10s", type=Path, default=here / "evaluate_expanded_families_10s.json")
    parser.add_argument("--families-30s", type=Path, default=here / "evaluate_expanded_families_30s.json")
    parser.add_argument("--training-scopes", type=Path, default=here / "evaluate_expanded_training_scopes.json")
    parser.add_argument("--evaluation-slices", type=Path, default=here / "evaluate_expanded_slices.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--latex-output-dir", type=Path, default=artifact / "latex")
    parser.add_argument("--prior-heldout", type=Path, action="append", default=[])
    parser.add_argument("--exclusion-manifest", type=Path,
                        help="Optional CSV/JSON of unavailable duplicate IDs and non-empty reasons")
    parser.add_argument("--expected-metadata-10s", type=int, default=10_141)
    parser.add_argument("--expected-expansion-10s", type=int, default=2_641)
    parser.add_argument("--expected-development-10s", type=int, default=8_361)
    parser.add_argument("--expected-locked-10s", type=int, default=1_030)
    parser.add_argument("--expected-stress-10s", type=int, default=200)
    parser.add_argument("--expected-pilot-10s", type=int, default=50)
    parser.add_argument("--expected-provisional-10s", type=int, default=500)
    parser.add_argument("--expected-metadata-30s", type=int, default=4_497)
    parser.add_argument("--expected-development-30s", type=int, default=3_317)
    parser.add_argument("--expected-locked-30s", type=int, default=1_022)
    parser.add_argument("--expected-stress-30s", type=int, default=108)
    parser.add_argument("--expected-pilot-30s", type=int, default=50)
    parser.add_argument("--expected-provisional-30s", type=int, default=0)
    parser.add_argument("--expected-human-sources-10s", type=int, default=7)
    parser.add_argument("--expected-ai-sources-10s", type=int, default=13)
    parser.add_argument("--seed", type=int, default=evaluator.DEFAULT_SEED)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def id_set_sha256(values: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "1.0", "true", "yes", "available"}
    )


def load_exclusions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, low_memory=False)
        id_column = next((name for name in ("track", "id", "item_id", "row_id") if name in table), None)
        reason_column = next((name for name in ("reason", "exclusion_reason", "justification") if name in table), None)
        if id_column is None or reason_column is None:
            raise ValueError("Exclusion CSV needs an ID column and reason/exclusion_reason/justification")
        rows = zip(table[id_column], table[reason_column])
    else:
        if path.suffix.lower() == ".jsonl":
            payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.items()
        elif isinstance(payload, list):
            rows = ((item.get("track") or item.get("id") or item.get("item_id"),
                     item.get("reason") or item.get("exclusion_reason") or item.get("justification"))
                    for item in payload)
        else:
            raise ValueError("Exclusion JSON must be an ID-to-reason object or a list of records")
    output: dict[str, str] = {}
    for raw_id, raw_reason in rows:
        item_id, reason = str(raw_id).strip(), str(raw_reason).strip()
        if not item_id or not reason or "duplicate" not in reason.lower():
            raise ValueError("Every exclusion needs an ID and a duplicate-specific justification")
        if item_id in output:
            raise ValueError(f"Duplicate exclusion ID: {item_id}")
        output[item_id] = reason
    return output


def feature_id_column(table: pd.DataFrame) -> str:
    candidates = [name for name in ("item_id", "track", "id", "row_id") if name in table]
    if not candidates:
        raise ValueError("Feature table has no item_id/track/id/row_id")
    return candidates[0]


def load_failure_manifest(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, low_memory=False)
        id_column = next((name for name in ("item_id", "track", "id", "row_id") if name in table), None)
        reason_column = next((name for name in ("reason", "failure_reason", "error", "errors") if name in table), None)
        if id_column is None or reason_column is None:
            raise ValueError("Serialization-failure CSV needs ID and reason fields")
        rows = zip(table[id_column], table[reason_column])
    else:
        if path.suffix.lower() == ".jsonl":
            payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Serialization-failure JSON must be a list of records")
        rows_list: list[tuple[Any, Any]] = []
        for item in payload:
            nested = item.get("serialization_failure")
            if isinstance(nested, dict):
                # The canonical accounting JSONL contains one record per
                # feature row; an empty object means no serialization failure.
                if not nested:
                    continue
                raw_reason = nested.get("reason")
            else:
                raw_reason = (
                    item.get("reason") or item.get("failure_reason")
                    or item.get("error") or item.get("errors")
                )
            rows_list.append((item.get("item_id") or item.get("track") or item.get("id"), raw_reason))
        rows = rows_list
    output: dict[str, str] = {}
    for raw_id, raw_reason in rows:
        item_id, reason = str(raw_id).strip(), str(raw_reason).strip()
        if not item_id or not reason or item_id in output:
            raise ValueError("Serialization-failure manifest has missing/duplicate IDs or reasons")
        output[item_id] = reason
    return output


def validate_verification(
    path: Path,
    duration: str,
    duration_evidence: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Verification artifact must be an object: {path}")
    if payload.get("schema_version") != 1 or payload.get("status") != "passed" or payload.get("passed") is not True:
        raise ValueError(f"Verification artifact lacks a positive schema-v1 pass: {path}")
    if payload.get("duration_view") != duration:
        raise ValueError(f"Verification duration mismatch in {path}")
    for section in ("metadata", "features", "expected", "identity", "feature_status_counts",
                    "failure_accounting", "hash_definition"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"Verification artifact lacks required section {section}: {path}")
    metadata, features = payload["metadata"], payload["features"]
    expected, identity = payload["expected"], payload["identity"]
    accounting = payload["failure_accounting"]
    expected_values = {
        "metadata.path": (str(Path(metadata.get("path", "")).resolve()), duration_evidence["metadata_path"]),
        "metadata.sha256": (metadata.get("sha256"), duration_evidence["metadata_sha256"]),
        "metadata.row_count": (metadata.get("row_count"), duration_evidence["active_rows"]),
        "metadata.unique_id_count": (metadata.get("unique_id_count"), duration_evidence["active_rows"]),
        "metadata.id_set_sha256": (metadata.get("id_set_sha256"), duration_evidence["id_set_sha256"]),
        "features.path": (str(Path(features.get("path", "")).resolve()), duration_evidence["feature_path"]),
        "features.sha256": (features.get("sha256"), duration_evidence["feature_sha256"]),
        "features.row_count": (features.get("row_count"), duration_evidence["feature_rows"]),
        "features.unique_id_count": (features.get("unique_id_count"), duration_evidence["feature_rows"]),
        "features.id_set_sha256": (features.get("id_set_sha256"), duration_evidence["id_set_sha256"]),
        "expected.metadata_row_count": (expected.get("metadata_row_count"), duration_evidence["active_rows"]),
        "expected.feature_row_count": (expected.get("feature_row_count"), duration_evidence["feature_rows"]),
    }
    mismatches = {name: {"recorded": actual, "required": required}
                  for name, (actual, required) in expected_values.items() if actual != required}
    if mismatches:
        raise ValueError(f"Verification counts/paths/hashes do not match current inputs: {mismatches}")
    if identity.get("exact_id_set_match") is not True or any(identity.get(name) != [] for name in (
        "duplicate_metadata_ids", "duplicate_feature_ids", "missing_feature_ids", "unexpected_feature_ids"
    )):
        raise ValueError(f"Verification identity gate is not an exact duplicate-free match: {path}")
    status_counts = payload["feature_status_counts"]
    if any(not isinstance(status_counts.get(name), dict) for name in ("s16", "s8", "d", "r", "p", "overall")):
        raise ValueError(f"Verification feature_status_counts lacks S8/S16/D/R/P/overall maps: {path}")
    normalized_status_counts = {
        family: {str(key): int(value) for key, value in status_counts[family].items()}
        for family in ("s16", "s8", "d", "r", "p", "overall")
    }
    if normalized_status_counts != duration_evidence["feature_status_counts"]:
        raise ValueError(f"Verification family status counts do not match feature rows: {path}")
    if accounting.get("unexplained_failure_count") != 0 or accounting.get("unexplained_failure_ids") != []:
        raise ValueError(f"Verification has unexplained feature failures: {path}")
    serialization_count = accounting.get("serialization_failure_count")
    mapped_count = accounting.get("serialization_failures_mapped_count")
    if not isinstance(serialization_count, int) or serialization_count < 0:
        raise ValueError(f"Invalid serialization failure count: {path}")
    if accounting.get("serialization_failures_all_mapped") is not True or mapped_count != serialization_count:
        raise ValueError(f"Serialization failures are not completely mapped: {path}")
    if accounting.get("natural_missing_allowed") is not True or not isinstance(
        accounting.get("natural_missing_counts"), dict
    ) or any(name not in accounting["natural_missing_counts"] for name in ("d", "r", "p")):
        raise ValueError(f"Natural D/R/P missingness policy is absent: {path}")
    hash_definition = payload["hash_definition"]
    if hash_definition.get("id_set_sha256") != (
        "sha256 of UTF-8 sorted unique IDs joined by newline with trailing newline"
    ) or hash_definition.get("file_sha256") != "raw file bytes":
        raise ValueError(f"Verification hash definitions differ from the frozen contract: {path}")

    if serialization_count:
        manifest_raw = accounting.get("serialization_failure_manifest_path")
        manifest_hash = accounting.get("serialization_failure_manifest_sha256")
        if not manifest_raw:
            raise ValueError(f"Serialization failures lack a manifest: {path}")
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_file() or sha256(manifest_path) != manifest_hash:
            raise ValueError(f"Serialization-failure manifest hash/path mismatch: {path}")
        failures = load_failure_manifest(manifest_path)
        if len(failures) != serialization_count:
            raise ValueError(f"Serialization-failure manifest count mismatch: {path}")
        feature_table = pd.read_csv(duration_evidence["feature_path"], low_memory=False)
        id_column = feature_id_column(feature_table)
        reason_column = next((name for name in (
            "errors", "failure_reason", "feature_error", "feature_error_reason"
        )
                              if name in feature_table), None)
        if reason_column is None:
            raise ValueError("Mapped serialization failures require feature-row error reasons")
        feature_reasons = dict(zip(feature_table[id_column].astype(str),
                                   feature_table[reason_column].fillna("").astype(str)))
        mismatched_failures = [item_id for item_id, reason in failures.items()
                               if item_id not in feature_reasons
                               or reason.lower() not in feature_reasons[item_id].lower()]
        if mismatched_failures:
            raise ValueError(f"Serialization failures do not match feature reasons: {mismatched_failures[:10]}")
    elif accounting.get("serialization_failure_manifest_path") not in (None, "") or accounting.get(
        "serialization_failure_manifest_sha256"
    ) not in (None, ""):
        raise ValueError(f"Zero serialization failures must not cite a failure manifest: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "positive_gate": "schema_v1_passed"}


def validate_materialization_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "passed", "passed": True, "expected": 2641, "verified": 2641,
        "materialized_or_excluded": 2641, "zero_unexplained_failures": True,
        "unexplained_failure_count": 0, "rehash_enabled": True, "valid": True,
        "canonical_records": 2641, "success_items": 2641,
    }
    mismatches = {key: {"recorded": payload.get(key), "required": value}
                  for key, value in required.items() if payload.get(key) != value}
    if mismatches:
        raise ValueError(f"Physical materialization audit is not a positive 2,641-row gate: {mismatches}")
    for key in ("materialization_manifest_sha256", "frozen_manifest_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Physical materialization audit lacks {key}")
    exclusion_path = payload.get("duplicate_exclusion_manifest_path")
    exclusion_hash = payload.get("duplicate_exclusion_manifest_sha256")
    if (exclusion_path is None) != (exclusion_hash is None):
        raise ValueError("Duplicate-exclusion materialization fields must both be null or both populated")
    if payload.get("errors") not in (None, [], {}):
        raise ValueError("Physical materialization audit contains errors")
    return {"path": str(path.resolve()), "sha256": sha256(path), "positive_gate": "2641_verified"}


def validate_legacy_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"status": "passed", "passed": True, "expected": 1000,
                "verified": 1000, "unique_native_sha256": 1000}
    mismatches = {key: {"recorded": payload.get(key), "required": value}
                  for key, value in required.items() if payload.get(key) != value}
    details = payload.get("details")
    if mismatches or not isinstance(details, list) or len(details) != 1000 or any(
        item.get("status") != "verified" for item in details
    ):
        raise ValueError(f"Legacy native-audio audit is not a positive 1,000-row gate: {mismatches}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "positive_gate": "1000_verified"}


def validate_duration(
    duration: str,
    metadata_path: Path,
    features_path: Path,
    exclusions: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    required = {
        "track", "label", "source_group", "role", "group_id", "acquisition",
        "available", "evaluation_allowed", "duration_view",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{duration} metadata missing required fields: {missing}")
    if metadata.track.isna().any() or metadata.track.astype(str).duplicated().any():
        raise ValueError(f"{duration} metadata track IDs must be non-empty and unique")
    metadata["track"] = metadata.track.astype(str)
    if set(metadata.duration_view.astype(str)) != {duration}:
        raise ValueError(f"{duration} metadata has mismatched duration_view values")
    available = truthy(metadata.available)
    unavailable_ids = set(metadata.loc[~available, "track"])
    exclusion_ids = set(exclusions) & set(metadata.track)
    if unavailable_ids != exclusion_ids:
        unexplained = sorted(unavailable_ids - exclusion_ids)[:10]
        ineligible_exclusions = sorted(exclusion_ids - unavailable_ids)[:10]
        raise ValueError(
            f"{duration} unavailable rows must exactly equal justified duplicate exclusions; "
            f"unexplained={unexplained}, exclusions_marked_available={ineligible_exclusions}"
        )
    active = metadata[available].copy()

    expansion = metadata.acquisition.astype(str).eq("expansion")
    if duration == "10s":
        if len(metadata) != args.expected_metadata_10s:
            raise ValueError(f"10s metadata rows={len(metadata)}, expected={args.expected_metadata_10s}")
        if int(expansion.sum()) != args.expected_expansion_10s:
            raise ValueError(
                f"10s expansion rows={int(expansion.sum())}, expected={args.expected_expansion_10s}"
            )
        materialized_or_excluded = int((expansion & available).sum()) + len(set(
            metadata.loc[expansion, "track"]
        ) & exclusion_ids)
        if materialized_or_excluded != args.expected_expansion_10s:
            raise ValueError("Not all 2,641 expansion rows are materialized or justified duplicates")
        expected_roles = {
            "development": args.expected_development_10s,
            "locked": args.expected_locked_10s,
            "stress": args.expected_stress_10s,
            "pilot": args.expected_pilot_10s,
            "provisional": args.expected_provisional_10s,
        }
    else:
        if len(metadata) != args.expected_metadata_30s:
            raise ValueError(f"30s metadata rows={len(metadata)}, expected={args.expected_metadata_30s}")
        expected_roles = {
            "development": args.expected_development_30s,
            "locked": args.expected_locked_30s,
            "stress": args.expected_stress_30s,
            "pilot": args.expected_pilot_30s,
            "provisional": args.expected_provisional_30s,
        }

    raw_role_counts = metadata.role.astype(str).value_counts().to_dict()
    unknown_roles = sorted(set(raw_role_counts) - set(expected_roles))
    if unknown_roles:
        raise ValueError(f"{duration} metadata has unknown roles: {unknown_roles}")
    observed_roles = {
        role: int(raw_role_counts.get(role, 0)) for role in expected_roles
    }
    if observed_roles != expected_roles:
        raise ValueError(f"{duration} role counts={observed_roles}, expected={expected_roles}")

    expansion_metadata = metadata[expansion]
    expected_permission = expansion_metadata.role.map(EXPECTED_ROLE_PERMISSION)
    if expected_permission.isna().any():
        raise ValueError(f"{duration} expansion metadata has unknown roles")
    permission = expansion_metadata.evaluation_allowed.astype(str).str.strip()
    contradictions = expansion_metadata[permission.to_numpy() != expected_permission.astype(str).to_numpy()]
    if len(contradictions):
        raise ValueError(
            f"{duration} has {len(contradictions)} role/evaluation_allowed contradictions; "
            f"examples={contradictions.track.head(10).tolist()}"
        )
    prior = metadata[~expansion]
    if not set(prior.role.astype(str)).issubset({"development", "locked"}):
        raise ValueError(f"{duration} prior rows contain an unexpected role")
    cross_role_groups = metadata.groupby("group_id", dropna=False).role.nunique()
    if bool((cross_role_groups > 1).any()):
        raise ValueError(f"{duration} group_id crosses role boundaries")

    development = active[active.role.astype(str).eq("development")]
    labelled = pd.to_numeric(development.label, errors="coerce")
    if not labelled.isin([0, 1]).all():
        raise ValueError(f"{duration} development labels must be binary")
    source_counts = development.assign(__label=labelled).groupby("__label").source_group.nunique()
    if duration == "10s":
        human_sources, ai_sources = int(source_counts.get(0, 0)), int(source_counts.get(1, 0))
        if (human_sources, ai_sources) != (
            args.expected_human_sources_10s, args.expected_ai_sources_10s
        ):
            raise ValueError(
                f"10s source_group units Human/AI={(human_sources, ai_sources)}, expected="
                f"{(args.expected_human_sources_10s, args.expected_ai_sources_10s)}"
            )

    features = pd.read_csv(features_path, low_memory=False)
    id_column = feature_id_column(features)
    if features[id_column].isna().any() or features[id_column].astype(str).duplicated().any():
        raise ValueError(f"{duration} feature IDs must be non-empty and unique")
    feature_ids = set(features[id_column].astype(str))
    expected_ids = set(active.track)
    if feature_ids != expected_ids:
        raise ValueError(
            f"{duration} feature ID set mismatch: missing={sorted(expected_ids-feature_ids)[:10]}, "
            f"extra={sorted(feature_ids-expected_ids)[:10]}"
        )
    status_columns = {
        "s16": "s16_feature_status", "s8": "s8_feature_status",
        "d": "d_feature_status", "r": "r_feature_status", "p": "p_feature_status",
        "overall": "feature_status",
    }
    missing_status_columns = [column for column in status_columns.values() if column not in features]
    if missing_status_columns:
        raise ValueError(f"{duration} features lack final status columns: {missing_status_columns}")
    family_status_counts: dict[str, dict[str, int]] = {}
    for family, column in status_columns.items():
        values = features[column]
        if values.isna().any() or (values.astype(str).str.strip() == "").any():
            raise ValueError(f"{duration} features need non-empty {column} on every row")
        normalized = values.astype(str).str.strip()
        family_status_counts[family] = {
            str(key): int(value) for key, value in normalized.value_counts().sort_index().items()
        }
    vocabularies = {
        "s16": {"computed", "processing_failure", "missing_feature_source"},
        "s8": {"computed", "processing_failure", "missing_feature_source"},
        "d": {"eligible", "unavailable_fewer_than_8_events", "processing_failure",
              "missing_inference_output", "missing_feature_source"},
        "r": {"eligible", "unavailable_fewer_than_16_beats_or_invalid_intervals",
              "processing_failure", "missing_inference_output", "missing_feature_source"},
        "p": {"eligible", "unavailable_fewer_than_3_sections_or_downbeat_spans",
              "processing_failure", "missing_inference_output", "missing_feature_source"},
        "overall": {"complete", "accounted_serialization_failure", "unexplained_failure"},
    }
    for family, counts in family_status_counts.items():
        unknown = set(counts) - vocabularies[family]
        if unknown:
            raise ValueError(f"{duration} {family} status has unknown labels: {sorted(unknown)}")
    if family_status_counts["overall"].get("unexplained_failure", 0):
        raise ValueError(f"{duration} feature table contains unexplained failures")
    if any(counts.get("missing_feature_source", 0) for family, counts in family_status_counts.items()
           if family != "overall"):
        raise ValueError(f"{duration} feature table contains missing feature-source records")
    status = features["feature_status"].astype(str).str.strip()
    incomplete = status.ne("complete")
    if incomplete.any():
        reason_column = next((name for name in (
            "errors", "failure_reason", "feature_error", "feature_error_reason"
        )
                              if name in features), None)
        if reason_column is None:
            raise ValueError(f"{duration} incomplete feature rows lack an error-reason column")
        reasons = features.loc[incomplete, reason_column]
        unexplained = reasons.isna() | reasons.astype(str).str.strip().isin({"", "[]", "{}", "nan"})
        if unexplained.any():
            examples = features.loc[incomplete].loc[unexplained, id_column].head(10).tolist()
            raise ValueError(f"{duration} incomplete feature rows lack explicit reasons: {examples}")

    return {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "feature_path": str(features_path.resolve()),
        "feature_sha256": sha256(features_path),
        "metadata_rows": int(len(metadata)),
        "active_rows": int(len(active)),
        "excluded_duplicate_rows": int(len(unavailable_ids)),
        "development_rows_active": int(len(development)),
        "role_counts": observed_roles,
        "development_human_sources": int(source_counts.get(0, 0)),
        "development_ai_sources": int(source_counts.get(1, 0)),
        "feature_rows": int(len(features)),
        "id_set_sha256": id_set_sha256(expected_ids),
        "feature_status_counts": family_status_counts,
        "source_column": "source_group",
        "group_column": "group_id",
    }


def command_record(command: list[str], log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    record = {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "log": str(log_path.resolve()),
        "log_sha256": sha256(log_path),
    }
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return record


def command_records_parallel(
    jobs: list[tuple[list[str], Path]],
) -> list[dict[str, Any]]:
    """Run independent commands concurrently and join all before returning.

    Every subprocess is allowed to finish even if a peer fails.  The caller
    receives no records until the complete batch has joined, so downstream
    locked evaluation cannot begin after only one duration-specific CV run.
    """
    if not jobs:
        return []
    records: list[dict[str, Any] | None] = [None] * len(jobs)
    failures: list[tuple[int, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(command_record, command, log_path)
            for command, log_path in jobs
        ]
        for index, future in enumerate(futures):
            try:
                records[index] = future.result()
            except BaseException as exc:  # join and report every submitted CV process
                failures.append((index, exc))
    if failures:
        details = "; ".join(f"job {index}: {exc}" for index, exc in failures)
        raise RuntimeError(f"Parallel command batch failed after all jobs joined: {details}")
    return [record for record in records if record is not None]


def assert_frozen_inputs_unchanged(expected: dict[str, str]) -> None:
    changed = []
    for raw_path, expected_hash in expected.items():
        path = Path(raw_path)
        current_hash = sha256(path)
        if current_hash != expected_hash:
            changed.append({"path": raw_path, "expected": expected_hash, "current": current_hash})
    if changed:
        raise RuntimeError(f"Formal input changed after preflight: {changed}")


def verify_cv_outputs(cv_dir: Path, expected_feature_set: str) -> dict[str, Any]:
    summary_path = cv_dir / "evaluate_expanded_cv_summary.csv"
    frozen_path = cv_dir / "evaluate_expanded_frozen_selection.json"
    scope_path = cv_dir / "evaluate_expanded_training_scope_matched.csv"
    fold_path = cv_dir / "evaluate_expanded_fold_manifest.csv"
    summary = pd.read_csv(summary_path, low_memory=False)
    primary = summary[(summary.feature_set == expected_feature_set)
                      & (summary.training_scope == "all_development")]
    expected_quantities = {"25", "50", "100", "200", "all"}
    if primary.combination.nunique() != 15 or set(primary.quantity.astype(str)) != expected_quantities:
        raise ValueError("Formal CV output does not contain the primary 15 x 5 grid")
    if len(primary) != 75:
        raise ValueError(f"Primary 15 x 5 grid has {len(primary)} rows rather than 75")
    matched = pd.read_csv(scope_path, low_memory=False)
    if matched.empty or not {"prior_only", "all_development"}.issubset(
        set(summary.training_scope)
    ):
        raise ValueError("Prior-only versus expanded matched-fold ablation is absent")
    folds = pd.read_csv(fold_path, low_memory=False)
    if (folds.status == "evaluated").sum() == 0:
        raise ValueError("No evaluated fold-manifest entries")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("source_column") != "source_group" or frozen.get("group_column") != "group_id":
        raise ValueError("Frozen selection did not record the required source/group columns")
    if frozen.get("created_by_stage") != "cv_before_locked_evaluation":
        raise ValueError("Frozen selection stage marker is invalid")
    return {
        "summary_sha256": sha256(summary_path),
        "frozen_selection_path": str(frozen_path.resolve()),
        "frozen_selection_sha256": sha256(frozen_path),
        "fold_manifest_sha256": sha256(fold_path),
        "matched_prior_sha256": sha256(scope_path),
        "leader": frozen["leader"],
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / "evaluate_expanded_formal_orchestration_audit.json"
    if audit_path.exists():
        history = args.output_root / "preflight_history"
        history.mkdir(parents=True, exist_ok=True)
        preserved = history / f"evaluate_expanded_formal_orchestration_audit_{time.time_ns()}.json"
        shutil.copy2(audit_path, preserved)
    exclusions = load_exclusions(args.exclusion_manifest)
    audit: dict[str, Any] = {
        "status": "preflight_running",
        "mode": args.mode,
        "scientific_status": "no formal scores produced from partial data",
        "source_column": "source_group",
        "group_column": "group_id",
        "seed": args.seed,
        "code_hashes": {},
        "input_hashes_frozen_before_cv": {},
        "commands": [],
    }
    code_files = [
        Path(evaluator.__file__).resolve(), Path(__file__).resolve(),
        Path(__file__).with_name("evaluate_expanded_group_bootstrap.py"),
        Path(__file__).with_name("evaluate_expanded_report.py"),
        Path(__file__).with_name("evaluate_expanded_plot.py"),
        Path(__file__).with_name("evaluate_expanded_latex.py"),
        Path(__file__).with_name("evaluate_expanded_reporting.py"),
    ]
    try:
        for path in code_files:
            audit["code_hashes"][path.name] = sha256(path)
        audit["durations"] = {
            "10s": validate_duration("10s", args.metadata_10s, args.features_10s, exclusions, args),
            "30s": validate_duration("30s", args.metadata_30s, args.features_30s, exclusions, args),
        }
        audit["verifications"] = {
            "10s": validate_verification(
                args.verification_10s, "10s", audit["durations"]["10s"]
            ),
            "30s": validate_verification(
                args.verification_30s, "30s", audit["durations"]["30s"]
            ),
        }
        audit["materialization_audit"] = validate_materialization_audit(args.materialization_audit)
        audit["legacy_audit"] = validate_legacy_audit(args.legacy_audit)
        input_paths = [
            args.metadata_10s, args.features_10s, args.verification_10s, args.families_10s,
            args.metadata_30s, args.features_30s, args.verification_30s, args.families_30s,
            args.materialization_audit, args.legacy_audit,
            args.training_scopes, args.evaluation_slices, *args.prior_heldout,
            *code_files,
        ] + ([args.exclusion_manifest] if args.exclusion_manifest else [])
        audit["input_hashes_frozen_before_cv"] = {
            str(path.resolve()): sha256(path) for path in input_paths
        }
        audit["status"] = "preflight_passed"
        audit["scientific_status"] = "ready_for_formal_scoring"
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.mode == "preflight":
            print(json.dumps(audit, indent=2, allow_nan=False))
            return
        if not args.confirm_formal_ready:
            raise ValueError("Formal run requires --confirm-formal-ready")

        program_dir = Path(__file__).resolve().parent
        run_entries: list[dict[str, Any]] = []
        # Freeze both duration-specific CV leaders before opening either locked
        # result.  This prevents the 10 s locked scores from influencing the
        # independently prespecified 30 s candidate choice.
        cv_jobs: list[tuple[list[str], Path]] = []
        for duration, metadata, features, families, expected_primary in (
            ("10s", args.metadata_10s, args.features_10s, args.families_10s, "common8_10s"),
            ("30s", args.metadata_30s, args.features_30s, args.families_30s, "common8_30s"),
        ):
            duration_root = args.output_root / duration
            cv_dir, locked_dir = duration_root / "cv", duration_root / "locked"
            cv_dir.mkdir(parents=True, exist_ok=True)
            locked_dir.mkdir(parents=True, exist_ok=True)
            base = [
                args.python, str(Path(evaluator.__file__).resolve()),
                "--metadata", str(metadata), "--features", str(features),
                "--families-json", str(families),
                "--id-column", "track", "--label-column", "label", "--role-column", "role",
                "--source-column", "source_group", "--group-column", "group_id",
                "--seed", str(args.seed), "--opposite-class-folds", "5",
            ]
            cv_command = [
                *base, "--stage", "cv", "--output-dir", str(cv_dir),
                "--training-scopes-json", str(args.training_scopes),
                "--quantities", "25,50,100,200,all", "--bootstrap-replicates", "0",
            ]
            for prior in args.prior_heldout:
                cv_command.extend(("--prior-heldout", str(prior)))
            cv_jobs.append((cv_command, cv_dir / "formal_cv.log"))
            run_entries.append({
                "duration": duration, "base": base, "duration_root": duration_root,
                "cv_dir": cv_dir, "locked_dir": locked_dir,
                "metadata": metadata, "features": features,
                "expected_primary": expected_primary,
            })

        # The two CV searches are independent.  Join both subprocesses before
        # validating/freezing either leader; no locked command is constructed
        # until every CV process has succeeded and its outputs are verified.
        audit["commands"].extend(command_records_parallel(cv_jobs))
        assert_frozen_inputs_unchanged(audit["input_hashes_frozen_before_cv"])
        for entry in run_entries:
            duration = entry["duration"]
            cv_evidence = verify_cv_outputs(entry["cv_dir"], entry["expected_primary"])
            audit.setdefault("cv_evidence", {})[duration] = cv_evidence
            entry["frozen_path"] = Path(cv_evidence["frozen_selection_path"])

        # Persist both leaders and every relevant input/code hash before any
        # locked score is opened.
        audit["status"] = "both_cv_leaders_frozen_before_locked"
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        assert_frozen_inputs_unchanged(audit["input_hashes_frozen_before_cv"])

        for entry in run_entries:
            duration = entry["duration"]
            base = entry["base"]
            duration_root = entry["duration_root"]
            cv_dir = entry["cv_dir"]
            locked_dir = entry["locked_dir"]
            frozen_path = entry["frozen_path"]
            locked_command = [
                *base, "--stage", "locked", "--output-dir", str(locked_dir),
                "--frozen-selection", str(frozen_path),
                "--evaluation-slices-json", str(args.evaluation_slices),
            ]
            audit["commands"].append(command_record(locked_command, locked_dir / "formal_locked.log"))
            bootstrap_dir = duration_root / "group_bootstrap"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            audit["commands"].append(command_record([
                args.python, str(program_dir / "evaluate_expanded_group_bootstrap.py"),
                "--scores", str(locked_dir / "evaluate_expanded_locked_scores.csv"),
                "--frozen-selection", str(frozen_path), "--output-dir", str(bootstrap_dir),
                "--replicates", "2000", "--seed", str(args.seed),
            ], bootstrap_dir / "formal_group_bootstrap.log"))
            audit["commands"].append(command_record([
                args.python, str(program_dir / "evaluate_expanded_report.py"),
                "--cv-dir", str(cv_dir), "--locked-dir", str(locked_dir),
                "--metadata", str(entry["metadata"]), "--features", str(entry["features"]),
                "--duration", duration,
                "--output", str(duration_root / "evaluate_expanded_report.md"),
            ], duration_root / "formal_report.log"))
            audit["commands"].append(command_record([
                args.python, str(program_dir / "evaluate_expanded_plot.py"),
                "--cv-summary", str(cv_dir / "evaluate_expanded_cv_summary.csv"),
                "--frozen-selection", str(frozen_path),
                "--output", str(duration_root / "evaluate_expanded_15x5_auc_ba.png"),
            ], duration_root / "formal_plot.log"))
            audit["commands"].append(command_record([
                args.python, str(program_dir / "evaluate_expanded_plot.py"),
                "--cv-summary", str(cv_dir / "evaluate_expanded_cv_summary.csv"),
                "--frozen-selection", str(frozen_path),
                "--output", str(duration_root / "evaluate_expanded_15x5_auc_ba.pdf"),
            ], duration_root / "formal_plot_pdf.log"))
            for panel in ("auc", "ba"):
                audit["commands"].append(command_record([
                    args.python, str(program_dir / "evaluate_expanded_plot.py"),
                    "--cv-summary", str(cv_dir / "evaluate_expanded_cv_summary.csv"),
                    "--frozen-selection", str(frozen_path), "--panel", panel,
                    "--output", str(duration_root / f"evaluate_expanded_15x5_{panel}.png"),
                ], duration_root / f"formal_plot_{panel}.log"))
        audit["status"] = "formal_scoring_complete"
        audit["scientific_status"] = "both CV leaders frozen before locked scoring; reporting only remains"
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for entry in run_entries:
            duration = entry["duration"]
            duration_root = entry["duration_root"]
            audit["commands"].append(command_record([
                args.python, str(program_dir / "evaluate_expanded_latex.py"),
                "--duration", duration, "--cv-dir", str(entry["cv_dir"]),
                "--locked-dir", str(entry["locked_dir"]),
                "--bootstrap-dir", str(duration_root / "group_bootstrap"),
                "--formal-audit", str(audit_path), "--output-dir", str(args.latex_output_dir),
                "--metadata", str(entry["metadata"]), "--features", str(entry["features"]),
            ], duration_root / "formal_latex.log"))
        audit["status"] = "formal_complete"
        audit["scientific_status"] = "cv leaders frozen before locked scoring; locked scores never selected"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
