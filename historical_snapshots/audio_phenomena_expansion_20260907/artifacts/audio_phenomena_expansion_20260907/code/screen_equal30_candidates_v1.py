#!/usr/bin/env python3
"""Screen equal-30 metadata candidates without admitting audio or fitting models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIRED_IDENTITY = ("id", "label", "source_group", "group_id", "role")
ALLOWED_EVALUATION = {"", "1", "true", "yes", "allowed", "development"}


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def require_unchanged(path: Path, expected: dict[str, Any]) -> None:
    current = file_snapshot(path)
    if current != expected:
        fail(f"bound file changed during screen: {path}")


def safe_path(value: Path, *, output: bool = False) -> Path:
    value = Path(value)
    if not value.is_absolute() or ".." in value.parts:
        fail(f"absolute traversal-free path required: {value}")
    for existing in (value, *value.parents):
        if existing.exists() and existing.is_symlink():
            fail(f"symlink path component is forbidden: {existing}")
    if output:
        if value.exists() or value.is_symlink():
            fail(f"refusing to overwrite output: {value}")
        if not value.parent.is_dir() or value.parent.is_symlink():
            fail(f"output parent must be an existing regular directory: {value.parent}")
    elif not value.is_file() or value.is_symlink():
        fail(f"input must be an existing regular file: {value}")
    return value


def safe_id(value: str) -> str:
    if not value or "\x00" in value or "/" in value or "\\" in value or value in {".", ".."}:
        fail(f"unsafe or missing id: {value!r}")
    return value


def read_csv(path: Path, name: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            fail(f"{name}: invalid CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            fail(f"{name}: duplicate CSV header name")
        rows = list(reader)
    seen: set[str] = set()
    for number, row in enumerate(rows, 2):
        if None in row or None in row.values():
            fail(f"{name}:{number}: malformed CSV row")
        identity = safe_id(row.get("id", ""))
        if identity in seen:
            fail(f"{name}:{number}: ambiguous duplicate id {identity}")
        seen.add(identity)
        raw_hash = row.get("raw_sha256", "")
        if raw_hash and HEX64.fullmatch(raw_hash) is None:
            fail(f"{name}:{number}: malformed raw_sha256 for {identity}")
        label = row.get("label", "")
        if label and label not in {"0", "1"}:
            fail(f"{name}:{number}: label must be exactly 0 or 1 for {identity}")
        _validate_optional_numeric(row, "native_duration_s", name, number, identity)
        _validate_optional_numeric(row, "duration_sec", name, number, identity)
        _validate_optional_numeric(row, "available", name, number, identity)
    return list(reader.fieldnames), rows


def _validate_optional_numeric(
    row: dict[str, str], field: str, source: str, number: int, identity: str
) -> None:
    token = row.get(field, "")
    if token == "":
        return
    try:
        number_value = float(token)
    except ValueError:
        fail(f"{source}:{number}: malformed {field} for {identity}")
    if not math.isfinite(number_value):
        fail(f"{source}:{number}: non-finite {field} for {identity}")


def parse_duration_view(token: str) -> float | None:
    if not token:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", token)
    if match is None:
        fail(f"malformed duration_view: {token!r}")
    value = float(match.group(1))
    if not math.isfinite(value):
        fail(f"non-finite duration_view: {token!r}")
    return value


def strict_json(text: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        fail(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def load_freeze(path: Path, additional: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    try:
        payload = strict_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        fail(f"malformed accepted60 freeze: {exc}")
    if not isinstance(payload, dict):
        fail("accepted60 freeze root must be an object")
    if payload.get("status") != "frozen_for_measurement_only":
        fail("accepted60 freeze status is not frozen_for_measurement_only")
    selected = payload.get("selected")
    if not isinstance(selected, list):
        fail("accepted60 freeze missing selected list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in selected:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            fail("malformed accepted60 selected record")
        metadata = item["metadata"]
        identity = safe_id(str(metadata.get("id", "")))
        if identity in indexed:
            fail(f"accepted60 freeze has duplicate id {identity}")
        if metadata.get("duration_view") != "60s":
            fail(f"accepted60 freeze duration_view is not 60s for {identity}")
        for field in ("native_channels", "standardized_channels", "standardized_frames", "standardized_sr"):
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                fail(f"accepted60 freeze invalid {field} for {identity}")
        if item["native_channels"] != 2 or item["standardized_channels"] != 2:
            fail(f"accepted60 freeze channels are not the accepted stereo contract for {identity}")
        if item["standardized_frames"] != 2_646_000 or item["standardized_sr"] != 44_100:
            fail(f"accepted60 freeze framing differs from 2646000 frames at 44100 Hz for {identity}")
        standardized_path = item.get("standardized_path")
        if (
            not isinstance(standardized_path, str)
            or not standardized_path
            or "\x00" in standardized_path
            or not Path(standardized_path).is_absolute()
            or ".." in Path(standardized_path).parts
        ):
            fail(f"accepted60 freeze invalid standardized_path for {identity}")
        digest = item.get("standardized_file_sha256", "")
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            fail(f"accepted60 freeze invalid standardized hash for {identity}")
        if item["standardized_frames"] != 60 * item["standardized_sr"]:
            fail(f"accepted60 freeze context is not exactly 60 seconds for {identity}")
        indexed[identity] = item
    supplied = {row["id"]: row for row in additional}
    if set(supplied) != set(indexed):
        fail("additional60 ids do not exactly match accepted60 selected ids")
    for identity, row in supplied.items():
        if indexed[identity]["metadata"] != row:
            fail(f"additional60 metadata differs from accepted60 freeze for {identity}")
    return indexed


class UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def conditioning_tokens(row: dict[str, str]) -> set[str]:
    # group_id and condition_id deliberately share one namespace: identical aliases link.
    return {f"conditioning:{row[field]}" for field in ("group_id", "condition_id") if row.get(field)}


def reconcile(occurrences: dict[str, list[tuple[str, dict[str, str]]]]) -> None:
    for identity, records in occurrences.items():
        for field in ("label", "source_group", "group_id"):
            values = {row.get(field, "") for _, row in records}
            if len(values) != 1:
                detail = ", ".join(f"{source}={row.get(field, '')!r}" for source, row in records)
                fail(f"shared id conflict for {identity} field {field}: {detail}")


def duration_evidence(row: dict[str, str], freeze: dict[str, Any] | None) -> tuple[float | None, str]:
    if row.get("native_duration_s", "") != "":
        return float(row["native_duration_s"]), "registered_native_duration_s"
    if freeze is not None:
        return freeze["standardized_frames"] / freeze["standardized_sr"], "accepted_prior60_view"
    if row.get("duration_sec", "") != "":
        return float(row["duration_sec"]), "declared_duration_sec_not_native"
    return parse_duration_view(row.get("duration_view", "")), "declared_duration_view_not_native"


def channel_evidence(row: dict[str, str], freeze: dict[str, Any] | None) -> tuple[str, int | None, str]:
    for field in ("native_channels", "original_channels"):
        if row.get(field, "") != "":
            try:
                channels = int(row[field])
            except ValueError:
                fail(f"malformed {field} for {row['id']}")
            if channels <= 0:
                fail(f"invalid {field} for {row['id']}")
            return ("stereo" if channels == 2 else "mono" if channels == 1 else "multichannel"), channels, field
    if freeze is not None:
        channels = int(freeze["native_channels"])
        return ("stereo" if channels == 2 else "mono" if channels == 1 else "multichannel"), channels, "accepted60_freeze_recorded_native_channels"
    return "unknown", None, "no_native_channel_field"


def evaluation_denied(value: str) -> bool:
    return value.strip().lower() not in ALLOWED_EVALUATION


def screen(
    master10: Path,
    master30: Path,
    additional60: Path,
    accepted60_freeze: Path,
    accepted60_freeze_sha256: str,
    output: Path,
) -> dict[str, Any]:
    if HEX64.fullmatch(accepted60_freeze_sha256) is None:
        fail("accepted60 freeze caller SHA-256 must be canonical lowercase hex")
    paths = {
        "master10": safe_path(master10),
        "master30": safe_path(master30),
        "additional60": safe_path(additional60),
        "accepted60_freeze": safe_path(accepted60_freeze),
    }
    output = safe_path(output, output=True)
    implementation_paths = {
        "tool": safe_path(Path(__file__).absolute()),
        "tests": safe_path(Path(__file__).absolute().with_name("test_screen_equal30_candidates_v1.py")),
    }
    initial_bindings = {name: file_snapshot(path) for name, path in paths.items()}
    implementation_bindings = {name: file_snapshot(path) for name, path in implementation_paths.items()}
    if initial_bindings["accepted60_freeze"]["sha256"] != accepted60_freeze_sha256:
        fail("accepted60 freeze SHA-256 does not match caller-supplied value")
    tables: dict[str, list[dict[str, str]]] = {}
    fields: dict[str, list[str]] = {}
    for name in ("master10", "master30", "additional60"):
        fields[name], tables[name] = read_csv(paths[name], name)
    freeze = load_freeze(paths["accepted60_freeze"], tables["additional60"])

    occurrences: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for name, rows in tables.items():
        for row in rows:
            occurrences[row["id"]].append((name, row))
    reconcile(occurrences)

    all_ids = set(occurrences)
    union = UnionFind(all_ids)
    token_members: dict[str, list[str]] = defaultdict(list)
    for identity, records in occurrences.items():
        tokens: set[str] = set()
        for _, row in records:
            tokens.update(conditioning_tokens(row))
            if row.get("raw_sha256"):
                tokens.add(f"raw_sha256:{row['raw_sha256']}")
        for token in tokens:
            token_members[token].append(identity)
    for members in token_members.values():
        for identity in members[1:]:
            union.union(members[0], identity)

    component_members: dict[str, list[str]] = defaultdict(list)
    for identity in sorted(all_ids):
        component_members[union.find(identity)].append(identity)
    component_keys = {root: f"component_{index:06d}" for index, root in enumerate(sorted(component_members), 1)}
    protected_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for identity, records in occurrences.items():
        for source, row in records:
            if row.get("role", "") != "development":
                protected_by_root[union.find(identity)].append(
                    {"id": identity, "input": source, "role": row.get("role", "")}
                )

    chosen: dict[str, tuple[str, dict[str, str]]] = {
        row["id"]: ("master30", row) for row in tables["master30"]
    }
    for row in tables["additional60"]:
        chosen.setdefault(row["id"], ("additional60", row))

    row_results: list[dict[str, Any]] = []
    for identity in sorted(chosen):
        selected_from, row = chosen[identity]
        reasons: list[str] = []
        missing = [field for field in REQUIRED_IDENTITY if not row.get(field, "")]
        if missing:
            reasons.append("missing_required_metadata:" + ",".join(missing))
        role = row.get("role", "")
        if role != "development":
            reasons.append("role_not_development:" + (role or "missing"))
        for occurrence_input, occurrence in occurrences[identity]:
            occurrence_role = occurrence.get("role", "")
            original_role = occurrence.get("original_role", "")
            if "provisional" in occurrence_role.lower() or "provisional" in original_role.lower():
                reason = f"provisional_role:{occurrence_input}"
                if reason not in reasons:
                    reasons.append(reason)
            allowed = occurrence.get("evaluation_allowed", "")
            if allowed and evaluation_denied(allowed):
                reason = f"evaluation_not_allowed:{occurrence_input}:{allowed}"
                if reason not in reasons:
                    reasons.append(reason)
        if selected_from in {"master10", "master30"}:
            if "available" not in row or row.get("available", "") == "":
                reasons.append("missing_required_metadata:available")
            elif float(row["available"]) != 1.0:
                reasons.append("available_not_1")
        accepted = freeze.get(identity)
        duration, duration_basis = duration_evidence(row, accepted)
        if duration is None:
            reasons.append("missing_required_metadata:duration")
        elif duration < 30:
            reasons.append("native_or_accepted_context_duration_lt_30")
        if duration_basis in {"declared_duration_sec_not_native", "declared_duration_view_not_native"}:
            reasons.append("unverified_duration_evidence")
        root = union.find(identity)
        if protected_by_root[root]:
            reasons.append("component_touches_non_development")
        channel_status, channel_count, channel_basis = channel_evidence(row, accepted)
        row_results.append(
            {
                "id": identity,
                "label": row.get("label", ""),
                "source_group": row.get("source_group", ""),
                "group_id": row.get("group_id", ""),
                "role": role,
                "selected_metadata_input": selected_from,
                "input_occurrences": [source for source, _ in occurrences[identity]],
                "component_id": component_keys[root],
                "duration_context_s": duration,
                "duration_evidence": duration_basis,
                "full_native_duration_known": duration_basis == "registered_native_duration_s",
                "recorded_native_channel_status": channel_status,
                "recorded_native_channels": channel_count,
                "native_channel_evidence": channel_basis,
                "native_stereo_admitted": False,
                "duration_exposure_candidate": not reasons,
                "exclusion_reasons": reasons,
                "source_audio_path": row.get("source_audio_path", ""),
            }
        )

    rows_by_id = {row["id"]: row for row in row_results}
    components: list[dict[str, Any]] = []
    for root in sorted(component_members):
        members = sorted(component_members[root])
        member_set = set(members)
        components.append(
            {
                "component_id": component_keys[root],
                "members": members,
                "candidate_screen_members": sorted(member_set & set(rows_by_id)),
                "protected_relationships": sorted(
                    protected_by_root[root], key=lambda item: (item["id"], item["input"], item["role"])
                ),
                "link_tokens": sorted(
                    token for token, token_ids in token_members.items() if member_set.intersection(token_ids)
                ),
            }
        )

    source_summary: dict[str, dict[str, int]] = {}
    for source in sorted({row["source_group"] for row in row_results}):
        subset = [row for row in row_results if row["source_group"] == source]
        source_summary[source] = {
            "screened_rows": len(subset),
            "duration_exposure_candidates": sum(row["duration_exposure_candidate"] for row in subset),
            "excluded_rows": sum(not row["duration_exposure_candidate"] for row in subset),
            "recorded_stereo_rows_not_admitted": sum(row["recorded_native_channel_status"] == "stereo" for row in subset),
            "recorded_mono_rows": sum(row["recorded_native_channel_status"] == "mono" for row in subset),
            "unknown_native_channel_rows": sum(row["recorded_native_channel_status"] == "unknown" for row in subset),
            "label_0_rows": sum(row["label"] == "0" for row in subset),
            "label_1_rows": sum(row["label"] == "1" for row in subset),
            "candidate_recorded_stereo_rows_not_admitted": sum(
                row["duration_exposure_candidate"] and row["recorded_native_channel_status"] == "stereo"
                for row in subset
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_duration_exposure_candidate_screen_only",
        "publication": {"status": "completed", "exclusive_output": True},
        "scope": "metadata feasibility audit; not a frozen exact30 cohort or source-lineage/novelty certification",
        "feature_extraction_authorized": False,
        "classifier_fits": 0,
        "physical_validation_required": True,
        "native_stereo_admission_performed": False,
        "additional60_evidence_limit": "accepted prior 60-second view and recorded native channels; full-native duration remains unknown unless separately registered",
        "input_bindings": {
            name: {**initial_bindings[name], "rows": len(tables[name]) if name in tables else len(freeze)}
            for name in paths
        },
        "implementation_bindings": implementation_bindings,
        "selection_rule": "master30 preferred; additional60 contributes only ids absent from master30",
        "input_headers": fields,
        "counts": {
            "unique_ids_all_inputs": len(all_ids),
            "screened_rows": len(row_results),
            "duration_exposure_candidates": sum(row["duration_exposure_candidate"] for row in row_results),
            "excluded_rows": sum(not row["duration_exposure_candidate"] for row in row_results),
            "components": len(components),
            "protected_components": sum(bool(component["protected_relationships"]) for component in components),
        },
        "source_summaries": source_summary,
        "rows": row_results,
        "components": components,
    }
    for name, path in paths.items():
        require_unchanged(path, initial_bindings[name])
    for name, path in implementation_paths.items():
        require_unchanged(path, implementation_bindings[name])
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master10", type=Path, required=True)
    parser.add_argument("--master30", type=Path, required=True)
    parser.add_argument("--additional60", type=Path, required=True)
    parser.add_argument("--accepted60-freeze", type=Path, required=True)
    parser.add_argument("--accepted60-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen(
        args.master10, args.master30, args.additional60, args.accepted60_freeze,
        args.accepted60_freeze_sha256, args.output,
    )
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
