#!/usr/bin/env python3
"""Atomically prepare schema-v2 10 s or 60 s evaluation inputs.

The 10-second contract exposes the established S/D/R families, marks P planned,
and exposes new F/H while M remains planned because 10 seconds cannot measure
the intended long-range recurrence construct.  The 60-second contract is a
separate pure-F/H/M exploratory context: every old family is planned, so a
30-versus-60 change in old descriptors cannot masquerade as an M increment.

No labels are inspected for feature selection.  The output is refused if its
directory already exists and is published by one atomic directory rename.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


SCHEMA_VERSION = 2
OLD_CODES = ("S", "D", "R", "P")
NEW_CODES = ("F", "H", "M", "V", "B", "A", "T")
MEANINGS = {
    "F": "Original-mix phase evolution and group delay; exploratory DSP descriptor",
    "H": "Original-mix tonal/chroma distribution and path; not chord correctness",
    "M": "Long-range content recurrence over an explicitly long-duration excerpt",
    "V": "Note-level pitch microstructure",
    "B": "Audible breath-event organization",
    "A": "Articulation and phoneme-note coordination",
    "T": "Source-conditioned timbral continuity",
}
MINIMUM_OBSERVED_FRACTION = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def unique_ids(rows: list[dict[str, str]], column: str, context: str) -> set[str]:
    if not rows or column not in rows[0]:
        raise ValueError(f"{context} lacks ID column {column!r}")
    values = [str(row[column]).strip() for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{context} contains blank IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{context} IDs are not unique")
    return set(values)


def validate_extraction(
    extraction_dir: Path, extraction_audit_path: Path, metadata_path: Path, duration: int,
) -> dict[str, Any]:
    summary_path = extraction_dir / "summary.json"
    contract_path = extraction_dir / "contract.json"
    feature_path = extraction_dir / "features.csv"
    for path in (summary_path, contract_path, feature_path, extraction_audit_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    contract = read_json(contract_path)
    audit = read_json(extraction_audit_path)
    if not bool(summary.get("complete_accounting")):
        raise ValueError("Extraction summary is not complete")
    if bool(contract.get("preflight_only")):
        raise ValueError("Preflight extraction cannot become evaluation input")
    if float(contract.get("duration", -1)) != float(duration):
        raise ValueError("Extraction duration differs from requested preparation duration")
    contract_hash = str(contract.get("contract_hash", ""))
    if not contract_hash or summary.get("contract_hash") != contract_hash:
        raise ValueError("Extraction summary and contract hashes disagree")
    if not bool(audit.get("passed")) or audit.get("errors"):
        raise ValueError("Independent extraction audit did not pass cleanly")
    if str(audit.get("extraction_contract_hash", "")) != contract_hash:
        raise ValueError("Independent audit describes a different extraction contract")
    if sha256_file(feature_path) != summary.get("features_csv_sha256"):
        raise ValueError("Extraction feature table differs from completed summary")

    code_hashes = contract.get("code_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("Extraction contract code_sha256 must be an object")
    code_root = Path(__file__).resolve().parent
    for name, expected in code_hashes.items():
        local = code_root / str(name)
        if not local.is_file() or sha256_file(local) != str(expected):
            raise ValueError(f"Local extractor code differs from frozen extraction: {name}")

    feature_header, feature_rows = read_csv_rows(feature_path)
    metadata_header, metadata_rows = read_csv_rows(metadata_path)
    feature_ids = unique_ids(feature_rows, "id", "extraction table")
    metadata_ids = unique_ids(metadata_rows, "id", "metadata")
    selected_ids = contract.get("selected_ids")
    if not isinstance(selected_ids, list) or any(not str(value).strip() for value in selected_ids):
        raise ValueError("Extraction contract selected_ids must be a non-empty ID list")
    selected_set = {str(value) for value in selected_ids}
    expected = int(summary.get("expected", -1))
    if len(feature_rows) != expected or len(selected_ids) != len(selected_set) or len(selected_set) != expected:
        raise ValueError("Extraction row accounting is inconsistent")
    if feature_ids != selected_set or metadata_ids != selected_set:
        raise ValueError("Extraction, contract, and metadata ID sets differ")
    if int(audit.get("row_count", -1)) != expected:
        raise ValueError("Independent audit row count differs from extraction")

    required_metadata = {"id", "source_group", "group_id", "role", "duration_view", "available"}
    if not required_metadata.issubset(metadata_header):
        raise ValueError(f"Metadata lacks {sorted(required_metadata - set(metadata_header))}")
    view = f"{duration}s"
    if any(row["duration_view"].strip() != view for row in metadata_rows):
        raise ValueError(f"Metadata contains rows outside duration_view={view}")
    if any(row["available"].strip().lower() not in {"1", "1.0", "true", "yes", "available"}
           for row in metadata_rows):
        raise ValueError("Metadata includes unavailable rows")
    for field in ("source_group", "group_id", "role"):
        if any(not row[field].strip() for row in metadata_rows):
            raise ValueError(f"Metadata contains blank {field}")

    feature_names = contract.get("feature_names")
    if not isinstance(feature_names, dict) or any(code not in feature_names for code in ("F", "H", "M")):
        raise ValueError("Extraction contract must define F/H/M feature_names")
    active_new = ("F", "H") if duration == 10 else ("F", "H", "M")
    for code in active_new:
        names = feature_names[code]
        if not isinstance(names, list) or not names or not set(names).issubset(feature_header):
            raise ValueError(f"Active family {code} feature list is absent from extraction table")
        if f"{code}_status" not in feature_header:
            raise ValueError(f"Active family {code} lacks its status column")
    if "extraction_status" not in feature_header or "extraction_contract_hash" not in feature_header:
        raise ValueError("Extraction table lacks row-level extraction provenance")
    if any(not row["extraction_status"].strip() for row in feature_rows):
        raise ValueError("Extraction table contains blank extraction_status")
    if any(row["extraction_contract_hash"] != contract_hash for row in feature_rows):
        raise ValueError("Extraction rows carry a different contract hash")
    return {
        "summary_path": summary_path,
        "contract_path": contract_path,
        "feature_path": feature_path,
        "summary": summary,
        "contract": contract,
        "feature_header": feature_header,
        "feature_rows": feature_rows,
        "metadata_header": metadata_header,
        "metadata_rows": metadata_rows,
        "expected": expected,
        "active_new": active_new,
    }


def family_spec(
    state: str, columns: list[str], *, phenomenon: str = "", status_columns: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": state,
        "columns": columns,
        "eligibility": {},
        "status_columns": status_columns or [],
        "minimum_observed_fraction": MINIMUM_OBSERVED_FRACTION,
    }
    if phenomenon:
        result["phenomenon"] = phenomenon
    return result


def build_family_config(
    duration: int, extraction_contract: dict[str, Any], old_family_config: dict[str, Any] | None,
    old_feature_set: str,
) -> dict[str, Any]:
    feature_names = extraction_contract["feature_names"]
    if duration == 10:
        if old_family_config is None:
            raise ValueError("10-second preparation requires an old-family config")
        sets = old_family_config.get("feature_sets", {})
        if old_feature_set not in sets:
            raise ValueError(f"Old feature config lacks {old_feature_set}")
        old = sets[old_feature_set]
        if not bool(old.get("selection_eligible")):
            raise ValueError(f"Old feature set {old_feature_set} is not selection eligible")
        old_columns = old.get("families", {})
        if any(code not in old_columns for code in ("S", "D", "R")):
            raise ValueError("10-second old config must define S/D/R")
        old_specs = {
            code: family_spec("available", list(old_columns[code])) for code in ("S", "D", "R")
        }
        old_specs["P"] = family_spec("planned", [])
        active_new = {"F", "H"}
        eligibility = old.get("eligibility", {})
        status_columns = list(old.get("status_columns", []))
        cohort = "Exact established 10s crop; S/D/R plus original-mix F/H; P and M unavailable"
    elif duration == 60:
        old_specs = {code: family_spec("planned", []) for code in OLD_CODES}
        active_new = {"F", "H", "M"}
        eligibility = {"equals": {"duration_view": "60s", "available": 1}}
        status_columns = []
        cohort = "Separate eligible 60s exploratory crop; pure original-mix F/H/M context"
    else:
        raise ValueError("Only 10-second and 60-second preparations are supported")
    new_specs = {
        code: family_spec(
            "available" if code in active_new else "planned",
            list(feature_names[code]) if code in active_new else [],
            phenomenon=MEANINGS[code],
            status_columns=[f"{code}_status"] if code in active_new else [],
        )
        for code in NEW_CODES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "eligibility": eligibility,
        "status_columns": status_columns,
        "old_families": old_specs,
        "new_families": new_specs,
    }


def validate_old_features(
    old_feature_path: Path | None, metadata_ids: set[str], family_config: dict[str, Any], duration: int,
) -> dict[str, Any]:
    if duration != 10:
        if old_feature_path:
            raise ValueError("60-second pure-F/H/M preparation must not accept old features")
        return {}
    if old_feature_path is None or not old_feature_path.is_file():
        raise FileNotFoundError("10-second preparation requires authoritative old features")
    header, rows = read_csv_rows(old_feature_path)
    id_column = "item_id" if "item_id" in header else "id" if "id" in header else ""
    old_ids = unique_ids(rows, id_column, "old feature table")
    if old_ids != metadata_ids:
        raise ValueError("Old feature and metadata ID sets differ")
    required = {
        column
        for code in ("S", "D", "R")
        for column in family_config["old_families"][code]["columns"]
    } | set(family_config["status_columns"])
    if not required.issubset(header):
        raise ValueError(f"Old feature table lacks {sorted(required - set(header))}")
    return {"header": header, "rows": rows, "id_column": id_column}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, choices=(10, 60), required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--extraction-audit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--old-family-config", type=Path)
    parser.add_argument("--old-feature-set", default="common8_10s")
    parser.add_argument("--old-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    extraction = validate_extraction(
        args.extraction_dir.resolve(), args.extraction_audit.resolve(),
        args.metadata.resolve(), args.duration,
    )
    old_config = read_json(args.old_family_config.resolve()) if args.old_family_config else None
    if args.duration == 10 and old_config is None:
        raise ValueError("10-second preparation requires --old-family-config")
    if args.duration == 60 and (args.old_family_config or args.old_features):
        raise ValueError("60-second pure-F/H/M preparation refuses old-family inputs")
    family_config = build_family_config(
        args.duration, extraction["contract"], old_config, args.old_feature_set
    )
    metadata_ids = {row["id"] for row in extraction["metadata_rows"]}
    validate_old_features(args.old_features.resolve() if args.old_features else None,
                          metadata_ids, family_config, args.duration)

    selected = ["id"] + [
        column for column in extraction["feature_header"] if column.startswith(("F_", "H_", "M_"))
    ] + ["new_extraction_status", "new_extraction_contract_hash"]
    if len(selected) != len(set(selected)):
        raise ValueError("Prepared new-feature column list contains duplicates")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        metadata_name = f"metadata_{args.duration}s.csv"
        new_name = f"new_features_{args.duration}s.csv"
        family_name = f"families_{args.duration}s_v2.json"
        shutil.copyfile(args.metadata.resolve(), temporary / metadata_name)
        if args.duration == 10:
            shutil.copyfile(args.old_features.resolve(), temporary / "old_features_10s.csv")
        translated_rows = []
        for row in extraction["feature_rows"]:
            translated = {field: row.get(field, "") for field in selected}
            translated["new_extraction_status"] = row["extraction_status"]
            translated["new_extraction_contract_hash"] = row["extraction_contract_hash"]
            translated_rows.append(translated)
        write_csv(temporary / new_name, selected, translated_rows)
        (temporary / family_name).write_text(
            json.dumps(family_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        data_hashes = {
            path.name: sha256_file(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared_not_authorized_for_scoring",
            "duration_seconds": args.duration,
            "rows": extraction["expected"],
            "id_sets_exact": True,
            "extraction_contract_hash": extraction["contract"]["contract_hash"],
            "source": {
                "summary": {"path": str(extraction["summary_path"]),
                            "sha256": sha256_file(extraction["summary_path"])},
                "contract": {"path": str(extraction["contract_path"]),
                             "sha256": sha256_file(extraction["contract_path"])},
                "features": {"path": str(extraction["feature_path"]),
                             "sha256": sha256_file(extraction["feature_path"])},
                "independent_audit": {"path": str(args.extraction_audit.resolve()),
                                      "sha256": sha256_file(args.extraction_audit.resolve())},
                "metadata": {"path": str(args.metadata.resolve()),
                             "sha256": sha256_file(args.metadata.resolve())},
                "old_features": (
                    {"path": str(args.old_features.resolve()),
                     "sha256": sha256_file(args.old_features.resolve())}
                    if args.old_features else None
                ),
                "old_family_config": (
                    {"path": str(args.old_family_config.resolve()),
                     "sha256": sha256_file(args.old_family_config.resolve()),
                     "feature_set": args.old_feature_set}
                    if args.old_family_config else None
                ),
            },
            "outputs_sha256_before_audit": data_hashes,
            "active_old_families": [
                code for code in OLD_CODES if family_config["old_families"][code]["state"] == "available"
            ],
            "active_new_families": list(extraction["active_new"]),
            "planned_old_families": [
                code for code in OLD_CODES if family_config["old_families"][code]["state"] == "planned"
            ],
            "planned_new_families": [
                code for code in NEW_CODES if family_config["new_families"][code]["state"] == "planned"
            ],
            "labels_used_for_feature_selection": False,
            "atomic_directory_publish": True,
            "existing_output_refused": True,
            "scoring_authorized": False,
        }
        (temporary / "preparation_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
