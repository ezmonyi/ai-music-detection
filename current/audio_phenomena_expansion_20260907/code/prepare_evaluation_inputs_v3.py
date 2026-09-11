#!/usr/bin/env python3
"""Prepare an atomic schema-v3 exact-60-second pure-F/H/M input package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import prepare_evaluation_inputs_v2 as V2


SCHEMA_VERSION = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--extraction-audit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    extraction = V2.validate_extraction(
        args.extraction_dir.resolve(), args.extraction_audit.resolve(),
        args.metadata.resolve(), 60,
    )
    if extraction["contract"].get("metadata_sha256") != V2.sha256_file(args.metadata.resolve()):
        raise ValueError("Metadata SHA-256 differs from the frozen extraction contract")
    family_config = V2.build_family_config(60, extraction["contract"], None, "")
    family_config["schema_version"] = SCHEMA_VERSION
    if any(spec["state"] != "planned" for spec in family_config["old_families"].values()):
        raise AssertionError("60-second package exposed a fake old-family baseline")
    if tuple(code for code, spec in family_config["new_families"].items()
             if spec["state"] == "available") != ("F", "H", "M"):
        raise AssertionError("60-second package must expose exactly F/H/M")

    selected = ["id"] + [
        column for column in extraction["feature_header"]
        if column.startswith(("F_", "H_", "M_"))
    ] + ["new_extraction_status", "new_extraction_contract_hash"]
    if len(selected) != len(set(selected)):
        raise ValueError("Prepared feature columns contain duplicates")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent))
    try:
        shutil.copyfile(args.metadata.resolve(), temporary / "metadata_60s.csv")
        translated = []
        for row in extraction["feature_rows"]:
            current = {field: row.get(field, "") for field in selected}
            current["new_extraction_status"] = row["extraction_status"]
            current["new_extraction_contract_hash"] = row["extraction_contract_hash"]
            translated.append(current)
        V2.write_csv(temporary / "new_features_60s.csv", selected, translated)
        (temporary / "families_60s_v3.json").write_text(
            json.dumps(family_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        hashes = {
            path.name: V2.sha256_file(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared_not_authorized_for_scoring",
            "duration_seconds": 60,
            "rows": extraction["expected"],
            "id_sets_exact": True,
            "metadata_sha256_matches_extraction_contract": True,
            "extraction_contract_hash": extraction["contract"]["contract_hash"],
            "source_sha256": {
                "summary": V2.sha256_file(extraction["summary_path"]),
                "contract": V2.sha256_file(extraction["contract_path"]),
                "features": V2.sha256_file(extraction["feature_path"]),
                "independent_audit": V2.sha256_file(args.extraction_audit.resolve()),
                "metadata": V2.sha256_file(args.metadata.resolve()),
            },
            "outputs_sha256_before_audit": hashes,
            "active_old_families": [],
            "active_new_families": ["F", "H", "M"],
            "planned_old_families": ["S", "D", "R", "P"],
            "planned_new_families": ["V", "B", "A", "T"],
            "labels_used_for_feature_selection": False,
            "same_exact_60s_scope_for_all_fhm_candidates": True,
            "fake_30s_old_baseline": False,
            "atomic_directory_publish": True,
            "existing_output_refused": True,
            "scoring_authorized": False,
        }
        (temporary / "preparation_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
