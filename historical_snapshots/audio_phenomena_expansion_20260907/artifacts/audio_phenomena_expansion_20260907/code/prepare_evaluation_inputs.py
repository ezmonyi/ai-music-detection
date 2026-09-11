#!/usr/bin/env python3
"""Build a collision-free F/H input table and explicit 30s family registry.

Only copies published extractor measure-name lists from the frozen extraction
contract. No scores or labels are inspected to select feature columns. Rich
input/audit fields stay in the extraction artifact, not in model inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--old-family-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.extraction_dir / "summary.json").read_text())
    contract = json.loads((args.extraction_dir / "contract.json").read_text())
    if not summary["complete_accounting"] or contract["preflight_only"] or contract["duration"] != 30:
        raise ValueError("Requires completed real 30-second extraction, not a preflight/long cohort")
    feature_path = args.extraction_dir / "features.csv"
    if sha(feature_path) != summary["features_csv_sha256"]:
        raise ValueError("Extraction table differs from its completed audit")
    for name, expected in contract["code_sha256"].items():
        if sha(Path(__file__).resolve().parent / name) != expected:
            raise ValueError("Local extractor code differs from frozen run: " + name)
    rows = list(csv.DictReader(feature_path.open(newline="")))
    if len(rows) != summary["expected"] or {r["id"] for r in rows} != set(contract["selected_ids"]):
        raise ValueError("Extraction identity mismatch")
    selected = ["id"] + [c for c in rows[0] if c.startswith(("F_", "H_", "M_"))]
    selected += ["new_extraction_status", "new_extraction_contract_hash"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "new_features_30s.csv"
    with destination.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get({"new_extraction_status": "extraction_status",
                                        "new_extraction_contract_hash": "extraction_contract_hash"}.get(c, c), "")
                             for c in selected})
    old = json.loads(args.old_family_config.read_text())["feature_sets"]["common8_30s"]
    old_specs = {k: {"state": "available", "columns": v, "eligibility": {},
                     "status_columns": [], "minimum_observed_fraction": 0.05}
                 for k, v in old["families"].items()}
    meanings = {"F": "Original-mix phase evolution and group delay; exploratory DSP descriptor",
                "H": "Original-mix tonal/chroma distribution and path; not chord correctness",
                "M": "Long-range content recurrence; separate >=45-second cohort required",
                "V": "Note-level pitch microstructure", "B": "Audible breath-event organization",
                "A": "Articulation and phoneme-note coordination", "T": "Source-conditioned timbral continuity"}
    new_specs = {}
    for key, meaning in meanings.items():
        new_specs[key] = {"state": "available" if key in ("F", "H") else "planned",
                          "columns": contract["feature_names"][key] if key in ("F", "H") else [],
                          "phenomenon": meaning, "eligibility": {},
                          "status_columns": [key + "_status"] if key in ("F", "H") else [],
                          "minimum_observed_fraction": 0.05}
    config = {"schema_version": 1, "cohort": "Exact historical 30s crop; S8/D/R/P plus original-mix F/H",
              "eligibility": old["eligibility"], "status_columns": old["status_columns"],
              "old_families": old_specs, "new_families": new_specs}
    family_path = args.output_dir / "families_30s_fh_v1.json"
    family_path.write_text(json.dumps(config, indent=2) + "\n")
    provenance = {"rows": len(rows), "extraction_contract_hash": contract["contract_hash"],
                  "extraction_table_sha256": sha(feature_path), "exported_table_sha256": sha(destination),
                  "family_json_sha256": sha(family_path), "old_family_json_sha256": sha(args.old_family_config),
                  "phenomenon_selection": "F/H predefined; M explicitly ineligible at 30s; V/B/A/T pending",
                  "labels_used_for_feature_selection": False}
    (args.output_dir / "preparation_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
