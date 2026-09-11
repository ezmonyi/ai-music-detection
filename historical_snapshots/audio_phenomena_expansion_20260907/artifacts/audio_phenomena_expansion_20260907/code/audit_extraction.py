#!/usr/bin/env python3
"""Independent table/item accounting audit without loading scientific runtimes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extraction-dir", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    root = a.extraction_dir
    summary = json.loads((root / "summary.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    rows = list(csv.DictReader((root / "features.csv").open(newline="")))
    metadata = {r["id"]: r for r in csv.DictReader(a.metadata.open(newline=""))}
    errors = []
    if sha(root / "features.csv") != summary["features_csv_sha256"]:
        errors.append("CSV checksum differs from summary")
    if summary["contract_hash"] != contract["contract_hash"]:
        errors.append("Summary and extraction contract disagree")
    if sha(a.metadata) != contract["metadata_sha256"]:
        errors.append("Metadata checksum differs from extraction contract")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(contract["selected_ids"]):
        errors.append("IDs duplicated/missing/extra")
    if len(rows) != summary["expected"] or not summary["complete_accounting"]:
        errors.append("Incomplete accounting")
    fields = ["extraction_status", "F_status", "H_status", "M_status", "source_audio_sha256",
              "analysis_waveform_sha256", "analysis_frames", "analysis_sr"]
    for row in rows:
        key = row["id"]
        path = root / "items" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        if not path.exists():
            errors.append("Missing item: " + key)
            continue
        item = json.loads(path.read_text())
        expected_row_hash = hashlib.sha256(json.dumps(metadata[key], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if item["input_row_hash"] != expected_row_hash or item["extraction_contract_hash"] != contract["contract_hash"]:
            errors.append("Item input/config mismatch: " + key)
        for field in fields:
            if str(item.get(field, "")) != str(row.get(field, "")):
                errors.append("Item/CSV mismatch: " + key + ":" + field)
        if item["extraction_status"] == "ok":
            for field in ("source_audio_sha256", "analysis_waveform_sha256"):
                if len(item.get(field, "")) != 64:
                    errors.append("Missing hash: " + key + ":" + field)
            if item["analysis_frames"] != round(contract["duration"] * 16000):
                errors.append("Analysis duration mismatch: " + key)
    result = {"passed": not errors, "row_count": len(rows), "item_records_checked": len(rows),
              "source_audio_rehashed_by_this_audit": False,
              "scope": "Independent metadata/table/item/config/recorded-hash accounting; extractor performed full source hashes",
              "extraction_contract_hash": contract["contract_hash"], "errors": errors,
              "auditor_sha256": sha(__file__)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
