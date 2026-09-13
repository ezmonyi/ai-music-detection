#!/usr/bin/env python3
"""Summarize frozen recording identities, not waveform-view or stem counts.

This reads validated manifests. It is deliberately not a substitute for the
separate physical-file/hash audit, whose result is supplied as an optional input.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE_NAMES = {
    "human_magnatagatune": "MagnaTagATune", "human_maestro_v3": "MAESTRO v3",
    "human_musicnet": "MusicNet", "human_medleydb": "MedleyDB", "human_moisesdb": "MoisesDB",
    "human_urmp": "URMP", "human_deam": "DEAM", "human_gtzan": "GTZAN",
    "human_hindustani_raag_hf": "Hindustani HF", "ai_audiox_third_party": "AudioX (provisional)",
    "ai_diffrhythm_pilot": "DiffRhythm (pilot)",
}

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-10s", type=Path, required=True)
    parser.add_argument("--metadata-30s", type=Path, required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--physical-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    short, long = read_csv(args.metadata_10s), read_csv(args.metadata_30s)
    if len({r["id"] for r in short}) != len(short) or len({r["id"] for r in long}) != len(long):
        raise ValueError("Metadata must contain one row per identity per view")
    if not {r["id"] for r in long}.issubset({r["id"] for r in short}):
        raise ValueError("Long cohort contains an identity absent from the broad cohort")
    latest = {}
    with args.materialized.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[row["item_id"]] = row
    selected_ids = {r["id"] for r in short}
    if not set(latest).issubset(selected_ids):
        raise ValueError("Materialization ledger contains identities outside the frozen union")
    for row in short:
        if row["acquisition"] == "expansion" and row["available"] == "1":
            retained = latest.get(row["id"], {})
            if retained.get("status") != "success" or retained.get("native_sha256") != row["raw_sha256"]:
                raise ValueError("Registry/retained-native identity mismatch: " + row["id"])
    grouped, long_grouped = defaultdict(list), defaultdict(list)
    for row in short:
        grouped[(row["source_group"], row["label"])].append(row)
    for row in long:
        long_grouped[(row["source_group"], row["label"])].append(row)
    sources = []
    for (source, label), rows in sorted(grouped.items(), key=lambda pair: (int(pair[0][1]), pair[0][0])):
        present = [r for r in rows if r["available"] == "1"]
        long_rows = long_grouped[(source, label)]
        sources.append({
            "source": source, "claimed_label": "Human" if label == "0" else "AI",
            "selected_recordings": len(rows), "available_10s": len(present),
            "eligible_available_30s": len(long_rows),
            "identity_groups_selected": len({r["group_id"] for r in rows}),
            "unique_retained_native_hashes": len({r["raw_sha256"] for r in present if r["raw_sha256"]}),
            "roles": dict(Counter(r["role"] for r in rows)),
            "native_sample_rates_available": dict(Counter(r["native_sample_rate_hz"] for r in present)),
            "development_rows": sum(r["role"] == "development" and r["available"] == "1" for r in rows),
            "development_groups": len({r["group_id"] for r in present if r["role"] == "development"}),
        })
    inputs = {str(p): digest(p) for p in [args.metadata_10s, args.metadata_30s, args.materialized]}
    audit = None
    if args.physical_audit:
        audit = json.loads(args.physical_audit.read_text())
        inputs[str(args.physical_audit)] = digest(args.physical_audit)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": inputs,
        "counting_unit": "selected recording identity; not view, stem, or independent condition",
        "physical_verification": audit,
        "selected_recordings": len(short),
        "available_10s": sum(r["available"] == "1" for r in short),
        "eligible_available_30s": len(long),
        "all_selected_materialized_according_to_registry": all(r["available"] == "1" for r in short),
        "claimed_labels": dict(Counter("Human" if r["label"] == "0" else "AI" for r in short)),
        "roles": dict(Counter(r["role"] for r in short)),
        "global_group_count": len({r["group_id"] for r in short}),
        "materialization_latest_status": dict(Counter(r["status"] for r in latest.values())),
        "sources": sources,
        "warnings": [
            "AudioX labels are provisional; never interpret their scores as verified accuracy.",
            "DiffRhythm has 50 files but only 10 conditioning groups.",
            "Thirty-second membership differs from ten-second membership; this is not a matched duration comparison.",
            "Retained-byte hashes do not detect every transcoded duplicate, cover, or re-recording.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "corpus_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Frozen corpus inventory", "", f"Snapshot: {payload['generated_utc']}", "",
             f"Selected recording identities: **{len(short):,}**. Available 10-second views: "
             f"**{payload['available_10s']:,}**. Duration-qualified 30-second recordings: **{len(long):,}**.", "",
             "These are recording identities, not independent prompt groups or counts of all generated files.", "",
             "| Source | Claimed class | Selected | Available 10s | Eligible 30s | Development rows/groups | Roles |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in sources:
        roles = "; ".join(f"{key}: {value}" for key, value in row["roles"].items())
        lines.append(f"| {row['source']} | {row['claimed_label']} | {row['selected_recordings']} | "
                     f"{row['available_10s']} | {row['eligible_available_30s']} | "
                     f"{row['development_rows']}/{row['development_groups']} | {roles} |")
    lines.extend(["", *[f"- {warning}" for warning in payload["warnings"]], ""])
    (args.output_dir / "corpus_inventory.md").write_text("\n".join(lines))
    latex = [r"\begin{longtable}{p{42mm}lrrrr}",
             r"\caption{Frozen source inventory. Counts refer to recording identities, not stems or independent conditions.}\\",
             r"\toprule Source & Class & Selected & 10 s & 30 s & Dev. groups \\", r"\midrule", r"\endfirsthead",
             r"\toprule Source & Class & Selected & 10 s & 30 s & Dev. groups \\", r"\midrule", r"\endhead"]
    for row in sources:
        display = dict(row, source=SOURCE_NAMES.get(row["source"], row["source"]))
        latex.append(" & ".join(latex_escape(display[key]) for key in ["source", "claimed_label", "selected_recordings", "available_10s", "eligible_available_30s", "development_groups"]) + r" \\")
    latex.extend([r"\bottomrule", r"\end{longtable}", ""])
    (args.output_dir / "corpus_inventory.tex").write_text("\n".join(latex))
    print(json.dumps({key: payload[key] for key in ["selected_recordings", "available_10s", "eligible_available_30s", "all_selected_materialized_according_to_registry"]}))


if __name__ == "__main__":
    main()
