#!/usr/bin/env python3
"""Export descriptive Mureka frozen-v4 transfer tables after a passed audit.

The exporter reads an already scored six-file publication.  It does not fit,
score, select, rank, decode media, or reinterpret the frozen endpoints.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


FAMILY_ORDER = ("S", "D", "R", "P", "F", "H", "M")
COMBINATIONS = tuple("+".join(parts) for size in range(1, 8)
                     for parts in itertools.combinations(FAMILY_ORDER, size))
CAPS = ("25", "50", "100", "200", "all")
FOLDS = tuple(str(i) for i in range(5))
INDEX = ("combination", "quantity", "fold_index", "fold_uid", "model_sha256",
         "candidate_key", "train_id_set_sha256", "test_id_set_sha256")
SUMMARY = INDEX + ("synthetic_test_only", "rows", "unique_ids", "unique_groups",
                   "tp", "fn", "threshold", "ai_sensitivity", "false_negative_rate")
META = ("id", "label", "source_group", "group_id", "role", "acquisition_role")
SCORED_FILES = frozenset({"publication_manifest.json", "model_index.csv",
                          "identity_roles.csv", "predictions.csv",
                          "per_model_sensitivity.csv", "scoring_receipt.json"})
PUBLICATION_FILES = SCORED_FILES - {"publication_manifest.json"}
COMPACT_MIRROR_FILES = SCORED_FILES - {"predictions.csv"}
OVERVIEW = ("S", "D", "R", "P", "F", "H", "M", "S+D+R+P", "S+D+R+P+F",
            "S+D+R+P+H", "S+D+R+P+M", "S+D+R+P+F+H+M")
AUDITOR = Path(__file__).with_name("audit_mureka60_transfer_v1.py")
HEX64 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key {key}: {path}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value {value}: {path}")

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=constant)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_csv(path, header):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == tuple(header), f"CSV schema changed: {path}")
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows),
            f"Empty or malformed CSV: {path}")
    return rows


def integer(row, name):
    text = row.get(name, "")
    require(re.fullmatch(r"0|[1-9][0-9]*", text) is not None, f"Invalid integer {name}")
    return int(text)


def finite(row, name):
    try:
        value = float(row[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {name}") from exc
    require(math.isfinite(value), f"Nonfinite numeric field {name}")
    return value


def snapshot_inputs(results_dir, audit_path):
    paths = [audit_path, AUDITOR.resolve(), Path(__file__).resolve()]
    require(results_dir.is_dir() and results_dir.resolve() == results_dir,
            "Results directory must be an existing canonical absolute directory")
    names = {p.name for p in results_dir.iterdir()}
    require(names == SCORED_FILES or names == COMPACT_MIRROR_FILES,
            "Results directory must contain the scored publication or its five-file verified compact mirror")
    paths.extend(results_dir / name for name in sorted(names))
    hashes = {}
    for path in paths:
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.resolve() == path, f"Missing or noncanonical regular input: {path}")
        hashes[str(path)] = sha(path)
    return hashes


def audit_binding(path, observed, checked, verified_mirror):
    exact = checked.get(str(path))
    if exact is not None:
        require(exact == observed, f"Independent audit hash mismatch: {path}")
        return str(path)
    require(verified_mirror, f"Independent audit does not bind current path: {path}")
    matches = [(name, value) for name, value in checked.items()
               if Path(name).name == path.name and value == observed]
    require(len(matches) == 1, f"Verified mirror has no unique audited basename/hash match: {path.name}")
    return matches[0][0]


def validate_gate(results_dir, audit_path, synthetic, verified_mirror, hashes):
    report = strict_json(audit_path)
    manifest_path = results_dir / "publication_manifest.json"
    manifest = strict_json(manifest_path)
    rows = report.get("rows")
    expected = {
        "schema_version": 1, "status": "passed", "synthetic_test_only": synthetic,
        "rows": rows, "unique_ids": rows, "positive_class_count": rows,
        "model_instances": 3175, "combinations": 127, "caps": list(CAPS),
        "fold_models_per_cap": 5, "prediction_rows": 3175 * rows if type(rows) is int else None,
        "sensitivity_summary_rows": 3175, "denominator_per_model": rows,
        "all_ids_in_every_model": True,
        "all_decisions_match_published_and_independent_scores": True,
        "model_fitting_performed": False, "new_source_transform_learning_performed": False,
        "source_media_rehashed": False, "source_media_decoded": False,
        "endpoints": ["ai_sensitivity", "false_negative_rate"],
    }
    require(type(rows) is int and ((synthetic and 0 < rows <= 16) or (not synthetic and rows == 500)),
            "Real export requires 500 rows; a small fixture requires --synthetic-test-only")
    for key, wanted in expected.items():
        require(report.get(key) == wanted, f"Independent audit gate mismatch: {key}")
    require((not synthetic and report["prediction_rows"] == 1_587_500) or synthetic,
            "Real export requires 1,587,500 audited predictions")
    require(report.get("auditor_sha256") == hashes[str(AUDITOR.resolve())],
            "Audit was not produced by the current independent auditor")
    require(report.get("publication_manifest_sha256") == hashes[str(manifest_path)],
            "Audit does not bind the current publication manifest")
    checked = report.get("checked_files_sha256")
    require(isinstance(checked, dict) and checked, "Audit checked-file inventory is missing")
    provenance_paths = {}
    for name in sorted(SCORED_FILES):
        path = results_dir / name
        if path.is_file():
            provenance_paths[name] = audit_binding(path, hashes[str(path)], checked, verified_mirror)
        else:
            require(name == "predictions.csv" and verified_mirror,
                    f"Missing local scored artifact: {name}")
            record = manifest.get("files", {}).get(name, {})
            matches = [(original, value) for original, value in checked.items()
                       if Path(original).name == name and value == record.get("sha256")]
            require(len(matches) == 1,
                    "Compact mirror omission requires one audit-bound original predictions hash")
            provenance_paths[name] = matches[0][0]

    fixed_manifest = {
        "status": "scored", "synthetic_test_only": synthetic,
        "classifier_fitted": False, "original_v4_development_admission": False,
        "unique_new_ids": rows, "model_instances": 3175,
        "prediction_rows": 3175 * rows, "summary_rows": 3175,
    }
    require(set(manifest) == set(fixed_manifest) | {"contract_sha256", "files"},
            "Publication manifest schema changed")
    require(all(manifest.get(key) == value for key, value in fixed_manifest.items()),
            "Publication manifest scope or counts changed")
    require(isinstance(manifest.get("contract_sha256"), str)
            and HEX64.fullmatch(manifest["contract_sha256"]), "Invalid publication contract hash")
    require(set(manifest.get("files", {})) == PUBLICATION_FILES,
            "Publication manifest file inventory changed")
    for name, record in manifest["files"].items():
        path = results_dir / name
        require(set(record) == {"sha256", "bytes"} and type(record["bytes"]) is int
                and record["bytes"] >= 0, f"Publication manifest record mismatch: {name}")
        if path.is_file():
            require(record["sha256"] == hashes[str(path)] and record["bytes"] == path.stat().st_size,
                    f"Publication manifest record mismatch: {name}")
    require(manifest["files"]["per_model_sensitivity.csv"]["sha256"]
            == hashes[str(results_dir / "per_model_sensitivity.csv")],
            "Sensitivity table hash mismatch")
    return report, manifest, provenance_paths, rows


def validate_identities(results_dir, rows):
    identities = read_csv(results_dir / "identity_roles.csv", META)
    require(len(identities) == rows and len({r["id"] for r in identities}) == rows,
            "Identity table does not contain the exact audit denominator")
    require(all(r["id"].strip() and r["group_id"].strip() and r["label"] == "1"
                and r["source_group"] == "Mureka_v9"
                and r["role"] == "external_generator_unscored"
                and r["acquisition_role"] == "reserved_unscored" for r in identities),
            "Identity roles or positive-class labels changed")


def validate_and_aggregate(results_dir, rows, synthetic):
    index_rows = read_csv(results_dir / "model_index.csv", INDEX)
    summary_rows = read_csv(results_dir / "per_model_sensitivity.csv", SUMMARY)
    require(len(index_rows) == len(summary_rows) == 3175, "Exactly 3,175 model rows are required")
    expected = [(combo, cap, fold) for combo in COMBINATIONS for cap in CAPS for fold in FOLDS]
    triples = [(r["combination"], r["quantity"], r["fold_index"]) for r in index_rows]
    require(len(set(triples)) == 3175 and set(triples) == set(expected),
            "Missing, duplicate, or extra combination/cap/fold cells")
    require([tuple(r[k] for k in INDEX) for r in summary_rows]
            == [tuple(r[k] for k in INDEX) for r in index_rows],
            "Summary model identities differ from model index")
    index_by_key = {key: row for key, row in zip(triples, index_rows)}
    summary_by_key = {key: row for key, row in zip(triples, summary_rows)}
    fold_refs = {}
    for key, model in index_by_key.items():
        combo, cap, fold = key
        require(all(model[name].strip() for name in INDEX), "Blank model or fold reference")
        require(all(HEX64.fullmatch(model[name]) for name in
                    ("model_sha256", "train_id_set_sha256", "test_id_set_sha256")),
                "Invalid model or fold hash reference")
        reference = tuple(model[name] for name in
                          ("fold_uid", "train_id_set_sha256", "test_id_set_sha256"))
        require(fold_refs.setdefault((cap, fold), reference) == reference,
                "Fold reference varies across family subsets")
        summary = summary_by_key[key]
        require(summary["synthetic_test_only"] == str(synthetic)
                and integer(summary, "rows") == rows and integer(summary, "unique_ids") == rows
                and integer(summary, "unique_groups") > 0 and finite(summary, "threshold") == 0.5,
                "Summary scope, denominator, or threshold changed")
        tp, fn = integer(summary, "tp"), integer(summary, "fn")
        sensitivity, fnr = finite(summary, "ai_sensitivity"), finite(summary, "false_negative_rate")
        require(tp + fn == rows and 0 <= tp <= rows
                and math.isclose(sensitivity, tp / rows, rel_tol=0, abs_tol=1e-15)
                and math.isclose(fnr, fn / rows, rel_tol=0, abs_tol=1e-15),
                "Summary TP/FN or endpoint arithmetic changed")
    require(len(set(ref[0] for ref in fold_refs.values())) == 25,
            "The 25 cap/fold identities must be distinct")

    output = []
    for combo in COMBINATIONS:
        for cap in CAPS:
            group = [summary_by_key[(combo, cap, fold)] for fold in FOLDS]
            values = [finite(row, "ai_sensitivity") for row in group]
            fnrs = [finite(row, "false_negative_rate") for row in group]
            tps = [integer(row, "tp") for row in group]
            fns = [integer(row, "fn") for row in group]
            output.append({
                "combination": combo, "quantity": cap, "stored_fold_models": 5,
                "songs_per_model": rows, "model_song_applications": 5 * rows,
                "mean_sensitivity": sum(values) / 5,
                "min_sensitivity": min(values), "max_sensitivity": max(values),
                "mean_fnr": sum(fnrs) / 5, "mean_tp": sum(tps) / 5,
                "mean_fn": sum(fns) / 5,
            })
    require(len(output) == 635, "Expected all 635 family/cap cells")
    return output


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.6f}"


def overview_rows(rows):
    lookup = {(r["combination"], r["quantity"]): r for r in rows}
    return [{"combination": combo, **{cap: lookup[combo, cap]["mean_sensitivity"] for cap in CAPS}}
            for combo in OVERVIEW]


def render_markdown(rows, overview, synthetic):
    mark = "Synthetic test fixture. These values are not research results.\n\n" if synthetic else ""
    lines = ["# Mureka frozen-v4 transfer sensitivity tables", "", mark.rstrip(), "" if mark else "",
             "Each stored fold model evaluates the same set of AI songs. The table summarizes five dependent stored-model applications for each family subset and cap. Minima and maxima describe those five models and are not confidence intervals.", "",
             "## Overview: mean sensitivity", "", "| Combination | 25 | 50 | 100 | 200 | All |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in overview:
        lines.append("| " + row["combination"] + " | " + " | ".join(fmt(row[c]) for c in CAPS) + " |")
    lines.extend(["", "## All 635 family-subset and cap cells", "",
                  "| Combination | Cap | Models | Songs per model | Applications | Mean sensitivity | Min sensitivity | Max sensitivity | Mean FNR | Mean TP | Mean FN |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['combination']} | {row['quantity']} | 5 | {row['songs_per_model']} | {row['model_song_applications']} | {fmt(row['mean_sensitivity'])} | {fmt(row['min_sensitivity'])} | {fmt(row['max_sensitivity'])} | {fmt(row['mean_fnr'])} | {row['mean_tp']:.1f} | {row['mean_fn']:.1f} |")
    return "\n".join(line for i, line in enumerate(lines) if line or i == 0 or lines[i-1]) + "\n"


def tex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("+", r"{+}").replace("_", r"\_").replace("%", r"\%")


def render_latex(rows, synthetic):
    lines = [r"% Requires booktabs and longtable.",
             r"% Five dependent stored fold models evaluate the same AI songs in every cell.",
             r"% Minima and maxima are descriptive and are not confidence intervals."]
    if synthetic:
        lines.append(r"% SYNTHETIC TEST FIXTURE. VALUES ARE NOT RESEARCH RESULTS.")
    lines.extend([r"\begingroup\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
                  r"\begin{longtable}{@{}llrrrrrrrrr@{}}",
                  r"\caption{Mureka frozen-v4 transfer sensitivity across five stored fold models for each family subset and cap. Each model evaluates the same AI songs. Minima and maxima are descriptive, not confidence intervals.}\label{tab:mureka-transfer-all}\\",
                  r"\toprule Combination & Cap & Models & Songs & Applications & Mean sensitivity & Min & Max & Mean FNR & Mean TP & Mean FN \\ \midrule\endfirsthead",
                  r"\toprule Combination & Cap & Models & Songs & Applications & Mean sensitivity & Min & Max & Mean FNR & Mean TP & Mean FN \\ \midrule\endhead",
                  r"\bottomrule\endlastfoot"])
    for row in rows:
        lines.append(" & ".join((tex_escape(row["combination"]), tex_escape(row["quantity"]), "5",
                                 str(row["songs_per_model"]), str(row["model_song_applications"]),
                                 fmt(row["mean_sensitivity"]), fmt(row["min_sensitivity"]),
                                 fmt(row["max_sensitivity"]), fmt(row["mean_fnr"]),
                                 f"{row['mean_tp']:.1f}", f"{row['mean_fn']:.1f}")) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def publish_directory(source, destination):
    """Use the platform's atomic directory rename with an exclusive target."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = fn(os.fsencode(source), os.fsencode(destination), 4)
    elif sys.platform.startswith("linux"):
        try:
            fn = libc.renameat2
        except AttributeError as exc:
            raise ValueError("Linux libc lacks renameat2 for atomic no-replace publication") from exc
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise ValueError("Atomic no-replace publication requires Linux or macOS")
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise ValueError("Refusing an existing output directory")
        raise OSError(code, os.strerror(code), str(destination))


def export(results_dir, audit_path, output_dir, synthetic=False, verified_mirror=False):
    results_dir, audit_path, output_dir = (Path(results_dir).resolve(), Path(audit_path).resolve(),
                                           Path(output_dir).absolute())
    require(not output_dir.exists() and not output_dir.is_symlink(), "Refusing an existing output directory")
    require(output_dir.parent.is_dir() and output_dir.parent.resolve() == output_dir.parent,
            "Output parent must already exist and be canonical")
    hashes = snapshot_inputs(results_dir, audit_path)
    report, manifest, original_paths, rows = validate_gate(results_dir, audit_path, synthetic,
                                                            verified_mirror, hashes)
    validate_identities(results_dir, rows)
    aggregated = validate_and_aggregate(results_dir, rows, synthetic)
    overview = overview_rows(aggregated)
    temporary = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".", dir=output_dir.parent))
    try:
        write_csv(temporary / "mureka60_transfer_all_635.csv", aggregated)
        write_csv(temporary / "mureka60_transfer_overview.csv", overview)
        (temporary / "MUREKA60_TRANSFER_TABLES_EN.md").write_text(render_markdown(aggregated, overview, synthetic))
        (temporary / "mureka60_transfer_all_635_en.tex").write_text(render_latex(aggregated, synthetic))
        output_hashes = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                         for path in sorted(temporary.iterdir())}
        receipt = {
            "schema_version": 1,
            "status": "synthetic_test_fixture_complete" if synthetic else "complete",
            "synthetic_test_only": synthetic,
            "scope": "Descriptive aggregation of accepted frozen-v4 Mureka sensitivity and false-negative summaries. Five dependent stored fold models evaluate the same AI songs per cell. Minima and maxima are not confidence intervals.",
            "rows_per_model": rows, "family_subsets": 127, "caps": list(CAPS),
            "stored_fold_models_per_cell": 5, "exported_cells": 635,
            "independent_audit": {"path": str(audit_path), "sha256": hashes[str(audit_path)],
                                  "publication_manifest_sha256": report["publication_manifest_sha256"],
                                  "auditor_sha256": report["auditor_sha256"]},
            "verified_mirror": verified_mirror,
            "locally_present_scored_files": sorted(path.name for path in results_dir.iterdir()),
            "locally_omitted_audit_bound_scored_files": sorted(SCORED_FILES - {path.name for path in results_dir.iterdir()}),
            "audited_original_scored_paths_by_basename": original_paths,
            "inputs_sha256": dict(sorted(hashes.items())),
            "exporter_sha256": hashes[str(Path(__file__).resolve())],
            "outputs": output_hashes,
        }
        (temporary / "export_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
        require({name: sha(Path(name)) for name in hashes} == hashes,
                "An audited input changed during export")
        publish_directory(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verified-mirror", action="store_true",
                        help="allow audited original paths to match this directory by unique basename and exact hash")
    parser.add_argument("--synthetic-test-only", action="store_true",
                        help="admit only an explicitly synthetic passed audit with at most 16 rows")
    args = parser.parse_args(argv)
    require(all(path.is_absolute() for path in (args.results_dir, args.audit, args.output_dir)),
            "All paths must be absolute")
    receipt = export(args.results_dir, args.audit, args.output_dir,
                     synthetic=args.synthetic_test_only, verified_mirror=args.verified_mirror)
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
