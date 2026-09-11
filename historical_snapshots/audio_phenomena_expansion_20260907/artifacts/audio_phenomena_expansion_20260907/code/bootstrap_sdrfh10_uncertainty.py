#!/usr/bin/env python3
"""Frozen-model 10s uncertainty, with independently frozen post-point protocol.

Reuse hash-pinned 30s numerical routines in a PRIVATE module instance. Never
call their 30s input loader/main or change their file/global imported module.
This adapter independently validates the audited 10s contract and cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

CONTRACT_SHA = "55ac2df4b3de27276f6af362ee6c98512469fb08b3c170396202314cb6c0435f"
AUDIT_SHA = "677391a4a6a00176329a2a270d7374b6a6f58613b1904658993b004f49063cc8"
RECEIPT_SHA = "85843c1b1c7497bf144a02fdce46418fa1b236debc133144d451aed0cf2c0e56"
CORE_SHA = "9707a7b443e29a4c9c86811bb9ff7f8a5594d895163984d9df3e870558e71371"
COHORT_SHA = "d418be0b6d721b2ed980a99b9185f79b4b622e60fe293e0f3fd43f473c14eb41"
BASELINE = "S+D"
ADDED = ("S+D+F", "S+D+H", "S+D+F+H")
COMBINATIONS = (BASELINE,) + ADDED
SEED = 20260907
GROUP_DRAWS, SOURCE_DRAWS = 1000, 10000
CORE_PATH = Path(__file__).with_name("bootstrap_matched_deltas.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


if sha(CORE_PATH) != CORE_SHA:
    raise ValueError("Preserved numerical core differs from pinned 30s code")
spec = importlib.util.spec_from_file_location("_private_sdrfh10_numerics", CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.BASELINE, core.ADDED, core.COMBINATIONS = BASELINE, ADDED, COMBINATIONS


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def resolve_record(root, path):
    """Relocate only the declared artifact-root suffix, never a basename guess."""
    parts = Path(path).parts
    marker = "audio_phenomena_expansion_20260907"
    if marker not in parts:
        raise ValueError(f"Path is outside the exact artifact contract: {path}")
    candidate = root.joinpath(*parts[parts.index(marker) + 1:]).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"Declared input is unavailable: {candidate}")
    return candidate


def check_contract(manifest):
    if manifest.get("stage") != "dev" or manifest.get("schema_version") != 2:
        raise ValueError("Requires completed schema-v2 development evaluation")
    auth = manifest.get("authorization")
    if not isinstance(auth, dict) or auth.get("status") != "frozen_verified":
        raise ValueError("Requires nested frozen_verified authorization")
    if (core._evaluator_contract_hash(manifest.get("contract", {})) != CONTRACT_SHA
            or auth.get("contract_sha256") != CONTRACT_SHA):
        raise ValueError("Wrong exact10 evaluation contract")
    if auth.get("receipt_sha256") != RECEIPT_SHA:
        raise ValueError("Wrong exact10 frozen receipt")
    if (manifest.get("plan_rows"), manifest.get("unique_candidates"), manifest.get("cohorts")) != (52, 31, 1):
        raise ValueError("Incomplete 31-candidate/52-plan-row evaluation")


def input_paths(root):
    evaluation = root / "results/evaluation_sdrfh_10s_v2"
    manifest_path = evaluation / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Incomplete evaluation: no completion manifest")
    manifest = read_json(manifest_path)
    check_contract(manifest)
    auth, contract = manifest["authorization"], manifest["contract"]
    receipt = resolve_record(root, auth["receipt_path"])
    if sha(receipt) != RECEIPT_SHA:
        raise ValueError("Frozen receipt bytes mismatch")
    declared = read_json(receipt)
    if (declared.get("status") != "frozen" or declared.get("contract") != contract
            or declared.get("contract_sha256") != CONTRACT_SHA):
        raise ValueError("Unfrozen or inconsistent receipt")
    audit = root / "audit/sdrfh_10s_v2_audit_report.json"
    if sha(audit) != AUDIT_SHA:
        raise ValueError("Independent completed audit bytes mismatch")
    report = read_json(audit)
    if (report.get("status"), report.get("checks_passed"), report.get("checks_total"), report.get("errors")) != ("passed", 21380, 21380, []):
        raise ValueError("Independent audit has not passed")
    paths = [manifest_path, receipt, audit, root / "UNCERTAINTY_SDRFH_10S_PROTOCOL_EN.md"]
    for name in (
        "development_source_holdout_predictions.csv",
        "development_source_holdout_metrics_by_source_pair.csv",
        "development_source_holdout_summary.csv",
        "development_source_holdout_matched_deltas.csv", "evaluation_plan.csv",
        "cohort_registry.csv", "frozen_dev_models.json", "process.json", "skipped_folds.csv",
    ):
        path = evaluation / name
        if not path.is_file():
            raise ValueError(f"Incomplete evaluation: missing {name}")
        paths.append(path)
    if read_json(evaluation / "process.json").get("state") != "finished":
        raise ValueError("Evaluation process has not finished")
    if (evaluation / "skipped_folds.csv").read_text().strip():
        raise ValueError("Unexpected skipped cells")
    if sha(evaluation / "frozen_dev_models.json") != manifest["frozen_dev_bundle_sha256"]:
        raise ValueError("Frozen model bytes mismatch")
    for record in [contract["metadata"], contract["families_json"], *contract["features"]]:
        path = resolve_record(root, record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError(f"Contract input hash mismatch: {path}")
        paths.append(path)
    return paths


METHOD = {
    "status": "frozen_before_resampling_after_point_estimates",
    "baseline": BASELINE, "added": list(ADDED), "quantity": "all",
    "baseline_rationale": "September 5 pre-new-feature exact10 baseline S+D; no new model selection",
    "group_replicates": GROUP_DRAWS, "source_replicates": SOURCE_DRAWS, "seed": SEED,
    "threshold": 0.5, "group_unit": "global group_id across all sources/classes/folds/models",
    "pairing": "identical (fold_uid,row_id) test identities and metadata across all four models",
    "macro": "mean pair/fold within held source; equal held-source macro per direction; equal two-direction mean",
    "interval": "2.5/97.5 linear percentile; finite draws only with undefined counts reported",
    "undefined": "zero class weight -> NaN cell; explicit audit; skip NaN within source then source macro",
    "crossed_source": "independent multinomial 7 Human and 13 AI draws; shared vectors in both directions and all three contrasts; five folds averaged per pair",
    "source_interpretation": "observed-source composition sensitivity; separate from group CI; no future-source coverage",
    "limits": "post-point-estimate uncertainty protocol; conditional on fitted models/folds/observed corpus; no refit, held-label score flipping, threshold changes, multiplicity adjustment, historical scoring, general discovery or population/future-generator guarantee",
}


def freeze(root, protocol):
    if protocol.exists():
        raise ValueError("Refusing to replace frozen uncertainty protocol")
    paths = input_paths(root)
    payload = dict(METHOD, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                   contract_sha256=CONTRACT_SHA,
                   code_sha256={p.name: sha(p) for p in (Path(__file__), CORE_PATH,
                       Path(__file__).with_name("test_bootstrap_sdrfh10_uncertainty.py"),
                       Path(__file__).with_name("test_bootstrap_matched_deltas.py"))},
                   inputs_sha256={str(p.relative_to(root)): sha(p) for p in paths})
    protocol.parent.mkdir(parents=True, exist_ok=True)
    with protocol.open("x") as f:
        f.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "protocol_frozen", "protocol_sha256": sha(protocol)}), flush=True)


def validate_cohort(root, observations):
    metadata = read_csv(root / "evaluation_inputs/sdrfh_10s_v2/metadata_10s.csv")
    eligible = {r["id"]: r for r in metadata if r["role"] == "development"
                and r["duration_view"] == "10s" and r["eligible_common8"] == "1"}
    unique = {}
    for row in observations["master"]:
        identity = row["row_id"]
        if identity not in eligible:
            raise ValueError("Prediction ID is not in the exact development cohort")
        expected = eligible[identity]
        if any(str(row[k]) != expected[k] for k in ("label", "source_group", "group_id")):
            raise ValueError("Prediction metadata inconsistent with frozen metadata")
        unique[identity] = row
    if set(unique) != set(eligible):
        raise ValueError("Prediction union is not the full exact10 development cohort")
    digest = hashlib.sha256("".join(f"{x}\n" for x in sorted(unique)).encode()).hexdigest()
    if digest != COHORT_SHA:
        raise ValueError("Wrong cohort ID-set digest")
    counts = {}
    for label, name in ((0, "human"), (1, "ai")):
        rows = [r for r in unique.values() if r["label"] == label]
        counts[name] = {"rows": len(rows), "groups": len({r["group_id"] for r in rows}),
                        "source_counts": dict(sorted(Counter(r["source_group"] for r in rows).items()))}
    if (counts["human"]["rows"], counts["ai"]["rows"], counts["human"]["groups"], counts["ai"]["groups"],
        len(counts["human"]["source_counts"]), len(counts["ai"]["source_counts"])) != (2161,6200,1539,1300,7,13):
        raise ValueError("Exact10 source/class/group counts mismatch")
    counts["global_group_union"] = len(observations["groups"])
    return counts


def load_inputs(root):
    evaluation = root / "results/evaluation_sdrfh_10s_v2"
    return {"predictions": core._read_filtered_csv(evaluation / "development_source_holdout_predictions.csv", COMBINATIONS),
            "metrics": core._read_filtered_csv(evaluation / "development_source_holdout_metrics_by_source_pair.csv", COMBINATIONS),
            "summary": core._read_filtered_csv(evaluation / "development_source_holdout_summary.csv", COMBINATIONS),
            "deltas": core._read_target_deltas(evaluation / "development_source_holdout_matched_deltas.csv"),
            "plan_path": evaluation / "evaluation_plan.csv"}


def validate_grid(inputs, cells):
    if len(cells) != 2 * 7 * 13 * 5:
        raise ValueError("Incomplete 910-cell source-pair grid")
    expected = {cell["key"] for cell in cells}
    for combo in COMBINATIONS:
        actual = [core._cell_key(r) for r in inputs["metrics"] if r["combination"] == combo]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Unpaired or duplicate metric grid")
    per_pair = defaultdict(set)
    for cell in cells:
        r = cell["row"]
        per_pair[(r["fold_type"], r["human_source"], r["ai_source"])].add(r["opposite_group_fold"])
    if len(per_pair) != 182 or any(folds != {str(i) for i in range(5)} for folds in per_pair.values()):
        raise ValueError("Incomplete five-fold crossed source grid")


def source_reweight(matrix, humans, ais):
    if matrix.ndim != 5 or matrix.shape[:3] != (2, len(humans), len(ais)):
        raise ValueError("Source matrix dimensions mismatch")
    if not np.isfinite(matrix).all() or np.any(humans < 0) or np.any(ais < 0):
        raise ValueError("Invalid source values/weights")
    weight = np.outer(humans, ais)
    if weight.sum() <= 0:
        raise ValueError("Empty source draw")
    return np.einsum("dhacm,ha->cm", matrix, weight) / (2 * weight.sum())


def source_outputs(inputs, point_deltas, output):
    paired = defaultdict(list)
    for row in inputs["deltas"]:
        if row["baseline_combination"] != BASELINE:
            raise ValueError("Wrong matched baseline")
        key = (row["fold_type"], row["human_source"], row["ai_source"], row["added_combination"])
        paired[key].append([float(row["delta_" + m + "__added_minus_baseline"])
                            for m in ("roc_auc", "balanced_accuracy")])
    humans = sorted({key[1] for key in paired})
    ais = sorted({key[2] for key in paired})
    if (len(humans), len(ais)) != (7,13):
        raise ValueError("Wrong source universe")
    matrix = np.empty((2,7,13,3,2))
    for d, direction in enumerate(core.FOLD_TYPES):
        for h, human in enumerate(humans):
            for a, ai in enumerate(ais):
                for c, added in enumerate(ADDED):
                    values = paired[(direction,human,ai,added)]
                    if len(values) != 5:
                        raise ValueError("Incomplete source pair")
                    matrix[d,h,a,c] = np.mean(values, axis=0)
    point = source_reweight(matrix, np.ones(7), np.ones(13))
    expected = np.array([[r["delta_J"], r["delta_equal_mean_balanced_accuracy"]] for r in point_deltas])
    error = float(np.abs(point - expected).max())
    if error > 1e-10:
        raise ValueError("Crossed-source macro does not reconstruct group point estimate")
    rng = np.random.default_rng(SEED)
    hd = rng.multinomial(7, np.ones(7)/7, size=SOURCE_DRAWS)
    ad = rng.multinomial(13, np.ones(13)/13, size=SOURCE_DRAWS)
    estimates = np.stack([source_reweight(matrix,h,a) for h,a in zip(hd,ad)])
    rows, summary = [], []
    for c, added in enumerate(ADDED):
        for m, metric in enumerate(("delta_J", "delta_equal_mean_balanced_accuracy")):
            low, high = np.percentile(estimates[:,c,m], [2.5,97.5])
            summary.append({"baseline_combination":BASELINE,"added_combination":added,"metric":metric,
                            "point_estimate":point[c,m],"empirical_source_95_low":low,
                            "empirical_source_95_high":high,"positive_fraction":float(np.mean(estimates[:,c,m]>0)),
                            "defined_replicates":SOURCE_DRAWS,"undefined_replicates":0})
        rows.extend({"replicate":i,"added_combination":added,"delta_J":v[0],
                     "delta_equal_mean_balanced_accuracy":v[1]} for i,v in enumerate(estimates[:,c]))
    output.mkdir()
    core.write_csv(output / "sensitivity_summary.csv", summary)
    core.write_csv(output / "replicates.csv", rows)
    np.savez_compressed(output / "source_multiplicities.npz", human=hd, ai=ad, matrix=matrix)
    write_json(output / "source_manifest.json", {"analysis":"observed_source_composition_sensitivity_only",
               "human_sources":humans,"ai_sources":ais,"replicates":SOURCE_DRAWS,"seed":SEED,
               "max_point_reproduction_error":error,"no_group_ci_combination":True,
               "undefined_point_cells":0,"undefined_replicates":0})
    return summary


def run(root, protocol, output):
    if output.exists():
        raise ValueError("Refusing existing output directory")
    frozen = read_json(protocol)
    if any(frozen.get(k) != v for k,v in METHOD.items()):
        raise ValueError("Uncertainty protocol is not exactly frozen")
    for name, digest in frozen["code_sha256"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("Code changed after protocol freeze")
    paths = input_paths(root)
    actual = {str(p.relative_to(root)):sha(p) for p in paths}
    if actual != frozen["inputs_sha256"]:
        raise ValueError("Inputs changed after protocol freeze")
    print("Frozen protocol, code, inputs and complete evaluation verified", flush=True)
    inputs = load_inputs(root)
    observations = core.build_observations(inputs["predictions"])
    counts = validate_cohort(root, observations)
    cells = core.build_cells(inputs["metrics"], observations)
    validate_grid(inputs, cells)
    points, deltas = core.validate_point_estimates(inputs, observations, cells)
    print(json.dumps({"point_reconstruction_passed":True,"cohort":counts,"point_deltas":deltas}), flush=True)
    # No resampling occurs before all point/cell/denominator reconstruction passes.
    replicates, summary, undefined, weights = core.bootstrap_outputs(
        observations, cells, GROUP_DRAWS, SEED, deltas, inputs["plan_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name+".tmp.", dir=output.parent))
    core.write_csv(temporary / "point_estimates.csv", points)
    core.write_csv(temporary / "point_matched_deltas.csv", deltas)
    core.write_csv(temporary / "bootstrap_replicates.csv", replicates)
    core.write_csv(temporary / "uncertainty_summary.csv", summary)
    core.write_csv(temporary / "undefined_pair_metrics.csv", undefined,
                   ["replicate", *core.CELL_KEY_FIELDS, "human_weight", "ai_weight", "reason", "applies_identically_to_combinations"])
    np.savez_compressed(temporary / "global_group_multiplicities.npz", multiplicities=weights,
                        group_id=np.array(observations["groups"]))
    source_outputs(inputs, deltas, temporary / "source_sensitivity")
    write_json(temporary / "run_manifest.json", {
        "status":"complete","analysis":"post_point_estimate_exact10_frozen_model_uncertainty",
        "protocol_sha256":sha(protocol),"code_sha256":frozen["code_sha256"],"inputs_sha256":actual,
        "numpy_version":np.__version__,"cohort":counts,"global_group_count":len(observations["groups"]),
        "group_universe_sha256":core.canonical_hash(observations["groups"]),
        "paired_prediction_observations_per_combination":len(observations["master"]),
        "source_pair_fold_cells":len(cells),"group_replicates":GROUP_DRAWS,"source_replicates":SOURCE_DRAWS,"seed":SEED,
        "undefined_pair_metric_rows":len(undefined),
        "replicates_with_any_undefined_pair":len({r["replicate"] for r in undefined}),
        "point_reproduction_max_abs_error":max(max(r["max_abs_cell_error_vs_evaluator"],r["max_abs_summary_error_vs_evaluator"]) for r in points),
        "matched_delta_reproduction_max_abs_error":max(r["max_abs_cell_delta_error_vs_evaluator"] for r in deltas),
        "outputs_sha256":{str(p.relative_to(temporary)):sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()},
        "interpretation":METHOD["limits"],"source_interpretation":METHOD["source_interpretation"]})
    os.replace(temporary, output)
    print(json.dumps({"status":"complete","output_dir":str(output),"undefined_cells":len(undefined)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--freeze-protocol",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    if args.freeze_protocol:
        freeze(args.root.resolve(),args.protocol.resolve())
    elif args.output_dir:
        run(args.root.resolve(),args.protocol.resolve(),args.output_dir.resolve())
    else:
        parser.error("Provide --freeze-protocol or --output-dir")
