#!/usr/bin/env python3
"""Benchmark the full evaluator on a realistic, label-synthetic 8,361-row cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_expanded as evaluator


SEED = 20_260_905
HUMAN_SOURCES = {
    "fma_medium": 400,
    "aime_mtg_jamendo": 500,
    "human_maestro_v3": 300,
    "human_magnatagatune": 500,
    "human_medleydb": 178,
    "human_moisesdb": 239,
    "human_urmp": 44,
}
AI_SOURCES = {
    "acestep": 400,
    "heartmula": 400,
    "suno": 400,
    **{f"aime_generator_{index:02d}": 500 for index in range(10)},
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def group_id(source: str, index: int) -> str:
    if source in {"acestep", "heartmula"}:
        return f"muse:source_song_{index:04d}"
    if source == "aime_mtg_jamendo" or source.startswith("aime_generator_"):
        return f"aime:condition_{index:04d}"
    return f"{source}:creator_or_work_{index:04d}"


def generator_family(source: str, label: int) -> str:
    if label == 0:
        return ""
    if source in {"acestep", "heartmula", "suno"}:
        return source
    index = int(source.rsplit("_", 1)[1])
    if index < 3:
        return "musicgen"
    if index < 5:
        return "audioldm2"
    if index < 7:
        return "stable_audio"
    return f"aime_architecture_{index:02d}"


def build_inputs(root: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    rng = np.random.default_rng(SEED)
    metadata_rows: list[dict[str, object]] = []
    labels: list[int] = []
    source_offsets: list[float] = []
    sources = [(name, count, 0) for name, count in HUMAN_SOURCES.items()]
    sources += [(name, count, 1) for name, count in AI_SOURCES.items()]
    offset_map = {name: float(rng.normal(0, 0.08)) for name, _, _ in sources}
    for source, count, label in sources:
        acquisition = "expansion" if source.startswith("human_") else "prior"
        for index in range(count):
            item_id = f"synthetic_{source}_{index:05d}"
            metadata_rows.append({
                "id": item_id,
                "track": item_id,
                "label": label,
                "source_id": source,
                "source_group": source,
                "role": "development",
                "original_role": "development" if not source.startswith("aime_") else "consumed_aime_test",
                "group_id": group_id(source, index),
                "generator_family": generator_family(source, label),
                "acquisition": acquisition,
                "available": 1,
                "native_sample_rate_hz": 32_000 if index % 5 == 0 else (44_100 if index % 2 else 48_000),
                "native_duration_s": 30.0,
                "duration_sec": 30,
                "duration_view": "30s",
                "eligible_common8": 1,
                "eligible_fullband": 0 if index % 5 == 0 else 1,
            })
            labels.append(label)
            source_offsets.append(offset_map[source])
    metadata = pd.DataFrame(metadata_rows)
    n_rows = len(metadata)
    if n_rows != 8_361 or len(HUMAN_SOURCES) != 7 or len(AI_SOURCES) != 13:
        raise AssertionError("Synthetic benchmark inventory contract changed")
    label_signal = np.asarray(labels, dtype=float) * 0.35 + np.asarray(source_offsets)
    feature_data: dict[str, object] = {
        "item_id": metadata.id.to_numpy(),
        "label": labels,
        "source_id": metadata.source_id.to_numpy(),
        "group_id": metadata.group_id.to_numpy(),
        "native_sample_rate_hz": metadata.native_sample_rate_hz.to_numpy(),
        "duration_sec": np.full(n_rows, 30.0),
        "feature_status": np.full(n_rows, "complete", dtype=object),
    }
    family_columns: dict[str, list[str]] = {"S8": [], "S16": [], "SREF": [], "D": [], "R": [], "P": []}
    for family, count, missing_rate in (
        ("S8", 15, 0.01), ("S16", 15, 0.01), ("SREF", 15, 0.01),
        ("D", 3, 0.05), ("R", 3, 0.10), ("P", 6, 0.40),
    ):
        prefix = family.lower()
        for index in range(count):
            column = f"{prefix}__feature_{index:02d}"
            values = label_signal + rng.normal(0, 1.0 + index * 0.01, n_rows)
            values[rng.random(n_rows) < missing_rate] = np.nan
            feature_data[column] = values
            family_columns[family].append(column)
    features = pd.DataFrame(feature_data)
    metadata_path, features_path = root / "metadata.csv", root / "features.csv"
    metadata.to_csv(metadata_path, index=False)
    features.to_csv(features_path, index=False)
    config = {
        "primary_feature_set": "common8",
        "feature_sets": {
            "common8": {
                "selection_eligible": True, "cohort": "synthetic benchmark common band",
                "eligibility": {}, "status_columns": ["feature_status"],
                "families": {"S": family_columns["S8"], "D": family_columns["D"],
                             "R": family_columns["R"], "P": family_columns["P"]},
            },
            "native16": {
                "selection_eligible": False, "cohort": "synthetic benchmark full band",
                "eligibility": {"min": {"native_sample_rate_hz": 40000}},
                "status_columns": ["feature_status"],
                "families": {"S": family_columns["S16"], "D": family_columns["D"],
                             "R": family_columns["R"], "P": family_columns["P"]},
            },
            "original16": {
                "selection_eligible": False, "cohort": "synthetic benchmark uncontrolled reference",
                "eligibility": {}, "status_columns": ["feature_status"],
                "families": {"S": family_columns["SREF"], "D": family_columns["D"],
                             "R": family_columns["R"], "P": family_columns["P"]},
            },
        },
    }
    config_path = root / "families.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    scopes_path = root / "scopes.json"
    scopes_path.write_text(json.dumps({
        "prior_only": {"equals": {"acquisition": "prior"}},
        "all_development": {},
    }), encoding="utf-8")
    inventory = {
        "development_rows": n_rows,
        "human_rows": int((metadata.label == 0).sum()),
        "ai_rows": int((metadata.label == 1).sum()),
        "human_sources": HUMAN_SOURCES,
        "ai_sources": AI_SOURCES,
        "unique_groups": int(metadata.group_id.nunique()),
        "aime_shared_group_rows": int(metadata.group_id.str.startswith("aime:").sum()),
        "muse_shared_group_rows": int(metadata.group_id.str.startswith("muse:").sum()),
        "active_columns_per_representation": {"S": 15, "D": 3, "R": 3, "P": 6},
        "spectral_representations": 3,
        "training_scopes": 2,
        "quantities": [25, 50, 100, 200, "all"],
        "combinations": 15,
        "folds": {"human_source": 35, "ai_source": 65, "ordinary_group": 5, "total": 105},
        "nominal_cv_fits": 3 * 2 * 5 * 15 * 105,
        "post_selection_diagnostic_fits": {
            "missingness_leader_and_s_max": 2 * 3 * 105,
            "generator_family_leader_and_s": 2 * 9 * 5,
            "frozen_all_candidate_models": 3 * 15,
        },
    }
    return metadata_path, features_path, config_path, scopes_path, inventory


def equivalence(metadata: Path, features: Path, config_path: Path) -> dict[str, float | int]:
    args = argparse.Namespace(
        metadata=metadata, features=[features], id_column=None, label_column=None,
        role_column=None, source_column=None, group_column=None,
    )
    table, _ = evaluator.load_table(args)
    config = evaluator.load_family_config(config_path)
    spec = config["feature_sets"]["common8"]
    table, _ = evaluator.apply_eligibility(table, spec)
    # A realistic fold-sized subset with every source and natural missingness.
    sample = table.groupby(["__label", "__source"], group_keys=False).head(220).copy()
    union = sum((spec["families"][name] for name in evaluator.FAMILY_ORDER), [])
    direct_started = time.perf_counter()
    direct_models = {
        name: evaluator.fit_model(sample, columns)
        for name, columns in evaluator.combinations(spec["families"]).items()
    }
    direct_seconds = time.perf_counter() - direct_started
    cached_started = time.perf_counter()
    prepared = evaluator.prepare_training(sample, union)
    cached_models = {
        name: evaluator.fit_prepared(prepared, columns)
        for name, columns in evaluator.combinations(spec["families"]).items()
    }
    cached_seconds = time.perf_counter() - cached_started
    maximum_coefficient_difference = 0.0
    maximum_prediction_difference = 0.0
    comparisons = 0
    for name, columns in evaluator.combinations(spec["families"]).items():
        direct = direct_models[name]
        cached = cached_models[name]
        maximum_coefficient_difference = max(
            maximum_coefficient_difference,
            float(np.max(np.abs(np.asarray(direct["coefficients_with_intercept"])
                                - np.asarray(cached["coefficients_with_intercept"])))),
        )
        maximum_prediction_difference = max(
            maximum_prediction_difference,
            float(np.max(np.abs(evaluator.predict(sample, direct) - evaluator.predict(sample, cached)))),
        )
        comparisons += 1
    return {
        "combination_models_compared": comparisons,
        "rows": len(sample),
        "max_abs_coefficient_difference": maximum_coefficient_difference,
        "max_abs_prediction_difference": maximum_prediction_difference,
        "equivalent_at_rtol_1e-11_atol_1e-12": bool(
            maximum_coefficient_difference <= 1e-11 and maximum_prediction_difference <= 1e-11
        ),
        "direct_15_fit_seconds": direct_seconds,
        "cached_prepare_plus_15_fit_seconds": cached_seconds,
        "microbenchmark_speedup": direct_seconds / cached_seconds if cached_seconds else math.inf,
    }


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(1 << 20), b""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    code_path = Path(evaluator.__file__).resolve()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    audit: dict[str, object] = {
        "status": "running",
        "purpose": "runtime benchmark only; synthetic values cannot select a scientific candidate",
        "seed": SEED,
        "evaluator_path": str(code_path),
        "evaluator_sha256": sha256(code_path),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="evaluate_expanded_benchmark_") as temporary:
            root = Path(temporary)
            metadata, features, config, scopes, inventory = build_inputs(root)
            audit["synthetic_inventory"] = inventory
            audit["numerical_equivalence"] = equivalence(metadata, features, config)
            output_dir = root / "evaluation"
            command = [
                sys.executable, str(code_path), "--stage", "cv",
                "--metadata", str(metadata), "--features", str(features),
                "--families-json", str(config), "--training-scopes-json", str(scopes),
                "--output-dir", str(output_dir), "--quantities", "25,50,100,200,all",
                "--opposite-class-folds", "5", "--bootstrap-replicates", "0",
            ]
            audit["command_template"] = [value.replace(str(root), "<temporary>") for value in command]
            subprocess_started = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            subprocess_seconds = time.perf_counter() - subprocess_started
            audit["subprocess"] = {
                "returncode": completed.returncode,
                "wall_seconds": subprocess_seconds,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            output_evidence = {}
            for path in sorted(output_dir.glob("evaluate_expanded*")):
                output_evidence[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "lines": line_count(path) if path.suffix in {".csv", ".json"} else None,
                }
            audit["output_evidence"] = output_evidence
            audit["estimated_full_run"] = {
                "measured_cv_wall_seconds": subprocess_seconds,
                "measured_cv_wall_minutes": subprocess_seconds / 60.0,
                "nominal_cv_fits": inventory["nominal_cv_fits"],
                "note": "Direct full-scale synthetic execution of all requested representations/scopes/quantities/combinations/folds",
            }
            if completed.returncode != 0:
                raise RuntimeError(f"Benchmark evaluator failed with return code {completed.returncode}")
            test_path = code_path.with_name("evaluate_expanded_test.py")
            test_started = time.perf_counter()
            tested = subprocess.run(
                [sys.executable, str(test_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            test_seconds = time.perf_counter() - test_started
            test_output = tested.stdout + tested.stderr
            test_count_match = re.search(r"Ran (\d+) tests?", test_output)
            audit["unit_test_evidence"] = {
                "command": [sys.executable, str(test_path)],
                "returncode": tested.returncode,
                "tests_run": int(test_count_match.group(1)) if test_count_match else None,
                "result": "OK" if tested.returncode == 0 else "FAILED",
                "wall_seconds": test_seconds,
                "test_file_sha256": sha256(test_path),
                "output_tail": test_output[-4000:],
            }
            if tested.returncode != 0:
                raise RuntimeError(f"Evaluator unit tests failed with return code {tested.returncode}")
            audit["status"] = "passed"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["total_wall_seconds"] = time.perf_counter() - started_wall
        audit["benchmark_driver_cpu_seconds"] = time.process_time() - started_cpu
        audit["child_peak_rss_platform_units"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        audit["completed_utc"] = pd.Timestamp.utcnow().isoformat()
        args.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
