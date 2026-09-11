#!/usr/bin/env python3
"""Independent, streaming numerical auditor for completed exploratory-v5 runs.

This module deliberately does not import the evaluator, its preserved scorer,
or an optimizer.  It reconstructs the frozen fold/cap schedule, validates every
saved training transform, checks the ridge normal equations, replays every raw
identity-link score column by column, and independently recomputes all emitted
metrics.  A receipt is published only after a complete, immutable publication
passes; partial or orphan result directories are rejected before scoring data
are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


STAGE = "exploratory_v5_development_cv_only"
SEED = 20_260_907
FOLDS = 5
CAPS: tuple[int | str, ...] = (25, 50, 100, 200, "all")
FAMILY_ORDER = ("S", "D", "R", "P", "F", "H", "M")
MODES = ("values_plus_missing", "median_only", "missingness_only")
POLICIES = {
    "values_plus_missing": "train-only median plus feature-missing indicators",
    "median_only": "train-only median; no indicators",
    "missingness_only": "feature-missing indicators only",
}
FIXED_CORE = {
    "evaluate_new_phenomena_v2.py": "4014492f3e3ddda2d9cbea9f37147b37be0124ba071f0fee803845537526bd2e",
    "frozen_evaluate_expanded_20260905.py": "d2ed30d9833fbe122f023de1223e44a40c1b4f95f99f63f0ba27647c87cc7232",
}
FAMILIES = {
    "S": ["s8__" + name for name in (
        "tilt_1_5k_db_oct", "hf_tilt_5_7p5k_db_oct", "hf_ratio_5_7p5_db",
        "sibilance_ratio_5_7p5_db", "hf_flatness_5_7p5", "hf_entropy_5_7p5",
        "hf_crest_5_7p5_db", "fakeprint_peak_density_5_7p5_per_khz",
        "fakeprint_periodicity_5_7p5", "hf_flux_5_7p5", "hf_frame_similarity_5_7p5",
        "hf_power_sd_5_7p5_db", "hf_mod_4_12_share_5_7p5",
        "sibilance_contrast_5_7p5_db", "sibilance_burst_rate_5_7p5_hz")],
    "D": ["d__" + name for name in ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")],
    "R": ["r__" + name for name in ("ibi_cv", "tempo_tv", "tempo_entropy")],
    "P": ["p__" + name for name in (
        "section_duration_cv", "section_duration_entropy", "section_bars_cv",
        "section_bars_offmode_fraction", "section_duration_median", "section_bars_median")],
    "F": ["F_" + name for name in (
        "phase_residual_cvar_all", "phase_residual_cvar_attack", "phase_residual_cvar_sustain",
        "phase_residual_cvar_decay", "group_delay_iqr_ms_all", "group_delay_iqr_ms_attack",
        "group_delay_iqr_ms_sustain", "group_delay_iqr_ms_decay",
        "group_delay_cross_band_iqr_ms_all", "group_delay_cross_band_iqr_ms_attack",
        "group_delay_cross_band_iqr_ms_sustain", "group_delay_cross_band_iqr_ms_decay",
        "phase_residual_cvar_decay_minus_sustain", "group_delay_iqr_ms_decay_minus_sustain",
        "group_delay_cross_band_iqr_ms_decay_minus_sustain")],
    "H": ["H_" + name for name in (
        "pitch_class_entropy_norm", "pc_token_entropy_norm", "chroma_path_change_median",
        "chroma_path_change_iqr", "pc_path_step_median", "pc_path_large_step_rate")],
    "M": ["M_" + name for name in (
        "recurrence_peak_similarity", "recurrence_density", "recurrence_lag_contrast",
        "best_lag_sec", "best_transposition_semitones", "returning_pattern_count")],
}
DESCRIPTORS = sum((FAMILIES[name] for name in FAMILY_ORDER), [])

META = ["id", "label", "source_group", "group_id", "role", "duration_view", "native_sample_rate_hz"]
LEDGER = ["id", "label", "source_group", "group_id", "original_role", "original_flags_json",
          "original_metadata_json", "origin_package", "origin_kind", "origin_row_sha256",
          "consumed_external", "exploratory_reuse_status"]
INDEX = ["model_uid", "model_sha256", "model_jsonl_line", "combination", "feature_mode", "quantity",
         "fold_uid", "fold_index", "fold_type", "heldout_source", "opposite_group_fold",
         "train_id_set_sha256", "test_id_set_sha256", "training_rows", "test_rows"]
PRED = ["model_uid", "row_id", "label", "source_group", "group_id", "role", "score", "threshold",
        "predicted_label"]
METRICS = ["roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity", "tp", "tn", "fp", "fn"]
PAIR = ["model_uid", "human_source", "ai_source", "test_human", "test_ai", "test_human_groups",
        "test_ai_groups", "threshold", *METRICS]
POOLED = ["model_uid", "test_rows", "test_groups", "threshold", *METRICS]
SOURCE = ["model_uid", "source_group", "label", "rows", "components", "threshold", "tp", "tn", "fp", "fn",
          "recording_positive_rate", "recording_negative_rate", "recording_ai_sensitivity",
          "recording_human_specificity", "equal_component_positive_rate", "equal_component_negative_rate",
          "equal_component_ai_sensitivity", "equal_component_human_specificity"]

EXPECTED_REAL_SOURCES = {
    "Suno": 396, "Mureka_v9": 500, "MTG-Jamendo": 481, "human_maestro_v3": 300,
    "human_medleydb": 156, "human_moisesdb": 238, "human_urmp": 33,
    "human_saraga_hindustani_v1": 103,
}
EXPECTED_REAL = {"valid_folds": 43, "primary_fits": 27_305, "primary_prediction_rows": 7_633_970,
                 "diagnostic_fits": 10_922, "diagnostic_prediction_rows": 3_053_588}
EXPECTED_RESULT_FILES = {
    "fold_registry.jsonl", "omitted_fold_cells.json", "frozen_authorization.json", "fold_models.jsonl",
    "primary_model_index.csv", "primary_predictions.csv", "primary_pair_metrics.csv",
    "primary_pooled_metrics.csv", "primary_per_source_endpoints.csv", "diagnostic_model_index.csv",
    "diagnostic_predictions.csv", "diagnostic_pair_metrics.csv", "diagnostic_pooled_metrics.csv",
    "diagnostic_per_source_endpoints.csv", "run_manifest.json",
}
EXPECTED_BINDING_BASENAMES = {
    "metadata_60s.csv", "features_60s.csv", "family_config.json", "origin_ledger.csv",
    "preparation_audit.json", "COMMIT.json", "evaluate_new_phenomena_v5.py",
    "prepare_evaluation_inputs_v5.py", "prepare_evaluation_inputs_v4.py",
    "present_saraga_mureka_frozen_transfer_v1.py", "evaluate_new_phenomena_v2.py",
    "frozen_evaluate_expanded_20260905.py", "EXPLORATORY_V5_EVALUATOR_PROTOCOL_EN.md",
    "EXPLORATORY_V5_INPUT_PACKAGE_PROTOCOL_EN.md", "EQUAL60_EXPLORATORY_V5_PROTOCOL_EN.md",
}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_loads(text: str, context: str) -> Any:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            require(key not in out, f"Duplicate JSON key in {context}: {key}")
            out[key] = value
        return out

    def reject(token: str) -> None:
        raise ValueError(f"Nonfinite JSON token in {context}: {token}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def read_json(path: Path) -> Any:
    return strict_loads(Path(path).read_text(encoding="utf-8"), str(path))


def read_small_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, f"CSV schema mismatch: {path}")
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows), f"Empty/malformed CSV: {path}")
    return rows


def combinations() -> list[str]:
    result: list[str] = []
    for size in range(1, len(FAMILY_ORDER) + 1):
        import itertools
        result.extend("+".join(parts) for parts in itertools.combinations(FAMILY_ORDER, size))
    return result


COMBINATIONS = combinations()


def id_set_hash(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(map(str, values))))
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_fold(value: str, folds: int = FOLDS, seed: int = SEED) -> int:
    raw = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % folds


def population(table: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(table)), "ids": sorted(table["__id"].astype(str)),
        "groups": sorted(set(table["__group"].astype(str))),
        "label_counts": {str(k): int(v) for k, v in table["__label"].value_counts().sort_index().items()},
        "source_rows": {str(k): int(v) for k, v in table["__source"].value_counts().sort_index().items()},
        "source_groups": {str(k): int(v) for k, v in table.groupby("__source")["__group"].nunique().items()},
    }


def valid_train(table: pd.DataFrame) -> bool:
    if table["__label"].nunique() != 2:
        return False
    counts = table.groupby("__label")["__group"].nunique()
    return all(int(counts.get(label, 0)) >= 2 for label in (0, 1))


def base_folds(table: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_bucket = table["__group"].map(hash_fold)
    humans = sorted(table.loc[table["__label"] == 0, "__source"].unique())
    generators = sorted(table.loc[table["__label"] == 1, "__source"].unique())
    require(len(humans) >= 2 and len(generators) >= 2, "Source-held-out CV requires >=2 sources per class")
    folds: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    expected: list[tuple[str, str, int]] = []
    for label, sources, kind in ((0, humans, "human_source_holdout"), (1, generators, "generator_holdout")):
        for source in sources:
            for inner in range(FOLDS):
                expected.append((kind, source, inner))
                held = (table["__label"] == label) & (table["__source"] == source)
                test = table[(group_bucket == inner) & (held | (table["__label"] != label))].copy()
                train = table[(~held) & (group_bucket != inner) & (~table["__group"].isin(set(test["__group"])))].copy()
                if train["__label"].nunique() == 2 and test["__label"].nunique() == 2:
                    folds.append({"fold_type": kind, "heldout_source": source,
                                  "opposite_group_fold": inner, "train": train, "test": test})
                else:
                    omitted.append({"fold_type": kind, "heldout_source": source,
                                    "opposite_group_fold": inner,
                                    "reason": "one_or_both_classes_absent_in_train_or_test",
                                    "train": population(train), "test": population(test)})
    for inner in range(FOLDS):
        kind, source = "ordinary_group_holdout_descriptive", "__all_sources__"
        expected.append((kind, source, inner))
        test, train = table[group_bucket == inner].copy(), table[group_bucket != inner].copy()
        if train["__label"].nunique() == 2 and test["__label"].nunique() == 2:
            folds.append({"fold_type": kind, "heldout_source": source,
                          "opposite_group_fold": inner, "train": train, "test": test})
        else:
            omitted.append({"fold_type": kind, "heldout_source": source,
                            "opposite_group_fold": inner,
                            "reason": "one_or_both_classes_absent_in_train_or_test",
                            "train": population(train), "test": population(test)})
    require(len(folds) + len(omitted) == len(expected), "Fold enumeration accounting changed")
    return folds, omitted


def cap_training(table: pd.DataFrame, cap: int | str) -> pd.DataFrame:
    if cap == "all":
        return table.copy()
    keyed = table.assign(__source_key=table["__label"].astype(int).astype(str) + ":" + table["__source"].astype(str))
    parts: list[pd.DataFrame] = []
    for _, current in keyed.groupby("__source_key", sort=True):
        groups = sorted(current["__group"].unique(),
                        key=lambda group: hashlib.sha256(f"{SEED}|{group}".encode()).hexdigest())
        chosen = set(groups[:min(int(cap), len(groups))])
        parts.append(current[current["__group"].isin(chosen)].drop(columns="__source_key"))
    return pd.concat(parts, ignore_index=True)


def build_schedule(table: pd.DataFrame) -> tuple[list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]],
                                                  list[dict[str, Any]], dict[str, int]]:
    folds, omitted = base_folds(table)
    result: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for fold_index, fold in enumerate(folds):
        previous: set[str] = set()
        for cap in CAPS:
            train, test = cap_training(fold["train"], cap), fold["test"].copy()
            require(valid_train(train), "Capped training fold violates two-groups-per-class rule")
            train_ids, test_ids = set(train["__id"]), set(test["__id"])
            require(not train_ids & test_ids, "Train/test ID overlap")
            require(not set(train["__group"]) & set(test["__group"]), "Train/test group overlap")
            require(train_ids <= set(fold["train"]["__id"]), "Capped train is not a base-train subset")
            require(fold["heldout_source"] == "__all_sources__" or
                    fold["heldout_source"] not in set(train["__source"]), "Held source leaked into training")
            if cap != "all":
                require(int(train.groupby("__source")["__group"].nunique().max()) <= int(cap), "Group cap exceeded")
            require(previous <= train_ids, "Quantity caps are not nested")
            previous = train_ids
            record: dict[str, Any] = {
                "fold_index": fold_index, "quantity": cap, "fold_type": fold["fold_type"],
                "heldout_source": fold["heldout_source"], "opposite_group_fold": fold["opposite_group_fold"],
                "train": population(train), "test": population(test), "uncapped_train": population(fold["train"]),
                "train_id_set_sha256": id_set_hash(train["__id"]), "test_id_set_sha256": id_set_hash(test["__id"]),
            }
            record["fold_uid"] = digest(record)[:24]
            result.append((record, train, test, fold))
    accounting = {
        "valid_folds": len(folds), "primary_fits": len(result) * len(COMBINATIONS),
        "primary_prediction_rows": sum(len(test) * len(COMBINATIONS) for _, _, test, _ in result),
        "diagnostic_fits": sum(record["quantity"] == "all" for record, *_ in result) * len(COMBINATIONS) * 2,
        "diagnostic_prediction_rows": sum(len(test) * len(COMBINATIONS) * 2
                                          for record, _, test, _ in result if record["quantity"] == "all"),
    }
    return result, omitted, accounting


def load_package(package: Path, synthetic: bool) -> tuple[pd.DataFrame, dict[str, Any], str]:
    require(package.is_absolute() and package.resolve() == package and not package.is_symlink(), "Canonical package path required")
    commit_path = package / "COMMIT.json"
    commit = read_json(commit_path)
    package_files = {"metadata_60s.csv", "features_60s.csv", "family_config.json", "origin_ledger.csv",
                     "preparation_audit.json"}
    require(commit.get("status") == "committed" and commit.get("publication") == "exclusive hardlinks, COMMIT last",
            "Package is not a committed exclusive publication")
    require(set(commit.get("files", {})) == package_files and
            {p.name for p in package.iterdir()} == package_files | {"COMMIT.json"}, "Package has missing/orphan files")
    for name, proof in commit["files"].items():
        path = package / name
        require(path.is_file() and not path.is_symlink(), f"Invalid package product: {name}")
        require(proof == {"sha256": sha256_file(path), "bytes": path.stat().st_size}, f"Package product changed: {name}")
    prep = read_json(package / "preparation_audit.json")
    require(prep.get("schema_version") == 5 and prep.get("status") == "prepared_not_authorized_for_fitting" and
            prep.get("synthetic_test_only") is synthetic and prep.get("scoring_authorized") is False and
            prep.get("fitting_authorized") is False, "Invalid v5 preparation audit")
    require(prep.get("files_sha256") == {name: proof["sha256"] for name, proof in commit["files"].items()
                                         if name != "preparation_audit.json"}, "Preparation/COMMIT hashes differ")
    require(prep.get("contract_sha256") == digest(prep.get("contract")), "Preparation contract hash mismatch")
    metadata = read_small_csv(package / "metadata_60s.csv", META)
    features = read_small_csv(package / "features_60s.csv", ["id", *DESCRIPTORS])
    ledger = read_small_csv(package / "origin_ledger.csv", LEDGER)
    ids = [row["id"] for row in metadata]
    require(ids == [row["id"] for row in features] == [row["id"] for row in ledger] and
            len(ids) == len(set(ids)), "Package ID order/uniqueness differs across products")
    require(all(row["role"] == "development" and row["duration_view"] == "60s" and row["group_id"].strip()
                and row["source_group"].strip() for row in metadata), "Non-development or invalid metadata row")
    require(all(row["label"] in {"0", "1"} for row in metadata), "Label token must be exactly 0 or 1")
    if synthetic:
        require(0 < len(ids) <= 16 and all(value.startswith("synthetic_v5_") for value in ids), "Synthetic16 boundary violated")
    else:
        require(len(ids) == 2207 and Counter(row["source_group"] for row in metadata) == Counter(EXPECTED_REAL_SOURCES)
                and Counter(row["label"] for row in metadata) == Counter({"0": 1311, "1": 896}),
                "Real2207 source/label composition changed")
    table = pd.DataFrame(metadata).rename(columns={"id": "__id", "label": "__label",
                                                   "source_group": "__source", "group_id": "__group", "role": "__role"})
    table["__label"] = pd.to_numeric(table["__label"], errors="raise").astype(int)
    require(set(table["__label"]) == {0, 1}, "Both binary classes required")
    require(int(table.groupby("__source")["__label"].nunique().max()) == 1 and
            int(table.groupby("__group")["__label"].nunique().max()) == 1 and
            int(table.groupby("__group")["__source"].nunique().max()) == 1, "Source/group crosses class or source")
    for column in DESCRIPTORS:
        values: list[float] = []
        for row in features:
            token = row[column].strip()
            value = math.nan if token.lower() in ("", "nan", "na", "null") else float(token)
            require(math.isnan(value) or math.isfinite(value), f"Infinite descriptor: {column}")
            values.append(value)
        table[column] = values
    family_config = read_json(package / "family_config.json")
    combined = {**family_config.get("old_families", {}), **family_config.get("new_families", {})}
    require(family_config.get("schema_version") == 4 and all(combined.get(code, {}).get("state") == "available"
            and combined[code].get("columns") == FAMILIES[code] for code in FAMILY_ORDER), "Seven-family configuration changed")
    for code in ("V", "B", "A", "T"):
        require(combined.get(code, {}).get("state") == "planned" and combined[code].get("columns") == [],
                f"Unexpected active family in v5: {code}")
    require(int(prep.get("rows", -1)) == len(table), "Preparation row count mismatch")
    return table, prep, sha256_file(commit_path)


class StrictCSVStream:
    """Line-oriented CSV reader with byte hash, strict order, and EOF proof."""
    def __init__(self, path: Path, fields: list[str], expected_sha: str, expected_bytes: int):
        self.path, self.fields = Path(path), fields
        self.expected_sha, self.expected_bytes = expected_sha, int(expected_bytes)
        self.handle = self.path.open("rb")
        self.hash = hashlib.sha256(); self.bytes = 0; self.count = 0; self.closed = False
        try:
            header = self._raw_line("header")
            require(header is not None, f"Missing CSV header: {path}")
            require(self._parse(header) == fields, f"CSV schema/order mismatch: {path}")
        except BaseException:
            self.close()
            raise

    def _raw_line(self, context: str) -> bytes | None:
        raw = self.handle.readline()
        if raw == b"":
            return None
        self.hash.update(raw); self.bytes += len(raw)
        require(raw.endswith(b"\n"), f"Truncated non-newline-terminated {context}: {self.path}")
        return raw

    def _parse(self, raw: bytes) -> list[str]:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Non-UTF8 CSV: {self.path}") from exc
        rows = list(csv.reader([text]))
        require(len(rows) == 1, f"Malformed CSV row: {self.path}")
        return rows[0]

    def next(self, context: str) -> dict[str, str]:
        raw = self._raw_line(context)
        require(raw is not None, f"Truncated CSV stream at {context}: {self.path}")
        values = self._parse(raw)
        require(len(values) == len(self.fields), f"Malformed field count at {context}: {self.path}")
        self.count += 1
        return dict(zip(self.fields, values))

    def finish(self) -> str:
        require(not self.closed, f"Stream already closed: {self.path}")
        extra = self._raw_line("unexpected extra row")
        require(extra is None, f"Unexpected/duplicate/out-of-order extra row: {self.path}")
        self.handle.close(); self.closed = True
        require(self.bytes == self.expected_bytes, f"Stream byte count changed: {self.path}")
        observed = self.hash.hexdigest()
        require(observed == self.expected_sha, f"Stream SHA mismatch: {self.path}")
        return observed

    def close(self) -> None:
        if not self.closed:
            self.handle.close(); self.closed = True


class StrictJSONLStream:
    def __init__(self, path: Path, expected_sha: str, expected_bytes: int):
        self.path = Path(path); self.expected_sha = expected_sha; self.expected_bytes = int(expected_bytes)
        self.handle = self.path.open("rb"); self.hash = hashlib.sha256(); self.bytes = 0; self.count = 0; self.closed = False

    def _line(self, context: str) -> bytes | None:
        raw = self.handle.readline()
        if raw == b"": return None
        self.hash.update(raw); self.bytes += len(raw)
        require(raw.endswith(b"\n"), f"Truncated JSONL line at {context}: {self.path}")
        return raw

    def next(self, context: str) -> Any:
        raw = self._line(context)
        require(raw is not None, f"Truncated JSONL stream at {context}: {self.path}")
        self.count += 1
        return strict_loads(raw.decode("utf-8"), f"{self.path}:{self.count}")

    def finish(self) -> str:
        require(not self.closed, f"Stream already closed: {self.path}")
        require(self._line("unexpected extra line") is None, f"Unexpected/duplicate JSONL line: {self.path}")
        self.handle.close(); self.closed = True
        require(self.bytes == self.expected_bytes, f"JSONL byte count changed: {self.path}")
        observed = self.hash.hexdigest(); require(observed == self.expected_sha, f"JSONL SHA mismatch: {self.path}")
        return observed

    def close(self) -> None:
        if not self.closed: self.handle.close(); self.closed = True


def verify_result_commit(result: Path) -> tuple[dict[str, Any], dict[str, tuple[int, int, int]]]:
    require(result.is_absolute() and result.resolve() == result and not result.is_symlink(), "Canonical result path required")
    commit_path = result / "COMMIT.json"
    require(commit_path.is_file() and not commit_path.is_symlink(), "Completed COMMIT.json required; partial/orphan run refused")
    marker = read_json(commit_path)
    require(marker.get("status") == "committed" and
            marker.get("publication") == "exclusive directory reservation; hardlink COMMIT last", "Invalid result COMMIT")
    require(set(marker.get("files", {})) == EXPECTED_RESULT_FILES and
            {p.name for p in result.iterdir()} == EXPECTED_RESULT_FILES | {"COMMIT.json"}, "Missing/orphan result products")
    snapshots: dict[str, tuple[int, int, int]] = {}
    for name, proof in marker["files"].items():
        path = result / name
        require(path.is_file() and not path.is_symlink() and Path(name).name == name, f"Unsafe result product: {name}")
        stat = path.stat(); snapshots[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        require(stat.st_size == proof.get("bytes") and SHA_RE.fullmatch(str(proof.get("sha256"))) is not None,
                f"Invalid COMMIT entry: {name}")
    stat = commit_path.stat(); snapshots["COMMIT.json"] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return marker, snapshots


def recheck_snapshots(result: Path, snapshots: dict[str, tuple[int, int, int]]) -> None:
    require({p.name for p in result.iterdir()} == set(snapshots), "Result directory changed during audit")
    for name, before in snapshots.items():
        path = result / name
        require(path.is_file() and not path.is_symlink(), f"Result product replaced: {name}")
        stat = path.stat()
        require((stat.st_ino, stat.st_size, stat.st_mtime_ns) == before, f"Result product changed during audit: {name}")


def compare_record(actual: dict[str, str], expected: dict[str, Any], *, integer: set[str] = set(),
                   floating: set[str] = set(), context: str) -> None:
    require(set(actual) == set(expected), f"Field set mismatch at {context}")
    for key, wanted in expected.items():
        token = actual[key]
        if wanted is None:
            require(token == "", f"Expected blank {key} at {context}")
        elif key in integer:
            require(token == str(int(wanted)), f"Integer mismatch {key} at {context}")
        elif key in floating:
            got = float(token)
            require(math.isfinite(got) and math.isclose(got, float(wanted), rel_tol=1e-11, abs_tol=1e-12),
                    f"Numeric mismatch {key} at {context}: {got} != {wanted}")
        else:
            require(token == str(wanted), f"Identity/order mismatch {key} at {context}: {token!r} != {wanted!r}")


def validate_prediction_row(row: dict[str, str], expected_identity: dict[str, Any], replayed_score: float,
                            context: str) -> tuple[float, int]:
    require(set(row) == set(PRED), f"Prediction field set mismatch at {context}")
    for key, value in expected_identity.items():
        require(row[key] == str(value), f"Prediction identity/label/order mismatch at {context}: {key}")
    score = float(row["score"])
    require(math.isfinite(score), f"Nonfinite saved prediction score at {context}")
    require(math.isclose(score, float(replayed_score), rel_tol=2e-10, abs_tol=2e-12),
            f"Saved score differs from independent affine replay at {context}")
    require(row["threshold"] == "0.5", f"Prediction threshold changed at {context}")
    try:
        decision = int(row["predicted_label"])
    except ValueError as exc:
        raise ValueError(f"Invalid decision token at {context}") from exc
    require(decision in (0, 1) and decision == int(score >= 0.5) == int(replayed_score >= 0.5),
            f"Saved threshold decision mismatch at {context}")
    return score, decision


def validate_schedule_records(actual: list[Any], expected: list[dict[str, Any]]) -> None:
    require(len(actual) == len(expected), "Fold registry row count/truncation mismatch")
    for index, (left, right) in enumerate(zip(actual, expected), 1):
        require(left == right, f"Fold registry membership/order mismatch at row {index}")


def validate_fixed_core_bindings(files: dict[str, str]) -> None:
    for name, expected in FIXED_CORE.items():
        matches = [value for path, value in files.items() if Path(path).name == name]
        require(matches == [expected], f"Original frozen core pin mismatch: {name}")


def independent_weights(train: pd.DataFrame) -> np.ndarray:
    labels = train["__label"].to_numpy(int)
    source_keys = (train["__label"].astype(int).astype(str) + ":" + train["__source"].astype(str)).to_numpy(str)
    weights = np.zeros(len(train), dtype=float)
    for label in (0, 1):
        label_positions = np.flatnonzero(labels == label)
        sources = sorted(set(source_keys[label_positions]))
        require(bool(sources), "Training class absent during weight audit")
        for source in sources:
            positions = label_positions[source_keys[label_positions] == source]
            groups_at_positions = train.iloc[positions]["__group"].astype(str).to_numpy()
            groups = sorted(set(groups_at_positions))
            for group in groups:
                selected = positions[groups_at_positions == group]
                weights[selected] = 0.5 / (len(sources) * len(groups) * len(selected))
    weights *= len(train) / weights.sum()
    require(np.isfinite(weights).all() and np.all(weights > 0) and math.isclose(float(weights.sum()), len(train), rel_tol=1e-13),
            "Invalid independently reconstructed weights")
    return weights


class TrainingCache:
    def __init__(self, train: pd.DataFrame):
        self.train = train.reset_index(drop=True)
        self.raw = self.train[DESCRIPTORS].to_numpy(float)
        self.missing = ~np.isfinite(self.raw)
        self.medians = np.asarray([float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.0
                                   for col in self.raw.T])
        self.imputed = np.where(self.missing, self.medians, self.raw)
        self.weights = independent_weights(self.train)
        self.y = self.train["__label"].to_numpy(float)
        self.value_mean = np.average(self.imputed, axis=0, weights=self.weights)
        self.value_scale = np.sqrt(np.average((self.imputed - self.value_mean) ** 2, axis=0, weights=self.weights))
        self.value_scale[self.value_scale < 1e-8] = 1.0
        miss = self.missing.astype(float)
        self.missing_mean = np.average(miss, axis=0, weights=self.weights)
        self.missing_scale = np.sqrt(np.average((miss - self.missing_mean) ** 2, axis=0, weights=self.weights))
        self.missing_scale[self.missing_scale < 1e-8] = 1.0
        self.positions = {column: index for index, column in enumerate(DESCRIPTORS)}
        self.source_counts = Counter((self.train["__label"].astype(int).astype(str) + ":" +
                                      self.train["__source"].astype(str)).tolist())

    def arrays(self, columns: list[str], mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray([self.positions[name] for name in columns], dtype=int)
        if mode == "values_plus_missing":
            x = np.concatenate((self.imputed[:, indices], self.missing[:, indices].astype(float)), axis=1)
            mean = np.concatenate((self.value_mean[indices], self.missing_mean[indices]))
            scale = np.concatenate((self.value_scale[indices], self.missing_scale[indices]))
        elif mode == "median_only":
            x, mean, scale = self.imputed[:, indices], self.value_mean[indices], self.value_scale[indices]
        elif mode == "missingness_only":
            x = self.missing[:, indices].astype(float)
            mean, scale = self.missing_mean[indices], self.missing_scale[indices]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return indices, x, mean, scale, (x - mean) / scale


def numeric_array(model: dict[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(model.get(key), dtype=float)
    require(value.shape == (length,) and np.isfinite(value).all(), f"Invalid model vector: {key}")
    return value


def validate_model(model: dict[str, Any], cache: TrainingCache, columns: list[str], mode: str) -> tuple[np.ndarray, float]:
    exact_keys = {"columns", "medians", "mean", "scale", "coefficients_with_intercept", "ridge", "threshold",
                  "feature_mode", "training_rows", "training_source_counts", "weighting", "missing_value_policy",
                  "observed_fraction_by_column", "model_type", "prediction_link"}
    require(set(model) == exact_keys, "Saved model field set changed")
    require(model["columns"] == columns and model["feature_mode"] == mode and model["training_rows"] == len(cache.train),
            "Saved model identity/training-row mismatch")
    require(model["ridge"] == 10.0 and model["threshold"] == 0.5 and
            model["model_type"] == "weighted_ridge_linear_probability" and model["prediction_link"] == "identity",
            "Model/ridge/threshold/link changed")
    require(model["weighting"] == "classes equal; sources equal within class; groups equal within source; samples equal within group"
            and model["missing_value_policy"] == POLICIES[mode], "Weighting or missingness policy changed")
    indices, x, expected_mean, expected_scale, standardized = cache.arrays(columns, mode)
    medians = numeric_array(model, "medians", len(columns))
    mean = numeric_array(model, "mean", x.shape[1]); scale = numeric_array(model, "scale", x.shape[1])
    coefficients = numeric_array(model, "coefficients_with_intercept", x.shape[1] + 1)
    require(np.allclose(medians, cache.medians[indices], rtol=1e-13, atol=1e-14), "Training median mismatch")
    require(np.allclose(mean, expected_mean, rtol=1e-12, atol=1e-13), "Training weighted-mean mismatch")
    require(np.allclose(scale, expected_scale, rtol=1e-12, atol=1e-13) and np.all(scale > 0),
            "Training weighted-scale mismatch")
    require(model["training_source_counts"] == dict(sorted(cache.source_counts.items())), "Training source counts mismatch")
    observed = {name: float((~cache.missing[:, cache.positions[name]]).mean()) for name in columns}
    require(set(model["observed_fraction_by_column"]) == set(observed), "Observed-fraction columns changed")
    require(all(math.isclose(float(model["observed_fraction_by_column"][name]), value, rel_tol=1e-13, abs_tol=1e-14)
                for name, value in observed.items()), "Observed-fraction value mismatch")
    design = np.column_stack((np.ones(len(cache.train)), standardized))
    residual = design.T @ (cache.weights * (design @ coefficients - cache.y))
    residual[1:] += 10.0 * coefficients[1:]
    residual_norm = float(np.max(np.abs(residual)))
    rhs_scale = 1.0 + float(np.max(np.abs(design.T @ (cache.weights * cache.y))))
    require(residual_norm <= 2e-9 * rhs_scale, f"Ridge normal-equation residual too large: {residual_norm}")
    return coefficients, residual_norm


def replay_scores(test: pd.DataFrame, columns: list[str], mode: str, model: dict[str, Any]) -> np.ndarray:
    raw = test[columns].to_numpy(float); missing = ~np.isfinite(raw)
    medians = np.asarray(model["medians"], dtype=float); imputed = np.where(missing, medians, raw)
    if mode == "values_plus_missing": matrix = np.concatenate((imputed, missing.astype(float)), axis=1)
    elif mode == "median_only": matrix = imputed
    elif mode == "missingness_only": matrix = missing.astype(float)
    else: raise ValueError(f"Unknown mode: {mode}")
    mean = np.asarray(model["mean"], dtype=float); scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coefficients_with_intercept"], dtype=float)
    scores = [math.fsum([float(coef[0]), *(((row - mean) / scale) * coef[1:]).tolist()]) for row in matrix]
    output = np.asarray(scores, dtype=float)
    require(np.isfinite(output).all(), "Nonfinite independently replayed score")
    return output


def independent_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int); s = np.asarray(list(scores), dtype=float)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    require(n0 and n1, "AUC requested for one-class data")
    order = sorted(range(len(s)), key=lambda i: (float(s[i]), i)); ranks = np.empty(len(s), float)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and s[order[end]] == s[order[position]]: end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]: ranks[index] = average_rank
        position = end
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def independent_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=int); s = np.asarray(list(scores), dtype=float); pred = s >= 0.5
    tp = int((pred & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum())
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {"roc_auc": independent_auc(y, s), "balanced_accuracy": 0.5 * (sensitivity + specificity),
            "ai_sensitivity": sensitivity, "human_specificity": specificity,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def expected_pair_rows(test: pd.DataFrame, scores: np.ndarray, record: dict[str, Any], model_uid: str) -> list[dict[str, Any]]:
    humans = sorted(test.loc[test["__label"] == 0, "__source"].unique())
    ais = sorted(test.loc[test["__label"] == 1, "__source"].unique())
    if record["fold_type"] == "ordinary_group_holdout_descriptive": pairs = [(h, a) for h in humans for a in ais]
    elif record["fold_type"] == "human_source_holdout": pairs = [(record["heldout_source"], a) for a in ais]
    else: pairs = [(h, record["heldout_source"]) for h in humans]
    result = []
    for human, ai in pairs:
        mask = (((test["__label"] == 0) & (test["__source"] == human)) |
                ((test["__label"] == 1) & (test["__source"] == ai))).to_numpy()
        part = test.iloc[np.flatnonzero(mask)]
        if part["__label"].nunique() != 2 or any(int(part.loc[part["__label"] == label, "__group"].nunique()) < 1 for label in (0, 1)):
            continue
        result.append({"model_uid": model_uid, "human_source": human, "ai_source": ai,
                       "test_human": int((part["__label"] == 0).sum()), "test_ai": int((part["__label"] == 1).sum()),
                       "test_human_groups": int(part.loc[part["__label"] == 0, "__group"].nunique()),
                       "test_ai_groups": int(part.loc[part["__label"] == 1, "__group"].nunique()),
                       "threshold": 0.5, **independent_metrics(part["__label"], scores[mask])})
    return result


def expected_source_rows(test: pd.DataFrame, scores: np.ndarray, model_uid: str) -> list[dict[str, Any]]:
    result = []
    for source, part in test.groupby("__source", sort=True):
        positions = test.index.get_indexer(part.index); positive = scores[positions] >= 0.5
        label, n = int(part["__label"].iloc[0]), len(part); group_rates = []
        for group in sorted(set(part["__group"])):
            mask = part["__group"].eq(group).to_numpy(); group_rates.append(float(positive[mask].mean()))
        pos, rate = int(positive.sum()), float(positive.mean()); equal = math.fsum(group_rates) / len(group_rates)
        result.append({"model_uid": model_uid, "source_group": source, "label": label, "rows": n,
                       "components": len(group_rates), "threshold": 0.5, "tp": pos if label else 0,
                       "tn": n - pos if not label else 0, "fp": pos if not label else 0, "fn": n - pos if label else 0,
                       "recording_positive_rate": rate, "recording_negative_rate": 1 - rate,
                       "recording_ai_sensitivity": rate if label else None,
                       "recording_human_specificity": 1 - rate if not label else None,
                       "equal_component_positive_rate": equal, "equal_component_negative_rate": 1 - equal,
                       "equal_component_ai_sensitivity": equal if label else None,
                       "equal_component_human_specificity": 1 - equal if not label else None})
    return result


class OOFAccumulator:
    """Compact unique-OOF correctness accumulator with global-group weighting."""
    def __init__(self, schedule: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]]):
        group_rows: dict[tuple[str, str, str], Counter[str]] = {}
        row_ids: dict[tuple[str, str, str], set[str]] = {}
        for record, _, test, _ in schedule:
            if record["quantity"] != "all": continue
            for source, part in test.groupby("__source", sort=True):
                base = (record["fold_type"], record["heldout_source"], str(source))
                ids = set(part["__id"].astype(str)); prior = row_ids.setdefault(base, set())
                require(not prior & ids, f"Duplicate OOF row within arm/source: {base}")
                prior.update(ids); counter = group_rows.setdefault(base, Counter())
                counter.update(part["__group"].astype(str))
        self.denominators = {key: dict(sorted(value.items())) for key, value in group_rows.items()}
        self.indices = {key: {group: index for index, group in enumerate(groups)}
                        for key, groups in ((key, list(value)) for key, value in self.denominators.items())}
        self.states: dict[tuple[str, str, str, str, str, str, str], tuple[np.ndarray, int]] = {}

    def add(self, category: str, combination: str, mode: str, quantity: int | str, record: dict[str, Any],
            test: pd.DataFrame, decisions: np.ndarray) -> None:
        correctness = decisions == test["__label"].to_numpy(int)
        for source, part in test.groupby("__source", sort=True):
            base = (record["fold_type"], record["heldout_source"], str(source))
            key = (category, combination, mode, str(quantity), *base)
            array, rows = self.states.get(key, (np.zeros(len(self.denominators[base]), dtype=np.int64), 0))
            positions = test.index.get_indexer(part.index); current = correctness[positions]
            for group in sorted(set(part["__group"])):
                mask = part["__group"].eq(group).to_numpy()
                array[self.indices[base][str(group)]] += int(current[mask].sum())
            self.states[key] = (array, rows + len(part))

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for key in sorted(self.states):
            array, rows = self.states[key]; base = key[-3:]; denominators = self.denominators[base]
            denom = np.asarray(list(denominators.values()), dtype=float)
            require(rows == int(denom.sum()) and np.all(array >= 0) and np.all(array <= denom),
                    f"OOF source denominator mismatch: {key}")
            correct = int(array.sum()); component_rate = float(np.mean(array / denom))
            yield {"category": key[0], "combination": key[1], "feature_mode": key[2], "quantity": key[3],
                   "fold_type": key[4], "heldout_source": key[5], "source_group": key[6], "rows": rows,
                   "components": len(denom), "correct": correct, "recording_correct_rate": correct / rows,
                   "equal_global_component_correct_rate": component_rate}

    def finalize(self) -> dict[str, Any]:
        h = hashlib.sha256(); total_rows = 0; total_components = 0; cells = 0
        for record in self.iter_records():
            h.update(canonical(record) + b"\n")
            total_rows += int(record["rows"]); total_components += int(record["components"])
            cells += 1
        return {"endpoint_cells": cells, "endpoint_rows_summed_across_cells": total_rows,
                "endpoint_components_summed_across_cells": total_components, "duplicate_oof_ids_within_arm_source": 0,
                "denominator_checks_passed": len(self.states), "canonical_endpoints_sha256": h.hexdigest()}


def validate_contract(contract: dict[str, Any], table: pd.DataFrame, prep: dict[str, Any], package: Path,
                      schedule: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]],
                      omitted: list[dict[str, Any]], accounting: dict[str, int], synthetic: bool) -> None:
    records = [record for record, *_ in schedule]
    require(contract.get("schema_version") == 5 and contract.get("authorized_stage") == STAGE and
            contract.get("synthetic_test_only") is synthetic, "Contract stage/schema/synthetic flag changed")
    require(contract.get("input_package") == str(package) and contract.get("package_contract_sha256") == prep["contract_sha256"],
            "Contract package binding mismatch")
    require(contract.get("rows") == len(table) and contract.get("label_counts") ==
            {str(k): int(v) for k, v in table["__label"].value_counts().sort_index().items()} and
            contract.get("source_counts") == {str(k): int(v) for k, v in table["__source"].value_counts().sort_index().items()},
            "Contract cohort counts changed")
    require(contract.get("families") == FAMILIES and contract.get("combinations") == COMBINATIONS and
            contract.get("seed") == SEED and contract.get("quantities") == list(CAPS) and
            contract.get("opposite_group_folds") == FOLDS, "Contract grid/family/seed changed")
    require(contract.get("fixed_ridge") == 10.0 and contract.get("fixed_threshold") == 0.5 and
            contract.get("positive_rule") == "score >= 0.5" and contract.get("prediction_link") == "raw_identity_unclipped",
            "Contract model/decision rule changed")
    require(contract.get("primary_feature_mode") == MODES[0] and contract.get("all_cap_diagnostics") == list(MODES[1:]),
            "Primary/diagnostic mode changed")
    require(contract.get("schedule") == records and contract.get("schedule_sha256") == digest(records) and
            contract.get("omitted_source_or_group_cells") == omitted and contract.get("accounting") == accounting,
            "Independently reconstructed schedule/accounting mismatch")
    for flag in ("selection", "threshold_tuning", "full_cohort_refit", "historical_locked_or_pilot_scoring",
                 "source_transfer_J_computed", "source_ranking", "winner_selection"):
        require(contract.get(flag) is False, f"Forbidden contract flag enabled: {flag}")
    cross = contract.get("cross_fold_reporting", {})
    require(cross.get("source_recording", "").startswith("within arm/candidate/cap/mode/source") and
            "never mean of fold component means" in cross.get("source_component", "") and
            cross.get("pooled_out_of_fold_auc") is False and cross.get("pooling_across_held_source_arms") is False,
            "Cross-fold estimand contract changed")
    files = contract.get("input_files_sha256", {})
    require(isinstance(files, dict) and len(files) == 15 and all(Path(path).is_absolute() for path in files),
            "Expected exactly 15 absolute input/code bindings")
    basenames = Counter(Path(path).name for path in files)
    require(set(basenames) == EXPECTED_BINDING_BASENAMES and all(value == 1 for value in basenames.values()),
            "Input/code binding basename inventory changed")
    for path, expected in files.items():
        require(SHA_RE.fullmatch(str(expected)) is not None and sha256_file(Path(path)) == expected,
                f"Bound input/code changed: {path}")
    validate_fixed_core_bindings(files)
    runtime = contract.get("runtime", {})
    executable = Path(runtime.get("python_executable", ""))
    require(executable.is_absolute() and executable.is_file() and
            sha256_file(executable) == runtime.get("python_executable_sha256"), "Frozen Python executable changed")
    if not synthetic:
        require(runtime.get("threads") == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "Real frozen runtime thread settings changed")
        require(accounting == EXPECTED_REAL and Counter(record["fold_type"] for record, *_ in schedule if record["quantity"] == "all") ==
                Counter({"human_source_holdout": 28, "generator_holdout": 10,
                         "ordinary_group_holdout_descriptive": 5}), "Real v5 expected accounting/folds changed")
        require(len(omitted) == 2 and all(row["fold_type"] == "human_source_holdout" and
                row["heldout_source"] == "human_saraga_hindustani_v1" for row in omitted), "Real omission ledger changed")


def validate_authorization(receipt_path: Path, receipt_sha: str, result_auth: Any,
                           manifest: dict[str, Any], contract: dict[str, Any]) -> None:
    require(receipt_path.is_absolute() and receipt_path.is_file() and not receipt_path.is_symlink(), "Canonical frozen receipt required")
    require(SHA_RE.fullmatch(receipt_sha) is not None and sha256_file(receipt_path) == receipt_sha, "Frozen receipt SHA mismatch")
    receipt = read_json(receipt_path)
    review = receipt.get("independent_review", {})
    require(receipt.get("status") == "frozen" and receipt.get("authorized_stage") == STAGE and
            receipt.get("contract") == contract and receipt.get("contract_sha256") == digest(contract),
            "Frozen receipt does not bind exact contract")
    require(review.get("approved") is True and review.get("reviewer") == "root" and
            isinstance(review.get("reviewed_utc"), str) and review["reviewed_utc"].strip(), "Parent frozen review absent")
    require(result_auth == receipt, "Saved frozen authorization differs from supplied receipt")
    require(manifest.get("authorization") == {"status": "frozen_verified", "contract_sha256": digest(contract),
                                               "receipt_sha256": receipt_sha}, "Run authorization proof changed")


def expected_common(record: dict[str, Any], combination: str, mode: str, line: int, model: dict[str, Any]) -> dict[str, Any]:
    identity = {"combination": combination, "feature_mode": mode, "quantity": record["quantity"], "fold_uid": record["fold_uid"]}
    return {"model_uid": digest(identity)[:24], "model_sha256": digest({"model": model, "threshold": 0.5}),
            "model_jsonl_line": line, **identity, "fold_index": record["fold_index"], "fold_type": record["fold_type"],
            "heldout_source": record["heldout_source"], "opposite_group_fold": record["opposite_group_fold"],
            "train_id_set_sha256": record["train_id_set_sha256"], "test_id_set_sha256": record["test_id_set_sha256"],
            "training_rows": record["train"]["rows"], "test_rows": record["test"]["rows"]}


def audit(result_dir: Path, package_dir: Path, receipt_path: Path, receipt_sha: str, synthetic: bool) -> dict[str, Any]:
    result, package = Path(result_dir), Path(package_dir)
    auditor_path = Path(__file__).resolve()
    protocol_path = auditor_path.parent.parent / "AUDIT_EQUAL60_V5_PROTOCOL_EN.md"
    require(protocol_path.is_file() and not protocol_path.is_symlink(), "Independent audit protocol is missing")
    auditor_sha_at_start, protocol_sha_at_start = sha256_file(auditor_path), sha256_file(protocol_path)
    audit_executable = Path(sys.executable).resolve()
    audit_runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "python_executable": str(audit_executable),
                     "python_executable_sha256": sha256_file(audit_executable)}
    marker, snapshots = verify_result_commit(result)
    table, prep, package_commit_sha = load_package(package, synthetic)
    manifest = read_json(result / "run_manifest.json"); result_auth = read_json(result / "frozen_authorization.json")
    contract = manifest.get("contract")
    require(isinstance(contract, dict), "Run manifest lacks contract")
    schedule, omitted, accounting = build_schedule(table)
    validate_contract(contract, table, prep, package, schedule, omitted, accounting, synthetic)
    validate_authorization(receipt_path, receipt_sha, result_auth, manifest, contract)
    require(manifest.get("schema_version") == 5 and manifest.get("stage") == STAGE and
            manifest.get("synthetic_test_only") is synthetic and manifest.get("status") == "completed_exploratory_cv",
            "Run manifest stage/status changed")
    require(manifest.get("model_instances") == accounting["primary_fits"] + accounting["diagnostic_fits"] and
            manifest.get("all_models_are_training_fold_only") is True and manifest.get("full_cohort_refit") is False and
            manifest.get("selection") is False and manifest.get("threshold_tuning") is False and
            manifest.get("historical_locked_or_pilot_scoring") is False and manifest.get("source_transfer_J_computed") is False and
            manifest.get("independent_numerical_audit_performed") is False, "Run manifest safety/accounting flags changed")
    require(read_json(result / "omitted_fold_cells.json") == omitted, "Saved omission ledger differs")

    # Verify the small committed products now. Large products are hashed while consumed.
    streamed_names = {"fold_registry.jsonl", "fold_models.jsonl"} | {
        f"{category}_{name}.csv" for category in ("primary", "diagnostic")
        for name in ("model_index", "predictions", "pair_metrics", "pooled_metrics", "per_source_endpoints")}
    checked_hashes: dict[str, str] = {}
    for name, proof in marker["files"].items():
        if name not in streamed_names:
            observed = sha256_file(result / name); require(observed == proof["sha256"], f"Committed file changed: {name}")
            checked_hashes[name] = observed

    fold_stream = StrictJSONLStream(result / "fold_registry.jsonl", **{
        "expected_sha": marker["files"]["fold_registry.jsonl"]["sha256"],
        "expected_bytes": marker["files"]["fold_registry.jsonl"]["bytes"]})
    try:
        for index, (record, *_) in enumerate(schedule, 1):
            require(fold_stream.next(f"fold registry row {index}") == record, f"Fold registry mismatch at row {index}")
        checked_hashes["fold_registry.jsonl"] = fold_stream.finish()
    finally: fold_stream.close()

    streams: dict[str, dict[str, StrictCSVStream]] = {}
    model_stream: StrictJSONLStream | None = None
    oof = OOFAccumulator(schedule); model_count = primary_count = diagnostic_count = score_count = 0
    max_residual = 0.0; max_score_error = 0.0; min_threshold_margin = math.inf
    try:
        for category in ("primary", "diagnostic"):
            streams[category] = {}
            for name, fields in (("model_index", INDEX), ("predictions", PRED), ("pair_metrics", PAIR),
                                 ("pooled_metrics", POOLED), ("per_source_endpoints", SOURCE)):
                filename = f"{category}_{name}.csv"; proof = marker["files"][filename]
                streams[category][name] = StrictCSVStream(result / filename, fields, proof["sha256"], proof["bytes"])
        model_proof = marker["files"]["fold_models.jsonl"]
        model_stream = StrictJSONLStream(result / "fold_models.jsonl", model_proof["sha256"], model_proof["bytes"])
        for record, train, original_test, _ in schedule:
            test = original_test.reset_index(drop=True); cache = TrainingCache(train)
            identities = list(test[["__id", "__label", "__source", "__group", "__role"]].itertuples(index=False, name=None))
            for combination in COMBINATIONS:
                columns = sum((FAMILIES[family] for family in combination.split("+")), [])
                modes = MODES if record["quantity"] == "all" else MODES[:1]
                for mode in modes:
                    model_count += 1; category = "primary" if mode == MODES[0] else "diagnostic"
                    if category == "primary": primary_count += 1
                    else: diagnostic_count += 1
                    saved = model_stream.next(f"model {model_count}")
                    require(isinstance(saved, dict) and "model" in saved, f"Malformed saved model {model_count}")
                    model = saved["model"]; common = expected_common(record, combination, mode, model_count, model)
                    require(set(saved) == set(common) | {"model"} and all(saved[key] == value for key, value in common.items()),
                            f"Model identity/order/hash mismatch at line {model_count}")
                    index_row = streams[category]["model_index"].next(f"model index {model_count}")
                    compare_record(index_row, common,
                                   integer={"model_jsonl_line", "fold_index", "opposite_group_fold", "training_rows", "test_rows"},
                                   context=f"model index {model_count}")
                    _, residual = validate_model(model, cache, columns, mode); max_residual = max(max_residual, residual)
                    replay = replay_scores(test, columns, mode, model); observed_scores = np.empty(len(test), float)
                    observed_decisions = np.empty(len(test), int)
                    for position, (identity, replayed) in enumerate(zip(identities, replay)):
                        row = streams[category]["predictions"].next(f"model {model_count} prediction {position + 1}")
                        row_id, label, source, group_id, role = identity
                        expected_identity = {"model_uid": common["model_uid"], "row_id": str(row_id), "label": int(label),
                                             "source_group": str(source), "group_id": str(group_id), "role": str(role)}
                        score, decision = validate_prediction_row(
                            row, expected_identity, float(replayed), f"model {model_count}, row {position + 1}")
                        max_score_error = max(max_score_error, abs(score - float(replayed)))
                        min_threshold_margin = min(min_threshold_margin, abs(score - 0.5))
                        observed_scores[position], observed_decisions[position] = score, decision; score_count += 1
                    for pair_index, expected in enumerate(expected_pair_rows(test, observed_scores, record, common["model_uid"]), 1):
                        actual = streams[category]["pair_metrics"].next(f"model {model_count} pair {pair_index}")
                        compare_record(actual, expected,
                                       integer={"test_human", "test_ai", "test_human_groups", "test_ai_groups", "tp", "tn", "fp", "fn"},
                                       floating={"threshold", "roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"},
                                       context=f"model {model_count} pair {pair_index}")
                    pooled = {"model_uid": common["model_uid"], "test_rows": len(test),
                              "test_groups": int(test["__group"].nunique()), "threshold": 0.5,
                              **independent_metrics(test["__label"], observed_scores)}
                    compare_record(streams[category]["pooled_metrics"].next(f"model {model_count} pooled"), pooled,
                                   integer={"test_rows", "test_groups", "tp", "tn", "fp", "fn"},
                                   floating={"threshold", "roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"},
                                   context=f"model {model_count} pooled")
                    source_float = {"threshold", "recording_positive_rate", "recording_negative_rate",
                                    "recording_ai_sensitivity", "recording_human_specificity",
                                    "equal_component_positive_rate", "equal_component_negative_rate",
                                    "equal_component_ai_sensitivity", "equal_component_human_specificity"}
                    for source_index, expected in enumerate(expected_source_rows(test, observed_scores, common["model_uid"]), 1):
                        compare_record(streams[category]["per_source_endpoints"].next(
                            f"model {model_count} source {source_index}"), expected,
                            integer={"label", "rows", "components", "tp", "tn", "fp", "fn"}, floating=source_float,
                            context=f"model {model_count} source {source_index}")
                    oof.add(category, combination, mode, record["quantity"], record, test, observed_decisions)
            print(json.dumps({"event": "audited_fold_cap", "fold_index": record["fold_index"],
                              "fold_type": record["fold_type"], "heldout_source": record["heldout_source"],
                              "quantity": record["quantity"], "models_checked": model_count,
                              "predictions_checked": score_count}, sort_keys=True), flush=True)
        require(model_count == accounting["primary_fits"] + accounting["diagnostic_fits"] and
                primary_count == accounting["primary_fits"] and diagnostic_count == accounting["diagnostic_fits"] and
                score_count == accounting["primary_prediction_rows"] + accounting["diagnostic_prediction_rows"],
                "Model/prediction accounting mismatch")
        checked_hashes["fold_models.jsonl"] = model_stream.finish()
        stream_counts: dict[str, dict[str, int]] = {}
        for category in ("primary", "diagnostic"):
            stream_counts[category] = {}
            for name, stream in streams[category].items():
                checked_hashes[f"{category}_{name}.csv"] = stream.finish(); stream_counts[category][name] = stream.count
    finally:
        if model_stream is not None: model_stream.close()
        for group in streams.values():
            for stream in group.values(): stream.close()
    require(manifest.get("stream_counts") == stream_counts, "Run manifest stream counts differ from audited streams")
    require(stream_counts["primary"]["model_index"] == accounting["primary_fits"] and
            stream_counts["primary"]["predictions"] == accounting["primary_prediction_rows"] and
            stream_counts["diagnostic"]["model_index"] == accounting["diagnostic_fits"] and
            stream_counts["diagnostic"]["predictions"] == accounting["diagnostic_prediction_rows"], "Primary/diagnostic stream accounting mismatch")
    require(set(checked_hashes) == EXPECTED_RESULT_FILES and
            all(checked_hashes[name] == marker["files"][name]["sha256"] for name in checked_hashes),
            "Not every committed result product was hash-checked")
    recheck_snapshots(result, snapshots)
    # Re-hash small mutable dependencies and the receipt after the long stream.
    for path, expected in contract["input_files_sha256"].items():
        require(sha256_file(Path(path)) == expected, f"Bound input changed during audit: {path}")
    require(sha256_file(receipt_path) == receipt_sha, "Frozen receipt changed during audit")
    require(sha256_file(auditor_path) == auditor_sha_at_start, "Auditor code changed during audit")
    require(sha256_file(protocol_path) == protocol_sha_at_start, "Auditor protocol changed during audit")
    oof_summary = oof.finalize()
    result_files = dict(sorted(checked_hashes.items())); result_files["COMMIT.json"] = sha256_file(result / "COMMIT.json")
    return {
        "status": "passed", "schema_version": 5, "stage": STAGE, "synthetic_test_only": synthetic,
        "rows": len(table), "label_counts": contract["label_counts"], "source_counts": contract["source_counts"],
        "families": {key: len(value) for key, value in FAMILIES.items()}, "candidates": len(COMBINATIONS),
        "caps": list(CAPS), "valid_folds": accounting["valid_folds"], "omitted_fold_cells": len(omitted),
        "contract_sha256": digest(contract), "schedule_sha256": contract["schedule_sha256"],
        "result_commit_sha256": result_files["COMMIT.json"], "run_manifest_sha256": result_files["run_manifest.json"],
        "frozen_receipt_sha256": receipt_sha, "package_commit_sha256": package_commit_sha,
        "result_files_sha256": result_files, "checked_binding_count": len(contract["input_files_sha256"]),
        "models": {"total": model_count, "primary": primary_count, "diagnostic": diagnostic_count,
                   "model_hashes_checked": model_count, "training_transforms_checked": model_count,
                   "normal_equation_residual_checked": model_count, "scores_replayed": score_count,
                   "score_threshold_decisions_checked": score_count,
                   "maximum_normal_equation_absolute_residual": max_residual,
                   "maximum_absolute_score_replay_error": max_score_error,
                   "minimum_absolute_saved_score_distance_from_threshold": min_threshold_margin},
        "streams": stream_counts,
        "metrics": {"primary_pair_rows": stream_counts["primary"]["pair_metrics"],
                    "primary_pooled_rows": stream_counts["primary"]["pooled_metrics"],
                    "primary_per_source_rows": stream_counts["primary"]["per_source_endpoints"],
                    "diagnostic_pair_rows": stream_counts["diagnostic"]["pair_metrics"],
                    "diagnostic_pooled_rows": stream_counts["diagnostic"]["pooled_metrics"],
                    "diagnostic_per_source_rows": stream_counts["diagnostic"]["per_source_endpoints"]},
        "oof_source_endpoints": oof_summary,
        "aggregation_contract": {"unique_oof_recording_rates_within_arm_source": True,
                                 "equal_global_component_rates": True, "equal_fold_component_mean": False,
                                 "fold_pair_macro_hierarchy": True, "pool_across_held_source_arms": False,
                                 "pooled_out_of_fold_auc": False},
        "model_fitting_performed": False, "scorer_predict_called": False,
        "optimizer_solution_independently_reproduced": False, "training_stationarity_checked": True,
        "metrics_independently_recomputed": True, "auditor_sha256": auditor_sha_at_start,
        "audit_protocol_sha256": protocol_sha_at_start, "audit_runtime": audit_runtime,
    }


def validate_new_receipt_path(path: Path) -> None:
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent and not path.exists() and not path.is_symlink(),
            "New canonical audit receipt path required")


def publish_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path = Path(path)
    validate_new_receipt_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.pending.{os.getpid()}"
    require(not pending.exists() and not pending.is_symlink(), "Stale audit pending file refused")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            stream.write(canonical(receipt).decode() + "\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(pending, path)
    finally:
        if pending.exists(): pending.unlink()
    require(read_json(path) == receipt and sha256_file(path) == hashlib.sha256(canonical(receipt) + b"\n").hexdigest(),
            "Published audit receipt verification failed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument("--package-dir", type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument("--frozen-receipt", type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument("--frozen-receipt-sha256", required=True)
    parser.add_argument("--audit-output", type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument("--synthetic-test-only", action="store_true")
    parser.add_argument("--parent-authorized-completed-audit", action="store_true",
                        help="Explicit guard: audit only a parent-authorized, COMMIT-complete publication")
    args = parser.parse_args(argv)
    require(args.parent_authorized_completed_audit, "Explicit parent authorization is required before a completed-run audit")
    validate_new_receipt_path(args.audit_output)
    try:
        args.audit_output.relative_to(args.result_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Audit receipt must be outside the immutable result publication")
    receipt = audit(args.result_dir, args.package_dir, args.frozen_receipt,
                    args.frozen_receipt_sha256, args.synthetic_test_only)
    publish_receipt(args.audit_output, receipt)
    print(json.dumps({"status": "passed", "audit_output": str(args.audit_output),
                      "audit_sha256": sha256_file(args.audit_output), "models": receipt["models"]["total"],
                      "predictions": receipt["models"]["scores_replayed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
