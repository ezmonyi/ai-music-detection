#!/usr/bin/env python3
"""Preserve exact fixture evidence that S16/D3/R3/P6 match prior code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from expanded_feature_definitions import (
    D_FEATURES, P_FEATURES, R_FEATURES, S16_FEATURES, allinone_boundaries,
    dynamics_features, load_beats, phrase_features, read_audio,
    read_spectral_audio, rhythm_features, spectral_families,
)


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compare(old: dict[str, str], new: dict[str, float], names: tuple[str, ...], prefix: str = "") -> dict[str, float]:
    result = {}
    for name in names:
        a, b = float(old[prefix+name]), float(new[name])
        if math.isfinite(a) and math.isfinite(b):
            result[name] = abs(a-b)
        elif math.isnan(a) and math.isnan(b):
            result[name] = 0.0
        else:
            result[name] = float("inf")
    return result


def nan_pattern_equal(old: dict[str, str], new: dict[str, float], names: tuple[str, ...], prefix: str = "") -> bool:
    return all(math.isnan(float(old[prefix + name])) == math.isnan(float(new[name])) for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heuristic-root", type=Path, required=True)
    parser.add_argument("--spectral-csv", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--metadata-csv", type=Path,
                        help="Optional metadata used to stratify 10s fixtures by source")
    args = parser.parse_args()
    with np.load(args.bias, allow_pickle=False) as payload:
        bias = np.asarray(payload["selected_bias_db"], dtype=float)
    all_heuristic_rows = rows(args.heuristic_root/"results/heuristic_features.csv")
    metadata_lookup = {}
    if args.metadata_csv:
        metadata_lookup = {
            (row.get("item_id") or row.get("track") or row.get("id")): row
            for row in rows(args.metadata_csv)
        }
    # Source-order slicing accidentally covered only ACE-Step. Round-robin by
    # source makes the cheap fixture cover Human plus AIME native-rate strata.
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in all_heuristic_rows:
        by_source.setdefault(row.get("source", "unknown"), []).append(row)
    heuristic_rows = []
    offset = 0
    while len(heuristic_rows) < args.samples:
        added = False
        for source in sorted(by_source):
            candidates = by_source[source]
            if offset < len(candidates):
                heuristic_rows.append(candidates[offset]); added = True
                if len(heuristic_rows) == args.samples:
                    break
        if not added:
            break
        offset += 1
    spectral_lookup = {row["track"]: row for row in rows(args.spectral_csv)}
    details10 = []
    for row in heuristic_rows:
        item_id = row["track"]; inference = args.heuristic_root/"inference"
        stems = {stem: read_audio(inference/"demix/htdemucs"/item_id/f"{stem}.wav", 10, strict=True) for stem in ("bass", "drums", "other")}
        d = dynamics_features(stems["bass"]+stems["drums"]+stems["other"])
        beat_times, beat_numbers = load_beats(inference/"beats"/f"{item_id}.beats", 10)
        r = rhythm_features(beat_times)
        p = phrase_features(allinone_boundaries(inference/"structure"/f"{item_id}.json", 10), beat_times[beat_numbers == 1])
        s16, _ = spectral_families(read_spectral_audio(inference/"demix/htdemucs"/item_id/"vocals.wav", 10), bias)
        metadata = metadata_lookup.get(item_id, {})
        details10.append({
            "item_id": item_id, "source_id": row.get("source", ""),
            "native_sample_rate_hz": metadata.get("native_sample_rate_hz", ""),
            "s16_abs_delta": compare(spectral_lookup[item_id], s16, S16_FEATURES, "spectral__"),
            "d3_abs_delta": compare(row, d, D_FEATURES), "r3_abs_delta": compare(row, r, R_FEATURES),
            "p6_abs_delta": compare(row, p, P_FEATURES),
            "s16_nan_pattern_equal": nan_pattern_equal(spectral_lookup[item_id], s16, S16_FEATURES, "spectral__"),
            "d3_nan_pattern_equal": nan_pattern_equal(row, d, D_FEATURES),
            "r3_nan_pattern_equal": nan_pattern_equal(row, r, R_FEATURES),
            "p6_nan_pattern_equal": nan_pattern_equal(row, p, P_FEATURES),
        })
    detector_rows = rows(args.detector_root/"detector_results/new_features.csv")
    detector_rows = [row for row in detector_rows if any(row[name] not in {"", "nan"} and math.isfinite(float(row[name])) for name in P_FEATURES)]
    detector_by_source: dict[str, list[dict[str, str]]] = {}
    for row in detector_rows:
        detector_by_source.setdefault(row.get("source", "unknown"), []).append(row)
    selected_detector_rows = []
    offset = 0
    while len(selected_detector_rows) < args.samples:
        added = False
        for source in sorted(detector_by_source):
            candidates = detector_by_source[source]
            if offset < len(candidates):
                selected_detector_rows.append(candidates[offset]); added = True
                if len(selected_detector_rows) == args.samples:
                    break
        if not added:
            break
        offset += 1
    details30 = []
    for row in selected_detector_rows:
        item_id = row["track"]; inference = args.detector_root/"detector_inference"
        stems = {stem: read_audio(inference/"demix/htdemucs"/item_id/f"{stem}.wav", 30, strict=True) for stem in ("bass", "drums", "other")}
        d = dynamics_features(stems["bass"]+stems["drums"]+stems["other"])
        beat_times, beat_numbers = load_beats(inference/"beats"/f"{item_id}.beats", 30)
        r = rhythm_features(beat_times)
        p = phrase_features(allinone_boundaries(inference/"structure"/f"{item_id}.json", 30), beat_times[beat_numbers == 1])
        details30.append({"item_id": item_id, "source_id": row.get("source", ""),
                          "d3_abs_delta": compare(row, d, D_FEATURES),
                          "r3_abs_delta": compare(row, r, R_FEATURES), "p6_abs_delta": compare(row, p, P_FEATURES),
                          "d3_nan_pattern_equal": nan_pattern_equal(row, d, D_FEATURES),
                          "r3_nan_pattern_equal": nan_pattern_equal(row, r, R_FEATURES),
                          "p6_nan_pattern_equal": nan_pattern_equal(row, p, P_FEATURES)})
    def maximum(group: list[dict[str, object]], key: str) -> float | None:
        values = [v for item in group for v in item.get(key, {}).values()]  # type: ignore[union-attr]
        return max(values) if values else None
    payload = {
        "fixture_10s_ids": [x["item_id"] for x in details10],
        "fixture_10s_sources": [x["source_id"] for x in details10],
        "fixture_10s_native_sample_rates_hz": [x["native_sample_rate_hz"] for x in details10],
        "fixture_30s_ids": [x["item_id"] for x in details30],
        "fixture_30s_sources": [x["source_id"] for x in details30],
        "reference_hashes": {
            "heuristic_features_csv": file_hash(args.heuristic_root/"results/heuristic_features.csv"),
            "spectral_features_csv": file_hash(args.spectral_csv),
            "detector_features_csv": file_hash(args.detector_root/"detector_results/new_features.csv"),
            "bias_npz": file_hash(args.bias),
            "definition_code": file_hash(Path(__file__).with_name("expanded_feature_definitions.py")),
        },
        "max_abs_delta": {
            "10s_s16": maximum(details10, "s16_abs_delta"), "10s_d3": maximum(details10, "d3_abs_delta"),
            "10s_r3": maximum(details10, "r3_abs_delta"), "10s_p6": maximum(details10, "p6_abs_delta"),
            "30s_d3": maximum(details30, "d3_abs_delta"), "30s_r3": maximum(details30, "r3_abs_delta"),
            "30s_p6": maximum(details30, "p6_abs_delta"),
        },
        "nan_pattern_mismatch_ids": {
            key: [item["item_id"] for item in group if not item[f"{key}_nan_pattern_equal"]]
            for key, group in (
                ("s16", details10), ("d3", details10), ("r3", details10), ("p6", details10),
            )
        } | {
            f"30s_{key}": [item["item_id"] for item in details30 if not item[f"{key}_nan_pattern_equal"]]
            for key in ("d3", "r3", "p6")
        },
        "details_10s": details10, "details_30s": details30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if not key.startswith("details")}, indent=2))


if __name__ == "__main__":
    main()
