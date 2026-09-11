#!/usr/bin/env python3
"""Compare every overlapping legacy S16 row with the expanded S16 table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


OLD_TO_NEW = {
    "spectral__tilt_1_5k_db_oct": "s16__tilt_1_5k_db_oct",
    "spectral__hf_tilt_5_16k_db_oct": "s16__hf_tilt_5_16k_db_oct",
    "spectral__hf_ratio_5_16_db": "s16__hf_ratio_5_16_db",
    "spectral__air_ratio_12_20_db": "s16__air_ratio_12_20_db",
    "spectral__sibilance_ratio_5_10_db": "s16__sibilance_ratio_5_10_db",
    "spectral__hf_flatness": "s16__hf_flatness",
    "spectral__hf_entropy": "s16__hf_entropy",
    "spectral__hf_crest_db": "s16__hf_crest_db",
    "spectral__fakeprint_peak_density": "s16__fakeprint_peak_density",
    "spectral__fakeprint_periodicity": "s16__fakeprint_periodicity",
    "spectral__hf_flux": "s16__hf_flux",
    "spectral__hf_frame_similarity": "s16__hf_frame_similarity",
    "spectral__hf_power_sd_db": "s16__hf_power_sd_db",
    "spectral__hf_mod_4_12_share": "s16__hf_mod_4_12_share",
    "spectral__sibilance_contrast_db": "s16__sibilance_contrast_db",
    "spectral__sibilance_burst_rate_hz": "s16__sibilance_burst_rate_hz",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-csv", type=Path, required=True)
    parser.add_argument("--expanded-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    legacy = read_rows(args.legacy_csv)
    expanded_rows = read_rows(args.expanded_csv)
    expanded = {row["item_id"]: row for row in expanded_rows}
    if len(expanded) != len(expanded_rows):
        raise RuntimeError("Expanded table contains duplicate item_id values")

    missing = [row["track"] for row in legacy if row["track"] not in expanded]
    extra = sorted(set(expanded) - {row["track"] for row in legacy})
    per_source: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0,
        "max_abs_delta": {new: 0.0 for new in OLD_TO_NEW.values()},
        "nan_pattern_mismatch_ids": set(),
        "above_atol_ids": set(),
        "exact_float_mismatch_ids": set(),
    })
    global_max = {new: 0.0 for new in OLD_TO_NEW.values()}
    nan_mismatch: set[str] = set()
    above_atol: set[str] = set()
    exact_mismatch: set[str] = set()

    for old in legacy:
        item_id = old["track"]
        if item_id not in expanded:
            continue
        new = expanded[item_id]
        source = new.get("source_id") or "unknown"
        source_result = per_source[source]
        source_result["rows"] = int(source_result["rows"]) + 1
        for old_name, new_name in OLD_TO_NEW.items():
            a, b = float(old[old_name]), float(new[new_name])
            if math.isnan(a) != math.isnan(b):
                nan_mismatch.add(item_id)
                source_result["nan_pattern_mismatch_ids"].add(item_id)  # type: ignore[union-attr]
                continue
            if math.isnan(a):
                continue
            delta = abs(a - b)
            global_max[new_name] = max(global_max[new_name], delta)
            source_max = source_result["max_abs_delta"]  # type: ignore[assignment]
            source_max[new_name] = max(source_max[new_name], delta)
            if a != b:
                exact_mismatch.add(item_id)
                source_result["exact_float_mismatch_ids"].add(item_id)  # type: ignore[union-attr]
            if delta > args.atol:
                above_atol.add(item_id)
                source_result["above_atol_ids"].add(item_id)  # type: ignore[union-attr]

    def finalize(group: dict[str, object]) -> dict[str, object]:
        return {
            **group,
            "nan_pattern_mismatch_ids": sorted(group["nan_pattern_mismatch_ids"]),  # type: ignore[arg-type]
            "above_atol_ids": sorted(group["above_atol_ids"]),  # type: ignore[arg-type]
            "exact_float_mismatch_count": len(group["exact_float_mismatch_ids"]),  # type: ignore[arg-type]
            "exact_float_mismatch_ids": sorted(group["exact_float_mismatch_ids"]),  # type: ignore[arg-type]
        }

    payload = {
        "legacy_rows": len(legacy),
        "expanded_rows": len(expanded_rows),
        "overlap_rows": len(legacy) - len(missing),
        "atol": args.atol,
        "missing_legacy_ids": missing,
        "expanded_only_ids": extra,
        "nan_pattern_mismatch_ids": sorted(nan_mismatch),
        "above_atol_ids": sorted(above_atol),
        "exact_float_mismatch_count": len(exact_mismatch),
        "exact_float_mismatch_ids": sorted(exact_mismatch),
        "max_abs_delta": global_max,
        "per_source": {source: finalize(group) for source, group in sorted(per_source.items())},
        "reference_hashes": {
            "legacy_csv": sha256(args.legacy_csv),
            "expanded_csv": sha256(args.expanded_csv),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"per_source", "exact_float_mismatch_ids", "expanded_only_ids"}}, indent=2))


if __name__ == "__main__":
    main()
