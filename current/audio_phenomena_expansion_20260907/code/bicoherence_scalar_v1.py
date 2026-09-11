"""Pure reducer for full bicoherence_audio_v1 JSON metadata, not descriptors.

No waveform/array input, imputation, admission decision, or significance test.
Pass extract(...)["metadata"] (or its saved JSON), plus explicit crop coordinates
in the standardized 16 kHz recording. Coordinates are checked for consistency,
not independently verified against audio. The caller owns provenance/hashes.

NaN belongs only in the extractor's NumPy arrays; JSON metadata missingness is
None. Invalid schemas raise ValueError/TypeError/KeyError; nonfinite numeric values
raise FloatingPointError. Exceptions are never converted to scientific missing.
The extractor raises on arithmetic failure and supplies no valid metadata then.
"""
from __future__ import annotations

import math
from numbers import Real
from statistics import median

VERSION = "bicoherence_scalar_v1"
TARGET = (32, 48, 80)
POOL_SAMPLES = 64000
_GRID = [[a, b, a + b] for a in range(8, 193, 8)
         for b in range(a, 193, 8) if a + b <= 256]
_MISSING = {"zero_energy", "missing_triad_energy", "missing_triad_product_energy"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be a JSON integer")
    _require(value >= 0, f"{name} must be nonnegative")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite {name}; not scientific missingness")
    return result


def _score(value, name):
    result = _finite(value, name)
    _require(0 <= result <= 1, f"{name} outside [0, 1]")
    return result


def _target(cell, pool_status):
    _require(type(cell["eligible"]) is bool and
             type(cell["energy_floor_passed"]) is bool, "masks must be bool")
    primitive = cell["primitive"]
    _require(primitive["version"] == "bicoherence_primitive_v2" and
             primitive["realizations"] == 247 and
             primitive["frequency_bins"] == list(TARGET), "primitive identity mismatch")
    status, raw = primitive["status"], primitive["squared_bicoherence"]
    if status == "ok":
        raw = _score(raw, "raw target b2")
    else:
        _require(status in _MISSING and raw is None, "invalid primitive missing/failure status")
    fractions = cell["energy_fractions"]
    _require(isinstance(fractions, list) and len(fractions) == 3, "invalid fractions")
    if pool_status == "ok":
        fractions = [_finite(v, "energy fraction") for v in fractions]
        _require(all(0 <= v <= 1 for v in fractions), "invalid energy fraction")
        floor = all(v >= 1e-6 for v in fractions)
    else:
        _require(fractions == [None, None, None], "zero-energy pool fractions must be missing")
        _require(status == "zero_energy", "zero-energy pool primitive mismatch")
        floor = False
    _require(cell["energy_floor_passed"] == floor, "energy floor mask mismatch")
    eligible = floor and status == "ok"
    _require(cell["eligible"] == eligible, "target eligibility mismatch")
    expected_status = "ok" if eligible else status if status != "ok" else "below_energy_fraction_floor"
    _require(cell["status"] == expected_status, "target status mismatch")
    if eligible:
        value = _score(cell["squared_bicoherence"], "target b2")
        _require(value == raw, "raw/masked target disagreement")
        return value
    _require(cell["squared_bicoherence"] is None, "ineligible JSON target must be None")
    return None


def reduce_metadata(metadata: dict, *, crop: dict) -> dict:
    """Reduce full extractor metadata; crop requires start/stop/source length.

    crop keys: start_sample, stop_sample_exclusive, source_resampled_samples.
    Pools and tail are relative to that crop, with no reordering or padding.
    Extra extractor metadata is allowed; compact producer descriptors are not.
    Only the fixed target's numerical contents are validated, not all grid scores.
    """
    if not isinstance(metadata, dict) or not isinstance(crop, dict):
        raise TypeError("metadata and crop must be dictionaries")
    constants = {"version": "bicoherence_audio_v1", "primitive_version": "bicoherence_primitive_v2",
                 "sample_rate_hz": 16000, "pool_samples": POOL_SAMPLES,
                 "pool_duration_seconds": 4.0, "n_fft": 1024, "hop_samples": 256,
                 "frames_per_pool": 247, "grid_cell_count": 228,
                 "energy_fraction_min_inclusive": 1e-6, "window": "periodic_hann",
                 "tail_policy": "discard_no_padding",
                 "pool_boundary_policy": "no_frame_crosses_pool_boundary"}
    for key, expected in constants.items():
        _require(metadata.get(key) == expected, f"extractor metadata mismatch: {key}")
    _require(set(crop) == {"start_sample", "stop_sample_exclusive", "source_resampled_samples"},
             "crop keys mismatch")
    coordinates = {key: _integer(value, key) for key, value in crop.items()}
    start, stop, source_n = (coordinates[key] for key in
                            ("start_sample", "stop_sample_exclusive", "source_resampled_samples"))
    _require(start <= stop <= source_n, "crop outside standardized source")
    n = _integer(metadata["input_samples"], "input_samples")
    _require(stop - start == n, "crop length differs from extractor input")
    count, tail = divmod(n, POOL_SAMPLES)
    for key, expected in {"pool_count": count, "analyzed_samples": count * POOL_SAMPLES,
                          "discarded_tail_samples": tail, "tail_start_sample": count * POOL_SAMPLES}.items():
        _require(_integer(metadata[key], key) == expected, f"pool/tail mismatch: {key}")
    _require(metadata["status"] == ("ok" if count else "insufficient_support"),
             "invalid extractor status; failures cannot become missing")
    construction = metadata.get("construction_status")
    _require(construction in (None, "ok", "unsupported_zero_background_rms"),
             "invalid construction status")
    pools = metadata["pools"]
    _require(isinstance(pools, list) and len(pools) == count, "pool count mismatch")
    records, values, mask = [], [], []
    for index, pool in enumerate(pools):
        for key, expected in {"pool_index": index, "start_sample": index * POOL_SAMPLES,
                              "stop_sample_exclusive": (index + 1) * POOL_SAMPLES,
                              "coefficient_rows": 247}.items():
            _require(_integer(pool[key], key) == expected, f"pool identity/order mismatch: {key}")
        status = pool["status"]
        _require(status in ("ok", "zero_amplitude", "zero_non_dc_energy"), "invalid pool status")
        energy = _finite(pool["total_non_dc_coefficient_energy"], "pool energy")
        _require((energy > 0 if status == "ok" else energy == 0), "pool energy/status mismatch")
        _require(type(pool["zero_amplitude"]) is bool and
                 pool["zero_amplitude"] == (status == "zero_amplitude"), "zero-amplitude mismatch")
        cells = pool["cells"]
        _require(isinstance(cells, list) and [c["frequency_bins"] for c in cells] == _GRID,
                 "grid order/identity mismatch")
        cell = cells[_GRID.index(list(TARGET))]
        value = _target(cell, status)
        if construction == "unsupported_zero_background_rms":
            _require(status == "zero_amplitude", "zero construction must have zero pools")
        values.append(value)
        mask.append(value is not None)
        records.append({"pool_index": index, "start_sample": index * POOL_SAMPLES,
                        "stop_sample_exclusive": (index + 1) * POOL_SAMPLES,
                        "source_start_sample": start + index * POOL_SAMPLES,
                        "source_stop_sample_exclusive": start + (index + 1) * POOL_SAMPLES,
                        "pool_status": status, "target_status": cell["status"],
                        "eligible": value is not None, "squared_bicoherence": value})
    available = [v for v in values if v is not None]
    return {"version": VERSION, "target_frequency_bins": list(TARGET),
            "target_frequency_hz": [500, 750, 1250], "crop": coordinates,
            "input_samples": n, "pool_count": count, "analyzed_samples": count * POOL_SAMPLES,
            "discarded_tail_samples": tail, "eligible_pool_count": len(available),
            "missing_pool_count": count - len(available), "eligibility_mask": mask,
            "status": "ok" if available else "no_eligible_target_pools" if count else "no_complete_pools",
            "median_squared_bicoherence": float(median(available)) if available else None,
            "pools": records, "construction_status": construction,
            "null_calibrated": False, "significance_inferred": False,
            "external_validation_passed": False, "classifier_admitted": False}


def compare_conditions(first_metadata: dict, second_metadata: dict, *, first_crop: dict,
                       second_crop: dict) -> dict:
    """First minus second: operational medians and distinct paired-pool mean.

    Both reductions are validated before comparison. Conditions must have exactly
    the same standardized crop coordinates and support. Names/lineage are caller
    responsibilities; use closed as first and independent as second for the assay.
    """
    first = reduce_metadata(first_metadata, crop=first_crop)
    second = reduce_metadata(second_metadata, crop=second_crop)
    _require(first["crop"] == second["crop"], "conditions have different crops")
    a, b = first["median_squared_bicoherence"], second["median_squared_bicoherence"]
    paired_mask, differences = [], []
    _require(first["pool_count"] == second["pool_count"], "condition pool counts differ")
    for left, right in zip(first["pools"], second["pools"]):
        paired = left["eligible"] and right["eligible"]
        paired_mask.append(paired)
        differences.append(left["squared_bicoherence"] - right["squared_bicoherence"] if paired else None)
    finite = [d for d in differences if d is not None]
    return {"version": VERSION, "direction": "first_minus_second", "first": first, "second": second,
            "operational_median_difference": a - b if a is not None and b is not None else None,
            "operational_status": "ok" if a is not None and b is not None else "missing_condition_scalar",
            "paired_pool_mean_difference": math.fsum(finite) / len(finite) if finite else None,
            "paired_status": "ok" if finite else "no_common_eligible_pools",
            "paired_pool_count": len(finite), "total_pool_count": first["pool_count"],
            "paired_pool_mask": paired_mask, "paired_pool_differences": differences}
