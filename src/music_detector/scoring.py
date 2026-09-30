"""Portable replay of the archived Native30 weighted-ridge decisions.

The historical score is an unbounded identity-link decision value. This module
does not fit models, pick a winning subset, infer provenance, or turn a score
into a probability. A probability requires a separately audited, pinned
calibration artifact. Hash checks establish artifact identity, not scientific
validity of training data; the original training audit remains authoritative.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


# Order is part of the frozen design matrix, copied from evaluate_native30_v3
# and its one-column BC adapter. H describes chroma, not overtone structure.
FAMILIES = {
    "S": ["s8__" + name for name in (
        "tilt_1_5k_db_oct", "hf_tilt_5_7p5k_db_oct", "hf_ratio_5_7p5_db",
        "sibilance_ratio_5_7p5_db", "hf_flatness_5_7p5", "hf_entropy_5_7p5",
        "hf_crest_5_7p5_db", "fakeprint_peak_density_5_7p5_per_khz",
        "fakeprint_periodicity_5_7p5", "hf_flux_5_7p5", "hf_frame_similarity_5_7p5",
        "hf_power_sd_5_7p5_db", "hf_mod_4_12_share_5_7p5",
        "sibilance_contrast_5_7p5_db", "sibilance_burst_rate_5_7p5_hz")],
    "D": ["d__" + name for name in (
        "dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")],
    "R": ["r__" + name for name in ("ibi_cv", "tempo_tv", "tempo_entropy")],
    "P": ["p__" + name for name in (
        "section_duration_cv", "section_duration_entropy", "section_bars_cv",
        "section_bars_offmode_fraction", "section_duration_median", "section_bars_median")],
    "F": ["F_" + name + "_" + region for name in (
        "phase_residual_cvar", "group_delay_iqr_ms", "group_delay_cross_band_iqr_ms")
        for region in ("all", "attack", "sustain", "decay")]
        + ["F_" + name + "_decay_minus_sustain" for name in (
            "phase_residual_cvar", "group_delay_iqr_ms", "group_delay_cross_band_iqr_ms")],
    "H": ["H_" + name for name in (
        "pitch_class_entropy_norm", "pc_token_entropy_norm", "chroma_path_change_median",
        "chroma_path_change_iqr", "pc_path_step_median", "pc_path_large_step_rate")],
    "SC": [f"SC_{lo}_{hi}hz_{metric}_median" for lo, hi in
        ((80, 500), (500, 2000), (2000, 6000))
        for metric in ("abs_iid_db", "side_energy_fraction")],
    "BC": ["BC_b2_500_750_1250hz_center8s_median"],
}
FEATURE_FAMILY = {column: family for family, columns in FAMILIES.items() for column in columns}
MODEL_KEYS = {
    "columns", "medians", "mean", "scale", "coefficients_with_intercept", "ridge",
    "threshold", "feature_mode", "training_rows", "training_source_counts", "weighting",
    "missing_value_policy", "observed_fraction_by_column",
}
SCOPE_KEYS = {
    "winner_selection", "threshold_tuning", "full_cohort_refit",
    "historical_locked_or_pilot_scoring", "M_predictor",
    "audio_or_neural_inference_performed", "source_ranking",
}
PAYLOAD_KEYS = {
    "status", "contract_sha256", "task", "task_sha256", "model", "model_sha256",
    "predictions", "predictions_sha256", "metrics", "training_numerical_audit", *SCOPE_KEYS,
}
MODES = {"values_plus_missing", "median_only", "missingness_only"}


def canonical_hash(value: Any) -> str:
    """Match materialize_native30_new1695_v1.value_hash, including final LF."""
    try:
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Artifact must contain finite JSON values") from error
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _families(selection: Sequence[str] | str) -> list[str]:
    selected = selection.split("+") if isinstance(selection, str) else list(selection)
    _require(bool(selected) and all(isinstance(x, str) and x in FAMILIES for x in selected),
             "Select known Native30 families; long-context M is not a Native30 predictor")
    _require(len(selected) == len(set(selected)), "Duplicate family selection")
    _require(selected == [key for key in FAMILIES if key in selected], "Families must use canonical S,D,R,P,F,H,SC,BC order")
    return selected


def validate_receipt(receipt: Mapping[str, Any], families: Sequence[str] | str | None = None,
                     *, allow_research: bool = False) -> dict[str, Any]:
    """Check archived receipt/schema/parameter identities without re-fitting.

    This does not repeat the original training-set numerical audit. Load from a
    trusted release using ``load_model_receipt`` to bind external file identity.
    """
    _require(isinstance(receipt, Mapping) and set(receipt) == {"payload", "receipt_sha256"}, "Expected archived model receipt envelope")
    payload = receipt["payload"]
    _require(isinstance(payload, dict) and set(payload) == PAYLOAD_KEYS, "Unexpected Native30 receipt payload schema")
    _require(receipt["receipt_sha256"] == canonical_hash(payload), "Receipt payload hash mismatch")
    _require(payload["status"] == "fold_only_model_and_predictions_not_selected"
             and all(payload[key] is False for key in SCOPE_KEYS), "Receipt scientific scope changed")
    _require(_digest(payload["contract_sha256"]), "Invalid parent contract hash")
    for field in ("model", "task", "predictions"):
        _require(payload[field + "_sha256"] == canonical_hash(payload[field]), field + " hash mismatch")
    task, model = payload["task"], payload["model"]
    _require(isinstance(task, dict) and isinstance(model, dict) and set(model) == MODEL_KEYS,
             "Unexpected Native30 task/model schema")
    chosen = _families(task.get("combination", ""))
    if families is not None:
        _require(_families(families) == chosen, "Selected families do not match fitted model; do not drop coefficients")
    mode = model["feature_mode"]
    _require(mode in MODES and task.get("feature_mode") == mode, "Unknown or mismatched feature mode")
    research = "BC" in chosen or mode != "values_plus_missing"
    _require(allow_research or not research, "BC and diagnostic feature modes require explicit research opt-in")
    columns = [column for family in chosen for column in FAMILIES[family]]
    _require(model["columns"] == columns, "Model must contain full frozen families in canonical order")
    count = len(columns)
    dimensions = count * (2 if mode == "values_plus_missing" else 1)
    for key, size in (("medians", count), ("mean", dimensions), ("scale", dimensions),
                      ("coefficients_with_intercept", dimensions + 1)):
        values = model[key]
        _require(isinstance(values, list) and len(values) == size and all(_number(x) for x in values),
                 "Invalid frozen parameter: " + key)
    _require(all(x > 0 for x in model["scale"]), "Frozen scales must be positive")
    _require(_number(model["ridge"]) and model["ridge"] == 10.0 and
             _number(model["threshold"]) and model["threshold"] == 0.5, "Historical ridge/threshold changed")
    _require(type(model["training_rows"]) is int and model["training_rows"] > 0, "Invalid training row count")
    counts = model["training_source_counts"]
    _require(isinstance(counts, dict) and bool(counts) and
             all(isinstance(k, str) and (k.startswith("0:") or k.startswith("1:"))
                 and type(v) is int and v > 0 for k, v in counts.items()) and
             sum(counts.values()) == model["training_rows"] and
             {key.split(":", 1)[0] for key in counts} == {"0", "1"}, "Invalid training source counts")
    _require(model["weighting"] == "classes equal; sources equal within class; groups equal within source; samples equal within group"
             and model["missing_value_policy"] == "training-only median imputation plus missingness indicators", "Frozen weighting/imputation policy changed")
    observed = model["observed_fraction_by_column"]
    _require(isinstance(observed, dict) and set(observed) == set(columns) and
             all(_number(v) and 0 <= v <= 1 for v in observed.values()), "Invalid observed fractions")
    return {"payload": payload, "model": model, "families": chosen, "research_only": research}


def load_model_receipt(path: str | Path, *, expected_sha256: str,
                       families: Sequence[str] | str | None = None,
                       allow_research: bool = False) -> dict[str, Any]:
    """Read a hash-pinned historical JSON file. No pickle or code is loaded."""
    raw = Path(path).read_bytes()
    _require(_digest(expected_sha256) and hashlib.sha256(raw).hexdigest() == expected_sha256,
             "Model file differs from trusted release hash")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    receipt = json.loads(raw, object_pairs_hook=unique_object)
    _require(canonical_hash(receipt) == expected_sha256, "Expected canonical historical receipt bytes")
    validate_receipt(receipt, families, allow_research=allow_research)
    return receipt


def _calibrated_probability(score: float, calibration: Mapping[str, Any], expected_sha256: str | None,
                            model_sha256: str, families: list[str]) -> tuple[float, str]:
    """Consume a separately audited calibrator; metadata is not a new audit.

    The caller must supply its trusted release hash separately. The evidence
    fields identify the development-only group-disjoint validation report that
    release approval verified. Merely fitting a sigmoid is not calibration.
    """
    _require(_digest(expected_sha256) and canonical_hash(calibration) == expected_sha256,
             "Calibration needs its independently supplied trusted release hash")
    keys = {"schema_version", "method", "model_sha256", "families", "feature_mode",
            "slope", "intercept", "validation"}
    _require(set(calibration) == keys and calibration["schema_version"] == "music-detector-calibration-v1"
             and calibration["method"] == "platt", "Unsupported calibration artifact")
    _require(calibration["model_sha256"] == model_sha256 and calibration["families"] == families
             and calibration["feature_mode"] == "values_plus_missing", "Calibration belongs to a different model")
    _require(_number(calibration["slope"]) and calibration["slope"] > 0 and _number(calibration["intercept"]),
             "Invalid monotonic Platt parameters")
    validation = calibration["validation"]
    required = {"status", "report_sha256", "fit_role", "validation_role", "group_disjoint",
                "both_classes", "locked_labels_used", "brier_score", "ece", "rows"}
    _require(isinstance(validation, dict) and set(validation) == required
             and validation["status"] == "accepted" and _digest(validation["report_sha256"])
             and validation["fit_role"] == validation["validation_role"] == "development"
             and validation["group_disjoint"] is True and validation["both_classes"] is True
             and validation["locked_labels_used"] is False, "Calibration lacks accepted development-only validation")
    _require(type(validation["rows"]) is int and validation["rows"] >= 2 and
             all(_number(validation[key]) and 0 <= validation[key] <= 1 for key in ("brier_score", "ece")),
             "Calibration validation metrics missing")
    linear = calibration["slope"] * score + calibration["intercept"]
    _require(math.isfinite(linear), "Calibration overflow")
    probability = 1 / (1 + math.exp(-linear)) if linear >= 0 else math.exp(linear) / (1 + math.exp(linear))
    return probability, expected_sha256


def score_features(receipt: Mapping[str, Any], features: Mapping[str, Any], *,
                   families: Sequence[str] | str | None = None, allow_research: bool = False,
                   calibration: Mapping[str, Any] | None = None,
                   expected_calibration_sha256: str | None = None) -> dict[str, Any]:
    """Replay a selected-family model and expose additive decision evidence.

    Supply exactly the model's named feature columns. An absent column is a
    pipeline failure; explicit null/NaN is an observed unavailable measurement.
    Values and missing indicators are distinct explanatory terms. Contributions
    are associations in the fitted decision, not causal evidence of authorship.
    Scalar fsum can differ from historical NumPy/BLAS by floating-point roundoff.
    """
    checked = validate_receipt(receipt, families, allow_research=allow_research)
    model, selected = checked["model"], checked["families"]
    columns = model["columns"]
    _require(isinstance(features, Mapping) and set(features) == set(columns),
             "Features must exactly match model columns (explicit null for unavailable measurements)")
    values, missing = [], []
    for column, median in zip(columns, model["medians"]):
        value = features[column]
        absent = value is None or (type(value) is float and math.isnan(value))
        _require(absent or _number(value), "Invalid measured value: " + column)
        values.append(float(median if absent else value))
        missing.append(absent)
    mode = model["feature_mode"]
    terms = ([] if mode == "missingness_only" else [(column, "value", value) for column, value in zip(columns, values)])
    if mode != "median_only":
        terms += [(column, "missingness", float(absent)) for column, absent in zip(columns, missing)]
    contributions = []
    for (column, kind, value), mean, scale, coefficient in zip(
            terms, model["mean"], model["scale"], model["coefficients_with_intercept"][1:]):
        standardized = (value - mean) / scale
        contribution = standardized * coefficient
        _require(math.isfinite(standardized) and math.isfinite(contribution), "Nonfinite transformed feature: " + column)
        contributions.append({"feature": column, "family": FEATURE_FAMILY[column], "kind": kind,
            "value_after_imputation": value, "standardized_value": standardized,
            "coefficient": coefficient, "contribution": contribution})
    intercept = model["coefficients_with_intercept"][0]
    raw = math.fsum([intercept, *(item["contribution"] for item in contributions)])
    _require(math.isfinite(raw), "Nonfinite raw score")
    family_totals = {family: math.fsum(item["contribution"] for item in contributions if item["family"] == family)
                     for family in selected}
    result = {"raw_score": raw, "score_kind": "unbounded_identity_decision_value",
        "historical_threshold": model["threshold"], "predicted_label": int(raw >= model["threshold"]),
        "families": selected, "feature_mode": mode, "research_only": checked["research_only"],
        "model_sha256": checked["payload"]["model_sha256"], "receipt_sha256": receipt["receipt_sha256"],
        "intercept": intercept, "contributions": contributions, "family_contributions": family_totals,
        "missing_features": [column for column, absent in zip(columns, missing) if absent],
        "missing_count": sum(missing), "feature_count": len(columns),
        "ai_probability": None, "calibration_sha256": None,
        "explanation_scope": "additive model associations; not causal proof of AI authorship",
        "verification_scope": "receipt identities and portable scoring; original training audit not rerun"}
    if calibration is not None:
        _require(not checked["research_only"], "Research-only BC/diagnostic scores cannot supply deployed probability")
        result["ai_probability"], result["calibration_sha256"] = _calibrated_probability(
            raw, calibration, expected_calibration_sha256, result["model_sha256"], selected)
    else:
        _require(expected_calibration_sha256 is None, "Calibration hash supplied without calibration artifact")
    return result
