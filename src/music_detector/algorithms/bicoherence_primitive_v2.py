"""Coefficient-domain pooled squared bicoherence; no music/admission pipeline.

Rows are supplied coefficient realizations, not asserted independent samples.
The statistic is amplitude-weighted and cannot establish nonlinear causation.
Raw sufficient sums use binary scaling: ``mantissa * 2**exponent2``. This
representation remains finite even when the raw value cannot fit in float64.
No STFT, frequency-grid search, null calibration or classifier is provided.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "bicoherence_primitive_v2"
_ROUNDING_TOLERANCE = 64 * np.finfo(np.float64).eps


def _binary_scaled(value: float, factors: tuple[float, ...]) -> dict:
    """Represent a finite value times positive finite factors without overflow."""
    mantissa, exponent = math.frexp(value)
    if mantissa == 0:
        return {"mantissa": 0.0, "exponent2": 0}
    for factor in factors:
        part, shift = math.frexp(factor)
        mantissa *= part
        mantissa, renormalize = math.frexp(mantissa)
        exponent += shift + renormalize
    return {"mantissa": mantissa, "exponent2": exponent}


def _checked_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    product = left * right
    if np.any((left != 0) & (right != 0) & (product == 0)):
        raise FloatingPointError("unsupported_underflow: nonzero coefficient product")
    return product


def _energy(values: np.ndarray) -> float:
    magnitude = np.abs(values)
    power = magnitude * magnitude
    if np.any((magnitude != 0) & (power == 0)):
        raise FloatingPointError("unsupported_underflow: nonzero squared magnitude")
    total = float(np.sum(power))
    if not math.isfinite(total):
        raise FloatingPointError("nonfinite normalized energy sum")
    return total


def estimate(spectra, pair=(4, 7)) -> dict:
    """Return pooled b² and sufficient sums for one positive-frequency triad.

    Numeric inputs are converted to complex128; strings, objects and booleans
    are rejected. Zero/missing energy returns missing b², never a zero score.
    Invalid inputs raise ValueError/TypeError. Unsupported arithmetic range
    raises FloatingPointError instead of silently reporting zero coherence.

    ``normalized_sums`` refers to independently scaled coefficient columns.
    ``raw_sums`` represents the same sums in original coefficient units, with
    each scalar encoded as a binary mantissa/exponent pair. The factors are
    also exposed in ``coefficient_scales`` for independent reconstruction.
    Any defined biphase is descriptive only, irrespective of coherence value;
    exact zero resultant has undefined biphase. No significance is inferred.
    """
    source = np.asarray(spectra)
    if source.dtype.kind not in "iufc":
        raise TypeError("spectra must contain numeric real or complex coefficients")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        a = np.asarray(source, dtype=np.complex128)
    if np.any((source != 0) & (a == 0)):
        raise FloatingPointError("unsupported_underflow: coefficient dtype conversion")
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("at least two finite coefficient realizations in [K,bins] required")
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise TypeError("pair must be a two-item tuple/list of integer bin indices")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in pair):
        raise TypeError("pair indices must be integers, not bools or floats")
    f1, f2 = map(int, pair)
    if not (0 < f1 <= f2 and f1 + f2 < a.shape[1]):
        raise ValueError("invalid positive bifrequency")
    columns = a[:, [f1, f2, f1 + f2]]
    # Component maxima avoid overflow in abs(complex(max_float,max_float)).
    scales = np.maximum(np.max(np.abs(columns.real), axis=0),
                        np.max(np.abs(columns.imag), axis=0))
    result = {"version": VERSION, "realizations": int(a.shape[0]),
              "frequency_bins": [f1, f2, f1 + f2], "status": "ok",
              "squared_bicoherence": None, "biphase_radians": None,
              "biphase_status": "missing_triad_energy",
              "biphase_interpretation": "descriptive_only_no_significance",
              "coefficient_scales": scales.tolist(), "normalized_sums": None,
              "raw_sums": None,
              "raw_sum_encoding": "value = mantissa * 2**exponent2",
              "external_validation_passed": False, "classifier_admitted": False}
    if np.any(scales == 0):
        result["status"] = "zero_energy" if not np.any(a) else "missing_triad_energy"
        return result
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        # Divide components directly: complex division can overflow internally
        # when its real denominator is subnormal or close to max_float.
        normalized = np.empty_like(columns)
        normalized.real = columns.real / scales
        normalized.imag = columns.imag / scales
        if np.any((columns != 0) & (normalized == 0)):
            raise FloatingPointError("unsupported_underflow: coefficient normalization")
        u = _checked_product(normalized[:, 0], normalized[:, 1])
        v = normalized[:, 2]
        u_energy, v_energy = _energy(u), _energy(v)
        terms = _checked_product(u, np.conj(v))
        triple_sum = complex(np.sum(terms))
        if not (math.isfinite(triple_sum.real) and math.isfinite(triple_sum.imag)):
            raise FloatingPointError("nonfinite normalized triple sum")
        sums = {"product_energy_sum": u_energy, "sum_frequency_energy_sum": v_energy,
                "triple_sum_real": triple_sum.real, "triple_sum_imag": triple_sum.imag}
        s1, s2, s3 = map(float, scales)
        result["normalized_sums"] = sums
        result["raw_sums"] = {
            "product_energy_sum": _binary_scaled(u_energy, (s1, s1, s2, s2)),
            "sum_frequency_energy_sum": _binary_scaled(v_energy, (s3, s3)),
            "triple_sum_real": _binary_scaled(triple_sum.real, (s1, s2, s3)),
            "triple_sum_imag": _binary_scaled(triple_sum.imag, (s1, s2, s3))}
        if u_energy == 0 or v_energy == 0:
            result["status"] = "missing_triad_product_energy"
            return result
        # Separate square roots avoid overflow/underflow in the denominator
        # product. Normalized sums stay available for independent replay.
        coherence_magnitude = abs(triple_sum) / math.sqrt(u_energy) / math.sqrt(v_energy)
        squared = coherence_magnitude * coherence_magnitude
        if triple_sum != 0 and squared == 0:
            raise FloatingPointError("unsupported_underflow: nonzero squared coherence")
        if not math.isfinite(squared) or squared < -_ROUNDING_TOLERANCE or squared > 1 + _ROUNDING_TOLERANCE:
            raise FloatingPointError("normalized squared coherence violates finite Cauchy-Schwarz bound")
        result["squared_bicoherence"] = min(1.0, max(0.0, squared))
        if triple_sum == 0:
            result["biphase_status"] = "undefined_zero_resultant"
        else:
            result["biphase_radians"] = math.atan2(triple_sum.imag, triple_sum.real)
            result["biphase_status"] = "descriptive_only_no_significance"
    return result
