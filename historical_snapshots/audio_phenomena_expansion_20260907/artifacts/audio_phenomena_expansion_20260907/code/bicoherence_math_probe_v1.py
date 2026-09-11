"""Coefficient-domain development probe, not an audio or classifier admission gate.

Illustrates why ensemble averaging and cautious physical interpretation matter.
No existing music is read, and no classifier is fit. Synthetic concatenated
frames are not natural music examples or independent real-world validation.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

SEED = 20260907
N = 256
K = 128
PAIR = (4, 7)


def estimate(spectra, pair=PAIR):
    a = np.asarray(spectra, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("at least two finite coefficient realizations required")
    f1, f2 = pair
    if not (0 < f1 <= f2 and f1+f2 < a.shape[1]):
        raise ValueError("invalid positive bifrequency")
    peak = float(np.max(np.abs(a)))
    if peak == 0:
        return {"status": "zero_energy", "squared_bicoherence": None, "biphase_radians": None}
    x = a/peak
    u, v = x[:, f1]*x[:, f2], x[:, f1+f2]
    denominator = float(np.sum(np.abs(u)**2)*np.sum(np.abs(v)**2))
    if denominator == 0:
        return {"status": "missing_triad_energy", "squared_bicoherence": None, "biphase_radians": None}
    triple_sum = np.sum(u*np.conj(v))
    value = float(np.abs(triple_sum)**2/denominator)
    if not np.isfinite(value) or value > 1+1e-12:
        raise ValueError("Cauchy-Schwarz bound violated")
    return {"status": "ok", "squared_bicoherence": min(value, 1.),
            "biphase_radians": float(np.angle(triple_sum))}


def fixtures():
    rng = np.random.default_rng(SEED)
    p = rng.uniform(-np.pi, np.pi, (K, 3))
    def coefficients(phase):
        z = np.zeros((K, N//2+1), dtype=np.complex128)
        z[:, [4, 7, 11]] = np.exp(1j*phase)
        return z
    closed = coefficients(np.column_stack((p[:, :2], p[:, 0]+p[:, 1])))
    random = coefficients(p)
    static = coefficients(np.tile([.2, .7, -1.1], (K, 1)))
    delay = np.exp(-2j*np.pi*np.arange(N//2+1)*13/N)
    return {"phase_closed_ensemble": closed, "independent_phase_ensemble": random,
            "fixed_three_oscillators_no_nonlinear_operation": static,
            "phase_closed_gain_0p1": .1*closed, "phase_closed_polarity": -closed,
            "phase_closed_circular_delay_13_samples": closed*delay,
            "silence": np.zeros_like(closed)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/bicoherence_math_probe_v1"
    if output.exists():
        raise ValueError("exclusive new output required")
    data = fixtures()
    measured = {k: estimate(v) for k, v in data.items()}
    coupled = measured["phase_closed_ensemble"]["squared_bicoherence"]
    uncoupled = measured["independent_phase_ensemble"]["squared_bicoherence"]
    if not (coupled > 1-1e-12 and uncoupled < .1):
        raise ValueError("fixed analytic sanity fixture failed")
    # Each frame has the same power, but the ensemble phase relationships differ.
    np.testing.assert_allclose(np.abs(data["phase_closed_ensemble"])**2,
        np.abs(data["independent_phase_ensemble"])**2, atol=1e-14, rtol=1e-14)
    for condition in ("phase_closed_gain_0p1", "phase_closed_polarity",
                      "phase_closed_circular_delay_13_samples"):
        np.testing.assert_allclose(measured[condition]["squared_bicoherence"], coupled, atol=1e-12)
    x = data["independent_phase_ensemble"]
    u, v = x[:, 4]*x[:, 7], x[:, 11]
    wrong_single_frame_values = np.abs(u*np.conj(v))**2/(np.abs(u)**2*np.abs(v)**2)
    np.testing.assert_allclose(wrong_single_frame_values, 1., atol=1e-12)
    output.mkdir()
    np.savez(output / "synthetic_coefficients.npz", **data)
    np.savez(output / "synthetic_frames.npz", **{k: np.fft.irfft(v, n=N, axis=1) for k,v in data.items()})
    result = {"status": "synthetic_development_probe_complete", "seed": SEED, "frame_samples": N,
        "realizations": K, "sample_rate_hz_for_interpretation": 16000, "frequency_bins": [4,7,11],
        "frequency_hz": [250.,437.5,687.5], "results": measured,
        "incorrect_mean_single_frame_squared_bicoherence": float(wrong_single_frame_values.mean()),
        "external_validation_passed": False, "classifier_admitted": False, "classifier_fits": 0,
        "input_music_files_read": 0, "code_sha256": sha(Path(__file__)),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "limitations": ["Coefficient-domain fixtures do not validate segmentation/windowing on continuous music.",
            "High segment-averaged bicoherence alone does not identify nonlinear causation.",
            "Biphase shifts by pi under polarity; magnitude-squared remains invariant.",
            "Matched power and known phases are synthetic controls, not AI/Human examples."]}
    (output / "results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    products = {p.name: {"bytes":p.stat().st_size,"sha256":sha(p)} for p in output.iterdir()}
    (output / "COMMIT.json").write_text(json.dumps({"status":"committed","products":products},indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
