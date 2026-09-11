#!/usr/bin/env python3
"""Parent full-waveform DSP replay; no candidate/producer imports.

This reimplements the numerical sequence but uses the same NumPy/SciPy
libraries. It checks implementation consistency, not independent scientific
validity, perceptual validity, or an external measurement acceptance gate.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import butter, firwin, hilbert, resample_poly, sosfiltfilt


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("receipt must be new and outside immutable result")
    commit_path = root / "COMMIT.json"
    commit_hash = digest(commit_path)
    commit = json.loads(commit_path.read_text())
    if commit.get("status") != "committed" or commit.get("kind") != "development_pilot_result":
        raise ValueError("not a committed pilot")
    products = commit["products"]
    if {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != set(products) | {"COMMIT.json"}:
        raise ValueError("inventory mismatch")
    for name, expected in products.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("unsafe path")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("product bytes mismatch")
    measurements = sorted((root / "candidate_json").glob("*.json"))
    if len(measurements) != 432:
        raise ValueError("expected all 432 conditions")
    taps = firwin(numtaps=1281, cutoff=250, fs=16000, window=("kaiser", 8.6))
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(1000) / 1000)
    frequency = np.arange(501, dtype=np.float64) / 2
    maximum_absolute_spectrum_error = 0.
    maximum_absolute_energy_error = 0.
    maximum_absolute_envelope_mean_error = 0.
    comparisons = 0
    for path in measurements:
        record = json.loads(path.read_text())
        wav_path = root / "derivatives" / (path.stem + ".wav")
        fs, audio = wavfile.read(wav_path)
        if fs != 16000 or audio.shape != (64000,) or audio.dtype != np.float64:
            raise ValueError("derivative PCM contract")
        if digest(wav_path) != record["derivative"]["sha256"]:
            raise ValueError("derivative linkage")
        result = record["measurement"]
        if result["sample_count"] != 64000 or result["window_count"] != 1 or result["sample_rate_hz"] != 16000:
            raise ValueError("measurement dimensions")
        np.testing.assert_array_equal(frequency, result["frequency_hz"])
        peak = float(np.max(np.abs(audio)))
        scaled = audio / peak if peak else np.zeros(64000)
        for index, limits in enumerate(((250, 1000), (1000, 3000), (3000, 7000))):
            filtered = sosfiltfilt(butter(4, limits, btype="bandpass", output="sos", fs=16000),
                                   scaled, padtype="odd", padlen=27) if peak else np.zeros(64000)
            env = resample_poly(np.abs(hilbert(filtered)), up=1, down=32,
                                window=taps, padtype="constant")
            env = env[125:1125]
            mean = float(np.mean(env))
            if mean > 0:
                centered = env / mean - 1
                centered -= np.mean(centered)
                power = np.abs(np.fft.rfft(centered * window)) ** 2 / (1000 * np.sum(window ** 2))
                power[1:500] *= 2
            else:
                power = np.zeros(501)
            saved = result["bands"][index]["windows"][0]
            if saved["start_seconds"] != .25 or saved["end_seconds"] != 2.25:
                raise ValueError("analysis interval")
            observed = np.asarray(saved["modulation_power"])
            np.testing.assert_allclose(power, observed, rtol=1e-11, atol=1e-13)
            maximum_absolute_spectrum_error = max(maximum_absolute_spectrum_error, float(np.max(np.abs(power-observed))))
            actual_energies = [float(np.mean(scaled[4000:36000] ** 2)) * peak ** 2,
                               float(np.mean(filtered[4000:36000] ** 2)) * peak ** 2]
            stored_energies = [saved["input_mean_square"], saved["band_mean_square"]]
            np.testing.assert_allclose(actual_energies, stored_energies, rtol=1e-11, atol=1e-13)
            maximum_absolute_energy_error = max(maximum_absolute_energy_error,
                float(np.max(np.abs(np.asarray(actual_energies)-stored_energies))))
            np.testing.assert_allclose(mean * peak, saved["envelope_mean"], rtol=1e-11, atol=1e-13)
            maximum_absolute_envelope_mean_error = max(maximum_absolute_envelope_mean_error,
                abs(mean * peak - saved["envelope_mean"]))
            comparisons += 1
    if digest(commit_path) != commit_hash:
        raise ValueError("publication changed")
    receipt = {"passed": True, "scope": "full_derivative_waveform_DSP_replay_same_numerical_libraries",
               "result_commit_sha256": commit_hash, "code_sha256": digest(__file__),
               "derivatives_replayed": len(measurements), "band_windows_replayed": comparisons,
               "spectrum_bins_compared": comparisons * 501,
               "max_absolute_spectrum_error": maximum_absolute_spectrum_error,
               "max_absolute_energy_error": maximum_absolute_energy_error,
               "max_absolute_envelope_mean_error": maximum_absolute_envelope_mean_error,
               "rtol": 1e-11, "atol": 1e-13, "producer_imported": False,
               "external_gate_passed": False, "classifier_admitted": False,
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    replay(args.result_dir, args.audit_output)
