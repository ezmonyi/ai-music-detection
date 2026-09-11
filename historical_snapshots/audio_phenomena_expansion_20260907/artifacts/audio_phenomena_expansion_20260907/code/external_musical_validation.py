#!/usr/bin/env python3
"""Controlled H/M sensitivity on matched blocks from external real music.

This does NOT validate natural chord labels or musical motif recognition. It
tests known content recurrence and chroma-path changes using real MUSDB blocks.
Every rendered stimulus and its exact construction recipe is retained.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: hashlib.sha256(r["track_id"].encode()).hexdigest())
    rows = rows[:24]
    if len(rows) != 24:
        raise ValueError("Requires 24 external real tracks")
    for r in rows:
        r["mixture_sha256"] = sha(r["mixture"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selected_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.output / "files_from.txt").write_text("".join(Path(r["mixture"]).name + "\n" for r in rows))
    print(json.dumps({"selected_tracks": len(rows), "source_manifest_sha256": sha(args.manifest)}))


def run(args):
    import math
    import numpy as np
    import scipy
    import soundfile as sf
    import librosa
    from scipy.signal import resample_poly
    from musical_features import extract_musical_features, H_MEASURE_NAMES, M_MEASURE_NAMES
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    inputs = []
    for row in rows:
        path = args.audio_dir / Path(row["mixture"]).name
        if sha(path) != row["mixture_sha256"]:
            raise ValueError("Source audio checksum mismatch: " + str(path))
        y, sr = sf.read(path, dtype="float64", always_2d=True)
        y = y.mean(axis=1)
        y -= y.mean()
        if sr != 16000:
            d = math.gcd(sr, 16000)
            y = resample_poly(y, 16000 // d, sr // d, window=("kaiser", 5.0))
        # One-second block from a fixed interior offset; remove DC and retain
        # native amplitude. Five-ms edge fades are identical in every variant.
        block = y[2 * 16000:3 * 16000].copy()
        if len(block) != 16000:
            raise ValueError("Source too short for fixed block")
        block -= block.mean()
        block[:80] *= np.linspace(0, 1, 80)
        block[-80:] *= np.linspace(1, 0, 80)
        inputs.append(block.astype(np.float32))
    args.output.mkdir(parents=True, exist_ok=True)
    audio_out = args.output / "stimuli"
    audio_out.mkdir(exist_ok=True)
    feature_rows, recipes, checks = [], [], []
    for trial in range(12):
        seed = 20260907 + trial
        rng = np.random.default_rng(seed)
        phrase = rng.choice(len(inputs), 16, replace=False).tolist()
        repeated = phrase * 4
        shuffled = repeated.copy()
        rng.shuffle(shuffled)
        variants = {"repeated_16s": repeated, "matched_block_shuffle": shuffled,
                    "stationary_1s_texture": [phrase[0]] * 64}
        features = {}
        for name, sequence in variants.items():
            path = audio_out / f"trial_{trial:02d}_{name}.wav"
            waveform = np.concatenate([inputs[i] for i in sequence])
            sf.write(path, waveform, 16000, subtype="FLOAT")
            measured = extract_musical_features(waveform, 16000)
            features[name] = measured
            feature_rows.append({"trial": trial, "variant": name, "audio_path": str(path),
                                 "audio_sha256": sha(path), **measured})
            recipes.append({"trial": trial, "seed": seed, "variant": name,
                            "duration_s": 64, "source_block_start_s": 2,
                            "source_block_length_s": 1, "edge_fade_ms": 5,
                            "track_id_sequence": [rows[i]["track_id"] for i in sequence]})
        # Algebraic controls have no materialized duplicate audio: gain and
        # polarity are exact recorded transformations of the saved repeat file.
        base_y = np.concatenate([inputs[i] for i in repeated])
        base = features["repeated_16s"]
        gain = extract_musical_features(base_y * .5, 16000)
        polarity = extract_musical_features(-base_y, 16000)
        names = tuple(H_MEASURE_NAMES) + tuple(M_MEASURE_NAMES)
        control_deltas = [abs(float(base[k]) - float(other[k])) for other in (gain, polarity)
                          for k in names if np.isfinite(base[k]) and np.isfinite(other[k])]
        shuffle = features["matched_block_shuffle"]
        steady = features["stationary_1s_texture"]
        checks.append({"trial": trial,
                       "repeat_M_status": base["M_status"], "shuffle_M_status": shuffle["M_status"],
                       "stationary_M_status": steady["M_status"],
                       "repeat_minus_shuffle_peak_similarity": float(base["M_recurrence_peak_similarity"]) - float(shuffle["M_recurrence_peak_similarity"]),
                       "repeat_minus_shuffle_density": float(base["M_recurrence_density"]) - float(shuffle["M_recurrence_density"]),
                       "repeat_best_lag_sec": base["M_best_lag_sec"],
                       "changing_minus_stationary_H_path": float(base["H_chroma_path_change_median"]) - float(steady["H_chroma_path_change_median"]),
                       "stationary_recurrence_density": steady["M_recurrence_density"],
                       "stationary_lag_contrast": steady["M_recurrence_lag_contrast"],
                       "gain_polarity_max_abs_delta": max(control_deltas, default=float("nan"))})
        print(json.dumps({"trial_complete": trial, "of": 12}), flush=True)
    for name, records in (("feature_measurements.csv", feature_rows), ("paired_checks.csv", checks)):
        with (args.output / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    (args.output / "stimulus_recipes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recipes))
    summary = {"source_tracks": len(rows), "trials": 12, "materialized_stimuli": len(feature_rows),
               "duration_sec": 64, "phase_only_intervention": False,
               "natural_chord_or_motif_annotation_validation": False,
               "source_manifest_sha256": sha(args.manifest), "script_sha256": sha(__file__),
               "extractor_sha256": sha(Path(__file__).with_name("musical_features.py")),
               "runtime": {"numpy": np.__version__, "scipy": scipy.__version__,
                           "librosa": librosa.__version__, "soundfile": sf.__version__}}
    for key in ("repeat_minus_shuffle_peak_similarity", "repeat_minus_shuffle_density", "changing_minus_stationary_H_path"):
        values = np.array([c[key] for c in checks], dtype=float)
        valid = np.isfinite(values)
        summary[key] = {"valid_pairs": int(valid.sum()),
                        "positive_pairs": int(np.sum(values[valid] > 0)),
                        "median_delta": float(np.median(values[valid])) if valid.any() else None}
    summary["gain_polarity_max_abs_delta"] = float(np.nanmax([c["gain_polarity_max_abs_delta"] for c in checks]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = """# H/M controlled external-real-audio measurement checks

The sources are 24 real MUSDB preview recordings, independent of the AI/Human
classifier corpus. Twelve deterministic constructions use one-second blocks
from 16 different sources: a 16-second phrase repeated four times, the same 64
blocks randomly permuted (identical block counts), and a stationary one-second
texture loop. Each stimulus is 64 seconds at mono 16 kHz and saved as float WAV.

These are deliberately artificial mosaics. Success proves sensitivity to known
chroma-path changes or content recurrence, not natural harmony correctness,
semantic motif understanding, or AI authorship. Cuts introduce abrupt timbre and
rhythm changes. Real-world annotation validation remains incomplete.

Stationary textures are an explicit semantic negative control: they can have
high recurrence density without a meaningful musical motif. Inspect lag contrast
alongside recurrence rather than calling every returning window a distinct motif.
Gain/polarity are algebraic controls; no classifier labels were consulted.

Exact paired values: `paired_checks.csv`; every metric: `feature_measurements.csv`;
source order/seeds: `stimulus_recipes.jsonl`; decoded inputs/checksums are in the
selected source manifest. Numerical summary follows.

```json
""" + json.dumps(summary, indent=2) + "\n```\n"
    (args.output / "REPORT_EN.md").write_text(report)
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prepare", "run"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        if args.audio_dir is None:
            p.error("run requires --audio-dir")
        run(args)


if __name__ == "__main__":
    main()
