#!/usr/bin/env python3
"""Build a frozen, MIDI-grounded pairwise Music Flamingo probe from GMD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import mido
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


SEED = 20260903
KIT_CLASSES = {
    "kick": {35, 36},
    "snare": {37, 38, 39, 40},
    "tom": {41, 43, 45, 47, 48, 50},
    "hihat": {22, 26, 42, 44, 46},
    "cymbal": {49, 51, 52, 53, 55, 57, 59},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gmd-root", type=Path, required=True)
    p.add_argument(
        "--audio-parquet",
        type=Path,
        help="Optional HF transport mirror containing official GMD test audio chunks.",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--target-pairs", type=int, default=32)
    p.add_argument("--bars", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def drum_class(pitch: int) -> int:
    for idx, pitches in enumerate(KIT_CLASSES.values()):
        if pitch in pitches:
            return idx
    return len(KIT_CLASSES)


def midi_note_events(path: Path) -> list[tuple[float, int, int]]:
    midi = mido.MidiFile(path)
    tempo = 500000
    now = 0.0
    events: list[tuple[float, int, int]] = []
    for msg in mido.merge_tracks(midi.tracks):
        now += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            events.append((now, int(msg.note), int(msg.velocity)))
    return events


def pitch_conditional_dynamics(events: list[tuple[float, int, int]]) -> tuple[float, float]:
    by_pitch: dict[int, list[float]] = {}
    for _, pitch, velocity in events:
        by_pitch.setdefault(pitch, []).append(float(velocity))
    cv_num = cv_den = delta_num = delta_den = 0.0
    for values in by_pitch.values():
        if len(values) < 4:
            continue
        arr = np.asarray(values, dtype=np.float64)
        mean = max(float(np.mean(arr)), 1e-8)
        median = max(float(np.median(arr)), 1e-8)
        weight = float(len(arr))
        cv_num += weight * float(np.std(arr, ddof=0) / mean)
        cv_den += weight
        delta_num += weight * float(np.median(np.abs(np.diff(arr))) / median)
        delta_den += weight
    if cv_den == 0 or delta_den == 0:
        return math.nan, math.nan
    return cv_num / cv_den, delta_num / delta_den


def rhythm_metrics(
    events: list[tuple[float, int, int]], start: float, bar_seconds: float, bars: int
) -> tuple[float, float]:
    vectors = np.zeros((bars, len(KIT_CLASSES) + 1, 16), dtype=bool)
    counts = np.zeros(bars, dtype=np.float64)
    for time_s, pitch, _ in events:
        rel = time_s - start
        if rel < 0 or rel >= bars * bar_seconds:
            continue
        bar = min(int(rel / bar_seconds), bars - 1)
        phase = (rel - bar * bar_seconds) / bar_seconds
        grid = int(np.rint(phase * 16.0)) % 16
        vectors[bar, drum_class(pitch), grid] = True
        counts[bar] += 1
    distances = []
    flat = vectors.reshape(bars, -1)
    for i, j in itertools.combinations(range(bars), 2):
        union = np.logical_or(flat[i], flat[j]).sum()
        intersection = np.logical_and(flat[i], flat[j]).sum()
        distances.append(0.0 if union == 0 else 1.0 - intersection / union)
    mean_count = max(float(np.mean(counts)), 1e-8)
    return float(np.mean(distances)), float(np.std(counts, ddof=0) / mean_count)


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = max(1.4826 * mad, 1e-8)
    return (values - median) / scale


def locate_info(root: Path) -> Path:
    direct = root / "info.csv"
    if direct.exists():
        return direct
    hits = list(root.rglob("info.csv"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one info.csv below {root}, found {len(hits)}")
    return hits[0]


def load_audio_window(path: Path, start_s: float, duration_s: float) -> tuple[np.ndarray, int, float]:
    info = sf.info(path)
    start_frame = int(round(start_s * info.samplerate))
    frames = int(round(duration_s * info.samplerate))
    audio, sr = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=True)
    shortfall = frames - len(audio)
    if shortfall > 2:
        raise RuntimeError(f"Short audio crop {path}: {len(audio)} < {frames}")
    if shortfall > 0:
        audio = np.pad(audio, ((0, shortfall), (0, 0)), mode="constant")
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))) + 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    return audio, sr, rms_db


def build_candidates_from_files(args: argparse.Namespace) -> list[dict]:
    info_path = locate_info(args.gmd_root)
    base = info_path.parent
    rows = list(csv.DictReader(info_path.open(newline="", encoding="utf-8")))
    candidates: list[dict] = []
    source_errors = []
    for row in rows:
        if row.get("split") != "test" or row.get("beat_type") != "beat":
            continue
        if row.get("time_signature") not in {"4-4", "4/4"}:
            continue
        bpm = float(row["bpm"])
        bar_seconds = 4.0 * 60.0 / bpm
        duration_s = args.bars * bar_seconds
        if 2.0 * duration_s + 1.0 > 30.0:
            continue
        midi_path = base / row["midi_filename"]
        audio_path = base / row["audio_filename"]
        if not midi_path.exists() or not audio_path.exists():
            source_errors.append({"id": row.get("id"), "error": "missing source file"})
            continue
        try:
            events = midi_note_events(midi_path)
            audio_info = sf.info(audio_path)
            usable_s = min(float(row["duration"]), audio_info.duration, events[-1][0] if events else 0.0)
            n_windows = int(usable_s // duration_s)
            for win_idx in range(n_windows):
                start = win_idx * duration_s
                end = start + duration_s
                window_events = [e for e in events if start <= e[0] < end]
                if len(window_events) < 32:
                    continue
                velocity_cv, velocity_delta = pitch_conditional_dynamics(window_events)
                if not np.isfinite(velocity_cv) or not np.isfinite(velocity_delta):
                    continue
                rhythm_jaccard, density_cv = rhythm_metrics(window_events, start, bar_seconds, args.bars)
                _, _, rms_db = load_audio_window(audio_path, start, duration_s)
                candidates.append(
                    {
                        "source_id": row["id"],
                        "drummer": row["drummer"],
                        "session": row["session"],
                        "style": row["style"],
                        "bpm": bpm,
                        "bars": args.bars,
                        "window_index": win_idx,
                        "start_s": start,
                        "duration_s": duration_s,
                        "hit_count": len(window_events),
                        "velocity_cv": velocity_cv,
                        "velocity_delta_mad": velocity_delta,
                        "rhythm_jaccard": rhythm_jaccard,
                        "onset_density_cv": density_cv,
                        "rms_dbfs": rms_db,
                        "midi_source": str(midi_path.relative_to(args.gmd_root)),
                        "audio_source": str(audio_path.relative_to(args.gmd_root)),
                        "audio_source_base": "gmd_root",
                    }
                )
        except Exception as exc:
            source_errors.append({"id": row.get("id"), "error": repr(exc)})
    if not candidates:
        raise RuntimeError("No eligible GMD windows")
    velocity_cv = np.asarray([x["velocity_cv"] for x in candidates])
    velocity_delta = np.asarray([x["velocity_delta_mad"] for x in candidates])
    z_cv = robust_z(velocity_cv)
    z_delta = robust_z(velocity_delta)
    for idx, item in enumerate(candidates):
        item["dynamics_score"] = float(0.5 * (z_cv[idx] + z_delta[idx]))
        item["rhythm_score"] = float(item["rhythm_jaccard"])
        item["window_id"] = f"{item['source_id'].replace('/', '_')}__w{item['window_index']:03d}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "reports" / "source_errors.json").write_text(
        json.dumps(source_errors, indent=2), encoding="utf-8"
    )
    return candidates


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def build_candidates_from_parquet(args: argparse.Namespace) -> list[dict]:
    info_path = locate_info(args.gmd_root)
    info_rows = list(csv.DictReader(info_path.open(newline="", encoding="utf-8")))
    by_audio_name = {(x["drummer"], Path(x["audio_filename"]).name): x for x in info_rows}
    by_midi_name = {(x["drummer"], Path(x["midi_filename"]).stem): x for x in info_rows}
    table = pq.read_table(args.audio_parquet)
    rows = table.to_pylist()
    source_audio_dir = args.output_root / "source_audio_chunks"
    source_audio_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    source_errors = []
    midi_cache: dict[Path, list[tuple[float, int, int]]] = {}
    for parquet_row in rows:
        original_name = Path(parquet_row["original_file"]).name
        drummer = parquet_row["drummer"]
        info = by_audio_name.get((drummer, original_name)) or by_midi_name.get((drummer, Path(original_name).stem))
        if info is None:
            source_errors.append({"original_file": original_name, "error": "no official metadata match"})
            continue
        if info.get("split") != "test" or info.get("beat_type") != "beat":
            continue
        if info.get("time_signature") not in {"4-4", "4/4"}:
            continue
        bpm = float(info["bpm"])
        bar_seconds = 4.0 * 60.0 / bpm
        duration_s = args.bars * bar_seconds
        if 2.0 * duration_s + 1.0 > 30.0:
            continue
        audio_value = parquet_row["audio"]
        audio_bytes = audio_value.get("bytes") if isinstance(audio_value, dict) else None
        if not audio_bytes:
            source_errors.append({"original_file": original_name, "error": "missing embedded audio bytes"})
            continue
        chunk_idx = int(parquet_row["chunk_idx"])
        chunk_path = source_audio_dir / f"{safe_stem(Path(original_name).stem)}__c{chunk_idx:03d}.wav"
        if not chunk_path.exists():
            chunk_path.write_bytes(audio_bytes)
        try:
            audio_info = sf.info(chunk_path)
            downbeats = [
                float(x["timestamp"])
                for x in parquet_row.get("events", [])
                if int(x["beat"]) == 1
            ]
            possible_starts = [x for x in downbeats if x + duration_s <= audio_info.duration + 1e-3]
            starts = []
            for candidate_start in possible_starts:
                if not starts or candidate_start >= starts[-1] + duration_s - 1e-3:
                    starts.append(candidate_start)
            if not starts:
                continue
            midi_path = info_path.parent / info["midi_filename"]
            if midi_path not in midi_cache:
                midi_cache[midi_path] = midi_note_events(midi_path)
            events = midi_cache[midi_path]
            for subwindow_idx, local_start in enumerate(starts):
                global_start = float(parquet_row["start_time"]) + local_start
                global_end = global_start + duration_s
                window_events = [e for e in events if global_start <= e[0] < global_end]
                if len(window_events) < 24:
                    continue
                velocity_cv, velocity_delta = pitch_conditional_dynamics(window_events)
                if not np.isfinite(velocity_cv) or not np.isfinite(velocity_delta):
                    continue
                rhythm_jaccard, density_cv = rhythm_metrics(window_events, global_start, bar_seconds, args.bars)
                _, _, rms_db = load_audio_window(chunk_path, local_start, duration_s)
                candidates.append(
                    {
                        "source_id": info["id"],
                        "drummer": info["drummer"],
                        "session": info["session"],
                        "style": info["style"],
                        "bpm": bpm,
                        "bars": args.bars,
                        "window_index": chunk_idx * 100 + subwindow_idx,
                        "chunk_index": chunk_idx,
                        "subwindow_index": subwindow_idx,
                        "start_s": local_start,
                        "global_start_s": global_start,
                        "duration_s": duration_s,
                        "hit_count": len(window_events),
                        "velocity_cv": velocity_cv,
                        "velocity_delta_mad": velocity_delta,
                        "rhythm_jaccard": rhythm_jaccard,
                        "onset_density_cv": density_cv,
                        "rms_dbfs": rms_db,
                        "midi_source": str(midi_path.relative_to(args.gmd_root)),
                        "audio_source": str(chunk_path.relative_to(args.output_root)),
                        "audio_source_base": "experiment_root",
                        "transport_parquet": args.audio_parquet.name,
                    }
                )
        except Exception as exc:
            source_errors.append({"original_file": original_name, "chunk_idx": chunk_idx, "error": repr(exc)})
    if not candidates:
        raise RuntimeError("No eligible GMD windows from audio parquet")
    velocity_cv = np.asarray([x["velocity_cv"] for x in candidates])
    velocity_delta = np.asarray([x["velocity_delta_mad"] for x in candidates])
    z_cv = robust_z(velocity_cv)
    z_delta = robust_z(velocity_delta)
    for idx, item in enumerate(candidates):
        item["dynamics_score"] = float(0.5 * (z_cv[idx] + z_delta[idx]))
        item["rhythm_score"] = float(item["rhythm_jaccard"])
        item["window_id"] = f"{item['source_id'].replace('/', '_')}__c{item['window_index']:03d}"
    (args.output_root / "reports" / "source_errors.json").write_text(
        json.dumps(source_errors, indent=2), encoding="utf-8"
    )
    return candidates


def build_candidates(args: argparse.Namespace) -> list[dict]:
    if args.audio_parquet:
        return build_candidates_from_parquet(args)
    return build_candidates_from_files(args)


def pair_candidates(candidates: list[dict], task: str, seed: int) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for item in candidates:
        by_source.setdefault(item["source_id"], []).append(item)
    proposed = []
    target_key = "dynamics_score" if task == "dynamics" else "rhythm_score"
    nuisance_key = "rhythm_score" if task == "dynamics" else "dynamics_score"
    min_target_gap = 0.50 if task == "dynamics" else 0.08
    max_nuisance_gap = 0.35 if task == "dynamics" else 0.75
    for source_id, items in by_source.items():
        options = []
        for left, right in itertools.combinations(items, 2):
            target_gap = abs(left[target_key] - right[target_key])
            nuisance_gap = abs(left[nuisance_key] - right[nuisance_key])
            rms_gap = abs(left["rms_dbfs"] - right["rms_dbfs"])
            if target_gap < min_target_gap or nuisance_gap > max_nuisance_gap or rms_gap > 6.0:
                continue
            low, high = sorted((left, right), key=lambda x: x[target_key])
            quality = target_gap - 0.35 * nuisance_gap - 0.02 * rms_gap
            options.append(
                {
                    "task": task,
                    "source_id": source_id,
                    "style": left["style"],
                    "drummer": left["drummer"],
                    "low_window_id": low["window_id"],
                    "high_window_id": high["window_id"],
                    "target_gap": target_gap,
                    "nuisance_gap": nuisance_gap,
                    "rms_gap_db": rms_gap,
                    "quality": quality,
                }
            )
        if options:
            options.sort(key=lambda x: (-x["quality"], stable_key(str(x), seed)))
            proposed.append(options[0])
    proposed.sort(key=lambda x: (-x["quality"], stable_key(str(x), seed)))
    return proposed


def balanced_select(pairs: list[dict], target: int, seed: int) -> list[dict]:
    selected = []
    style_counts: dict[str, int] = {}
    for pair in pairs:
        primary_style = pair["style"].split("/")[0]
        if style_counts.get(primary_style, 0) >= 4:
            continue
        selected.append(pair)
        style_counts[primary_style] = style_counts.get(primary_style, 0) + 1
        if len(selected) == target:
            return selected
    chosen = {(x["task"], x["source_id"]) for x in selected}
    for pair in pairs:
        key = (pair["task"], pair["source_id"])
        if key in chosen:
            continue
        selected.append(pair)
        chosen.add(key)
        if len(selected) == target:
            break
    selected.sort(key=lambda x: stable_key(f"{x['task']}:{x['source_id']}", seed))
    return selected


def materialize(args: argparse.Namespace, candidates: list[dict], selected: list[dict]) -> None:
    info_path = locate_info(args.gmd_root)
    window_by_id = {x["window_id"]: x for x in candidates}
    used_ids = {p[k] for p in selected for k in ("low_window_id", "high_window_id")}
    audio_dir = args.output_root / "audio" / "windows"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for window_id in sorted(used_ids):
        item = window_by_id[window_id]
        source = (
            args.output_root / item["audio_source"]
            if item.get("audio_source_base") == "experiment_root"
            else args.gmd_root / item["audio_source"]
        )
        audio, sr, _ = load_audio_window(source, item["start_s"], item["duration_s"])
        output = audio_dir / f"{window_id}.flac"
        sf.write(output, audio, sr, subtype="PCM_16", format="FLAC")
        item["probe_audio"] = str(output.relative_to(args.output_root))
        item["probe_audio_sha256"] = sha256_file(output)
        item["probe_sample_rate"] = sr
        item["probe_frames"] = len(audio)

    with (args.output_root / "analysis" / "window_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(candidates[0].keys())
        for item in candidates[1:]:
            keys.extend(key for key in item if key not in keys)
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(candidates)

    prompt_manifest = []
    pair_rows = []
    pair_audio_dir = args.output_root / "audio" / "pairs"
    pair_audio_dir.mkdir(parents=True, exist_ok=True)
    for pair_idx, pair in enumerate(selected):
        pair = dict(pair)
        pair_id = f"{pair['task']}_{pair_idx:03d}"
        pair["pair_id"] = pair_id
        pair_rows.append(pair)
        low = window_by_id[pair["low_window_id"]]
        high = window_by_id[pair["high_window_id"]]
        high_first = int(stable_key(pair_id, args.seed), 16) % 2 == 0
        for reversal in (0, 1):
            effective_high_first = high_first ^ bool(reversal)
            a = high if effective_high_first else low
            b = low if effective_high_first else high
            audio_a, sr_a = sf.read(args.output_root / a["probe_audio"], dtype="float32", always_2d=True)
            audio_b, sr_b = sf.read(args.output_root / b["probe_audio"], dtype="float32", always_2d=True)
            if sr_a != sr_b:
                raise RuntimeError(f"Sample-rate mismatch in {pair_id}: {sr_a} vs {sr_b}")
            fade_frames = min(int(round(0.01 * sr_a)), len(audio_a) // 2, len(audio_b) // 2)
            if fade_frames:
                fade_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)[:, None]
                fade_out = fade_in[::-1]
                audio_a[:fade_frames] *= fade_in
                audio_a[-fade_frames:] *= fade_out
                audio_b[:fade_frames] *= fade_in
                audio_b[-fade_frames:] *= fade_out
            silence = np.zeros((sr_a, max(audio_a.shape[1], audio_b.shape[1])), dtype=np.float32)
            if audio_a.shape[1] != silence.shape[1]:
                audio_a = np.repeat(audio_a, silence.shape[1], axis=1)
            if audio_b.shape[1] != silence.shape[1]:
                audio_b = np.repeat(audio_b, silence.shape[1], axis=1)
            paired_audio = np.concatenate([audio_a, silence, audio_b], axis=0)
            if len(paired_audio) > 30 * sr_a:
                raise RuntimeError(f"Paired audio exceeds 30 s for {pair_id}")
            paired_path = pair_audio_dir / f"{pair_id}__order{reversal}.flac"
            sf.write(paired_path, paired_audio, sr_a, subtype="PCM_16", format="FLAC")
            prompt_manifest.append(
                {
                    "query_id": f"{pair_id}__order{reversal}",
                    "pair_id": pair_id,
                    "task": pair["task"],
                    "order_reversal": reversal,
                    "audio_a": a["probe_audio"],
                    "audio_b": b["probe_audio"],
                    "paired_audio": str(paired_path.relative_to(args.output_root)),
                    "paired_audio_sha256": sha256_file(paired_path),
                    "paired_duration_s": len(paired_audio) / sr_a,
                    "audio_a_window_id": a["window_id"],
                    "audio_b_window_id": b["window_id"],
                    "correct_choice": "A" if effective_high_first else "B",
                    "high_window_id": high["window_id"],
                    "low_window_id": low["window_id"],
                    "target_gap": pair["target_gap"],
                    "nuisance_gap": pair["nuisance_gap"],
                    "rms_gap_db": pair["rms_gap_db"],
                    "source_id": pair["source_id"],
                    "style": pair["style"],
                    "drummer": pair["drummer"],
                }
            )
    with (args.output_root / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)
    with (args.output_root / "prompt_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in prompt_manifest:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "seed": args.seed,
        "bars_per_window": args.bars,
        "eligible_windows": len(candidates),
        "eligible_sources": len({x["source_id"] for x in candidates}),
        "selected_pairs": {
            task: sum(x["task"] == task for x in selected) for task in ("dynamics", "rhythm")
        },
        "queries": len(prompt_manifest),
        "source_info_csv": str(info_path),
        "audio_parquet": str(args.audio_parquet) if args.audio_parquet else None,
        "audio_parquet_sha256": sha256_file(args.audio_parquet) if args.audio_parquet else None,
        "prompt_manifest_sha256": sha256_file(args.output_root / "prompt_manifest.jsonl"),
        "pairs_sha256": sha256_file(args.output_root / "pairs.csv"),
    }
    (args.output_root / "reports" / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    for subdir in ("analysis", "reports"):
        (args.output_root / subdir).mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args)
    selected = []
    for task in ("dynamics", "rhythm"):
        selected.extend(balanced_select(pair_candidates(candidates, task, args.seed), args.target_pairs, args.seed))
    if not selected:
        raise RuntimeError("No eligible matched pairs")
    materialize(args, candidates, selected)


if __name__ == "__main__":
    main()
