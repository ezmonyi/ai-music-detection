#!/usr/bin/env python3
"""Freeze, materialize, and verify the AIME 500-per-generator test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf


DATASET = "disco-eth/AIME"
MAIN_REVISION = "b84d4be5eda830b6eb714998569dba73530f2601"
PARQUET_REVISION = "1bdacac93127439e361bdd19d575d8b596bca4e3"
DEFAULT_SEED = 20_260_904
CLIP_SECONDS = 10.0
SAMPLE_RATE = 44_100
EXPECTED_FRAMES = int(CLIP_SECONDS * SAMPLE_RATE)

PRIMARY_MODELS = (
    "MusicGen Small",
    "MusicGen Medium",
    "MusicGen Large",
    "AudioLDM 2 Large",
    "AudioLDM 2 Music",
    "Mustango",
)
SECONDARY_MODELS = (
    "Udio",
    "Riffusion",
    "Stable Audio v1",
    "Stable Audio v2",
)
AI_MODELS = PRIMARY_MODELS + SECONDARY_MODELS
HUMAN_MODEL = "MTG-Jamendo"
INCLUDED_MODELS = set(AI_MODELS) | {HUMAN_MODEL}
FORBIDDEN_MODELS = {"Suno", "Suno v3", "Suno v3.5", "HeartMuLa", "ACE-Step"}
MODEL_SLUGS = {
    "MusicGen Small": "musicgen_small",
    "MusicGen Medium": "musicgen_medium",
    "MusicGen Large": "musicgen_large",
    "AudioLDM 2 Large": "audioldm2_large",
    "AudioLDM 2 Music": "audioldm2_music",
    "Mustango": "mustango",
    "Udio": "udio",
    "Riffusion": "riffusion",
    "Stable Audio v1": "stable_audio_v1",
    "Stable Audio v2": "stable_audio_v2",
    "MTG-Jamendo": "mtg_jamendo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "materialize", "verify"))
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def freeze_text(path: Path, text: str) -> None:
    """Create a frozen file, or prove an existing file is byte-identical."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed frozen file: {path}")
        return
    atomic_write_text(path, text)


def write_jsonl(path: Path, rows: list[dict[str, Any]], frozen: bool = False) -> None:
    text = "".join(json_line(row) for row in rows)
    if frozen:
        freeze_text(path, text)
    else:
        atomic_write_text(path, text)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_description(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(part) for part in value)
    return str(value)


def prior_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        keys.add((str(row["model"]), str(row["hf_original_id"])))
    return keys


def parquet_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"no parquet shards under {directory}")
    return paths


def scan_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_forbidden = Counter()
    for shard in paths:
        table = pq.read_table(shard, columns=["id", "model", "description"])
        for row_number, row in enumerate(table.to_pylist()):
            model = str(row["model"])
            if model in FORBIDDEN_MODELS:
                seen_forbidden[model] += 1
                continue
            if model not in INCLUDED_MODELS:
                raise RuntimeError(f"unexpected model label {model!r} in {shard.name}")
            description = normalize_description(row["description"])
            output.append(
                {
                    "hf_original_id": str(row["id"]),
                    "model": model,
                    "description": description,
                    "condition_id": hashlib.sha256(description.encode("utf-8")).hexdigest()[:16],
                    "source_parquet": shard.name,
                    "source_row_number": row_number,
                }
            )
    print("forbidden rows present in retained boundary shards:", dict(seen_forbidden), flush=True)
    return output


def freeze_selection(args: argparse.Namespace) -> None:
    metadata = scan_metadata(parquet_paths(args.parquet_dir))
    counts = Counter(str(row["model"]) for row in metadata)
    expected = {model: 500 for model in INCLUDED_MODELS}
    if dict(counts) != expected:
        raise RuntimeError(f"model counts differ from frozen expectation: {dict(counts)}")

    by_model_description: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metadata:
        model = str(row["model"])
        description = str(row["description"])
        if description in by_model_description[model]:
            raise RuntimeError(f"duplicate description within {model}: {description}")
        by_model_description[model][description] = row
    human_descriptions = set(by_model_description[HUMAN_MODEL])
    for model in AI_MODELS:
        if set(by_model_description[model]) != human_descriptions:
            missing = human_descriptions - set(by_model_description[model])
            extra = set(by_model_description[model]) - human_descriptions
            raise RuntimeError(f"prompt mismatch for {model}: missing={len(missing)} extra={len(extra)}")

    rng = random.Random(args.seed)
    shuffled_conditions = sorted(human_descriptions)
    rng.shuffle(shuffled_conditions)
    group_for_description = {
        description: f"group_{index // 100 + 1:02d}"
        for index, description in enumerate(shuffled_conditions)
    }
    prior = prior_keys(args.prior_manifest)
    rows: list[dict[str, Any]] = []
    order = (HUMAN_MODEL,) + AI_MODELS
    for model in order:
        label = 0 if model == HUMAN_MODEL else 1
        cohort = (
            "shared_external_human_control"
            if model == HUMAN_MODEL
            else "primary_pristine_generator_disjoint"
            if model in PRIMARY_MODELS
            else "secondary_previously_seen_generator_family"
        )
        for description in sorted(human_descriptions):
            source = by_model_description[model][description]
            original_id = str(source["hf_original_id"])
            slug = MODEL_SLUGS[model]
            rows.append(
                {
                    "id": f"aime_{slug}_{original_id}",
                    "label": label,
                    "class_name": "human" if label == 0 else "ai",
                    "source": f"aime_{slug}",
                    "model": model,
                    "cohort": cohort,
                    "split": "external_test_frozen",
                    "group_id": group_for_description[description],
                    "condition_id": source["condition_id"],
                    "description": description,
                    "hf_dataset": DATASET,
                    "hf_main_revision": MAIN_REVISION,
                    "hf_parquet_revision": PARQUET_REVISION,
                    "hf_original_id": original_id,
                    "source_parquet": source["source_parquet"],
                    "source_row_number": source["source_row_number"],
                    "prior_external_generator_family_seen": int(model in SECONDARY_MODELS or model == HUMAN_MODEL),
                    "prior_external_exact_track_overlap": int((model, original_id) in prior),
                    "selection_uses_audio_features": False,
                    "selection_uses_detector_scores": False,
                }
            )

    if len(rows) != 5_500:
        raise RuntimeError(f"expected 5500 frozen rows, got {len(rows)}")
    if any(str(row["model"]) in FORBIDDEN_MODELS for row in rows):
        raise RuntimeError("forbidden model entered selection")
    selection_path = args.output_dir / "frozen_selection.jsonl"
    write_jsonl(selection_path, rows, frozen=True)
    digest = sha256_file(selection_path)
    freeze_text(args.output_dir / "frozen_selection.sha256", f"{digest}  frozen_selection.jsonl\n")
    summary = {
        "dataset": DATASET,
        "main_revision": MAIN_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "seed": args.seed,
        "track_count": len(rows),
        "ai_track_count": sum(int(row["label"]) for row in rows),
        "human_track_count": sum(1 - int(row["label"]) for row in rows),
        "model_counts": dict(sorted(Counter(str(row["model"]) for row in rows).items())),
        "cohort_counts": dict(sorted(Counter(str(row["cohort"]) for row in rows).items())),
        "group_condition_counts": dict(sorted(Counter(str(row["group_id"]) for row in rows if row["model"] == HUMAN_MODEL).items())),
        "exact_prior_overlap_count": sum(int(row["prior_external_exact_track_overlap"]) for row in rows),
        "forbidden_models": sorted(FORBIDDEN_MODELS),
        "frozen_selection_sha256": digest,
    }
    freeze_text(
        args.output_dir / "frozen_selection_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


def suffix_for(blob: bytes, hinted_path: str) -> str:
    suffix = Path(hinted_path).suffix.lower()
    if suffix in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
        return suffix
    if blob.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return ".mp3"
    if blob.startswith(b"RIFF"):
        return ".wav"
    if blob.startswith(b"fLaC"):
        return ".flac"
    if blob.startswith(b"OggS"):
        return ".ogg"
    raise RuntimeError(f"unknown audio container for {hinted_path}")


def probe(path: Path, ffprobe: Path) -> dict[str, Any]:
    command = [
        str(ffprobe), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "original_sample_rate": int(stream["sample_rate"]),
        "original_channels": int(stream["channels"]),
        "original_codec": str(stream["codec_name"]),
        "original_duration_s": float(payload["format"]["duration"]),
    }


def crop_start(duration: float, seed: int, row_id: str) -> float:
    available = duration - CLIP_SECONDS
    if available < -0.002:
        raise RuntimeError(f"source shorter than {CLIP_SECONDS}s: {row_id}, duration={duration}")
    if available <= 0.002:
        return 0.0
    value = int(hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()[:16], 16)
    return round(random.Random(value).uniform(0.0, available), 6)


def standardize(source: Path, destination: Path, start: float, ffmpeg: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.flac")
    command = [
        str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{CLIP_SECONDS:.3f}",
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-ac", "2", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
        "-compression_level", "8", "-fflags", "+bitexact", "-flags:a", "+bitexact",
        "-y", str(temporary),
    ]
    subprocess.run(command, check=True)
    info = sf.info(temporary)
    if info.samplerate != SAMPLE_RATE or info.channels != 2 or info.frames != EXPECTED_FRAMES:
        raise RuntimeError(f"invalid standardized clip {temporary}: {info}")
    os.replace(temporary, destination)


def process_audio(
    row: dict[str, Any], blob: bytes, hinted_path: str, output_dir: Path,
    ffmpeg: Path, ffprobe: Path, seed: int, temporary_root: Path,
) -> dict[str, Any]:
    raw_digest = sha256_bytes(blob)
    suffix = suffix_for(blob, hinted_path)
    source_path = temporary_root / f"{row['id']}{suffix}"
    source_path.write_bytes(blob)
    try:
        audio_info = probe(source_path, ffprobe)
        start = crop_start(float(audio_info["original_duration_s"]), seed, str(row["id"]))
        clip_relpath = Path("clips_10s") / str(row["source"]) / f"{row['id']}.flac"
        clip_path = output_dir / clip_relpath
        if not clip_path.exists():
            standardize(source_path, clip_path, start, ffmpeg)
        info = sf.info(clip_path)
        if info.samplerate != SAMPLE_RATE or info.channels != 2 or info.frames != EXPECTED_FRAMES:
            raise RuntimeError(f"invalid existing standardized clip {clip_path}: {info}")
        clip_digest = sha256_file(clip_path)
    finally:
        source_path.unlink(missing_ok=True)
    native_nyquist = float(audio_info["original_sample_rate"]) / 2.0
    return {
        **row,
        **audio_info,
        "native_nyquist_hz": native_nyquist,
        "eligible_through_8khz": int(native_nyquist >= 8_000),
        "eligible_through_10khz": int(native_nyquist >= 10_000),
        "eligible_through_20khz": int(native_nyquist >= 20_000),
        "crop_start_s": start,
        "clip_duration_s": CLIP_SECONDS,
        "standardized_sample_rate": SAMPLE_RATE,
        "standardized_channels": 2,
        "standardized_subtype": "PCM_16",
        "standardized_relpath": clip_relpath.as_posix(),
        "raw_sha256": raw_digest,
        "standardized_sha256": clip_digest,
        "standardized_bytes": clip_path.stat().st_size,
    }


def load_state(path: Path, output_dir: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return state
    for row in read_jsonl(path):
        clip = output_dir / str(row["standardized_relpath"])
        if not clip.exists():
            continue
        info = sf.info(clip)
        if info.samplerate == SAMPLE_RATE and info.channels == 2 and info.frames == EXPECTED_FRAMES:
            state[str(row["id"])] = row
    return state


def append_state(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_line(row))
        handle.flush()
        os.fsync(handle.fileno())


def finalize_manifests(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: (str(row["model"]), str(row["hf_original_id"])))
    if len(rows) != 5_500 or len({str(row["id"]) for row in rows}) != 5_500:
        raise RuntimeError("materialized rows are incomplete or non-unique")
    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    digest = sha256_file(manifest)
    atomic_write_text(output_dir / "manifest.sha256", f"{digest}  manifest.jsonl\n")

    qc_fields = (
        "id", "model", "cohort", "group_id", "prior_external_exact_track_overlap",
        "original_sample_rate", "original_channels", "original_codec", "original_duration_s",
        "native_nyquist_hz", "eligible_through_8khz", "eligible_through_10khz",
        "eligible_through_20khz", "crop_start_s", "standardized_bytes", "raw_sha256",
        "standardized_sha256", "standardized_relpath",
    )
    write_csv(output_dir / "audio_qc.csv", [{key: row[key] for key in qc_fields} for row in rows])

    humans = [row for row in rows if row["model"] == HUMAN_MODEL]
    for model in AI_MODELS:
        slug = MODEL_SLUGS[model]
        model_rows = [row for row in rows if row["model"] == model]
        view = []
        for group_index in range(1, 6):
            group_id = f"group_{group_index:02d}"
            group = [row for row in humans + model_rows if row["group_id"] == group_id]
            random.Random(DEFAULT_SEED + group_index + sum(map(ord, model))).shuffle(group)
            if Counter(int(row["label"]) for row in group) != Counter({0: 100, 1: 100}):
                raise RuntimeError(f"unbalanced view for {model} {group_id}")
            view.extend(group)
            write_jsonl(output_dir / "views" / slug / f"{group_id}.jsonl", group)
        write_jsonl(output_dir / "views" / f"{slug}.jsonl", view)

    raw_hash_counts = Counter(str(row["raw_sha256"]) for row in rows)
    duplicate_raw_groups = sum(1 for count in raw_hash_counts.values() if count > 1)
    summary = {
        "dataset": DATASET,
        "main_revision": MAIN_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "track_count": len(rows),
        "ai_track_count": sum(int(row["label"]) for row in rows),
        "human_track_count": len(humans),
        "model_counts": dict(sorted(Counter(str(row["model"]) for row in rows).items())),
        "cohort_counts": dict(sorted(Counter(str(row["cohort"]) for row in rows).items())),
        "native_sample_rate_counts": dict(sorted(Counter(int(row["original_sample_rate"]) for row in rows).items())),
        "eligible_through_8khz": sum(int(row["eligible_through_8khz"]) for row in rows),
        "eligible_through_10khz": sum(int(row["eligible_through_10khz"]) for row in rows),
        "eligible_through_20khz": sum(int(row["eligible_through_20khz"]) for row in rows),
        "exact_prior_overlap_count": sum(int(row["prior_external_exact_track_overlap"]) for row in rows),
        "duplicate_raw_hash_groups": duplicate_raw_groups,
        "manifest_sha256": digest,
    }
    atomic_write_text(
        output_dir / "materialization_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


def materialize(args: argparse.Namespace) -> None:
    if args.ffmpeg is None or args.ffprobe is None:
        raise RuntimeError("materialize requires --ffmpeg and --ffprobe")
    selection_path = args.output_dir / "frozen_selection.jsonl"
    digest_path = args.output_dir / "frozen_selection.sha256"
    if not selection_path.exists() or not digest_path.exists():
        raise RuntimeError("run freeze before materialize")
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0]
    if sha256_file(selection_path) != expected_digest:
        raise RuntimeError("frozen selection hash mismatch")
    selection = read_jsonl(selection_path)
    by_location = {
        (str(row["source_parquet"]), int(row["source_row_number"])): row for row in selection
    }
    if len(by_location) != 5_500:
        raise RuntimeError("frozen selection source locations are not unique")

    state_path = args.output_dir / "state" / "materialize.jsonl"
    completed = load_state(state_path, args.output_dir)
    print(f"resuming with {len(completed)}/5500 valid clips", flush=True)
    temporary_parent = Path(os.environ.get("TMPDIR", "/tmp"))
    with tempfile.TemporaryDirectory(prefix="aime-500-build-", dir=temporary_parent) as temp_name:
        temp_root = Path(temp_name)
        futures: dict[Future[dict[str, Any]], str] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for shard in parquet_paths(args.parquet_dir):
                parquet = pq.ParquetFile(shard)
                for row_number, batch in enumerate(parquet.iter_batches(batch_size=1, columns=["id", "model", "audio"])):
                    key = (shard.name, row_number)
                    selected = by_location.get(key)
                    if selected is None or str(selected["id"]) in completed:
                        continue
                    source_row = batch.to_pylist()[0]
                    if str(source_row["model"]) != str(selected["model"]) or str(source_row["id"]) != str(selected["hf_original_id"]):
                        raise RuntimeError(f"source row drift at {shard.name}:{row_number}")
                    audio = source_row["audio"]
                    if not isinstance(audio, dict) or not audio.get("bytes"):
                        raise RuntimeError(f"missing embedded audio at {shard.name}:{row_number}")
                    future = executor.submit(
                        process_audio, selected, bytes(audio["bytes"]), str(audio.get("path") or "audio.bin"),
                        args.output_dir, args.ffmpeg, args.ffprobe, args.seed, temp_root,
                    )
                    futures[future] = str(selected["id"])
                    if len(futures) >= args.workers * 2:
                        done = next(as_completed(futures))
                        result = done.result()
                        append_state(state_path, result)
                        completed[str(result["id"])] = result
                        del futures[done]
                        if len(completed) % 50 == 0:
                            print(f"materialized {len(completed)}/5500", flush=True)
            for future in as_completed(futures):
                result = future.result()
                append_state(state_path, result)
                completed[str(result["id"])] = result
                if len(completed) % 50 == 0 or len(completed) == 5_500:
                    print(f"materialized {len(completed)}/5500", flush=True)
    missing = {str(row["id"]) for row in selection} - set(completed)
    if missing:
        raise RuntimeError(f"missing {len(missing)} selected rows after materialization")
    finalize_manifests(args.output_dir, list(completed.values()))


def verify(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.output_dir / "manifest.jsonl")
    errors = []
    for index, row in enumerate(rows, 1):
        clip = args.output_dir / str(row["standardized_relpath"])
        if not clip.exists():
            errors.append(f"missing {clip}")
            continue
        info = sf.info(clip)
        if (info.samplerate, info.channels, info.frames, info.subtype) != (SAMPLE_RATE, 2, EXPECTED_FRAMES, "PCM_16"):
            errors.append(f"invalid audio format {clip}: {info}")
        if sha256_file(clip) != str(row["standardized_sha256"]):
            errors.append(f"hash mismatch {clip}")
        if index % 250 == 0:
            print(f"verified {index}/{len(rows)}", flush=True)
    expected_manifest = (args.output_dir / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(args.output_dir / "manifest.jsonl") != expected_manifest:
        errors.append("manifest hash mismatch")
    if errors:
        raise RuntimeError("verification failed:\n" + "\n".join(errors[:100]))
    print(json.dumps({"verified_tracks": len(rows), "errors": 0}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise RuntimeError("--workers must be positive")
    if args.mode == "freeze":
        freeze_selection(args)
    elif args.mode == "materialize":
        materialize(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
