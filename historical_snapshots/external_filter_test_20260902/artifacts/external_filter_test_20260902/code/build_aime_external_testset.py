#!/usr/bin/env python3
"""Build five frozen, prompt-matched AIME Human-vs-AI external test groups.

The script reads only explicitly downloaded, revision-pinned AIME parquet
shards. Selection uses IDs, model provenance, and exact prompt-tag text; it
never uses detector scores or audio-derived features. Each selected Human
track and AI track shares the same three-tag description, and no description
is reused across groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


SEED = 20_260_902
SECONDS = 30.0
SAMPLE_RATE = 44_100
FRAMES = int(SECONDS * SAMPLE_RATE)
AIME_DATASET = "disco-eth/AIME"
AIME_MAIN_REVISION = "b84d4be5eda830b6eb714998569dba73530f2601"
AIME_PARQUET_REVISION = "1bdacac93127439e361bdd19d575d8b596bca4e3"
MODEL_OFFSETS = {
    "Udio": 4_500,
    "Riffusion": 2_500,
    "Stable Audio v1": 3_500,
    "Stable Audio v2": 4_000,
    "MTG-Jamendo": 6_000,
}
SOURCE_SLUGS = {
    "Udio": "aime_udio",
    "Riffusion": "aime_riffusion",
    "Stable Audio v1": "aime_stable_audio_v1",
    "Stable Audio v2": "aime_stable_audio_v2",
    "MTG-Jamendo": "aime_mtg_jamendo",
}
SOURCE_COUNTS = {
    "Udio": 13,
    "Stable Audio v2": 12,
    "Riffusion": 13,
    "Stable Audio v1": 12,
}
GROUP_ALLOCATIONS = {
    "Udio": (3, 3, 3, 2, 2),
    "Stable Audio v2": (2, 2, 2, 3, 3),
    "Riffusion": (3, 2, 2, 3, 3),
    "Stable Audio v1": (2, 3, 3, 2, 2),
}
FORBIDDEN_CURRENT_FILTER_SOURCES = {
    "FMA",
    "Suno",
    "Suno v3",
    "Suno v3.5",
    "HeartMuLa",
    "ACE-Step",
    "MUSDB18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_description(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return str(value)


def scan_metadata(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        table = pq.read_table(path, columns=["id", "model", "description"])
        for row_number, row in enumerate(table.to_pylist()):
            model = str(row["model"])
            if model not in MODEL_OFFSETS:
                continue
            original_id = int(str(row["id"]))
            prompt_index = original_id - MODEL_OFFSETS[model]
            rows.append(
                {
                    "original_id": f"{original_id:05d}",
                    "model": model,
                    "description": normalize_description(row["description"]),
                    "prompt_index": prompt_index,
                    "parquet_path": str(path.resolve()),
                    "row_number": row_number,
                }
            )
    return rows


def select_pairs(metadata: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    by_model: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metadata:
        model = str(row["model"])
        description = str(row["description"])
        if description in by_model[model]:
            raise RuntimeError(f"duplicate {model} description: {description}")
        by_model[model][description] = row

    rng = random.Random(seed)
    human = by_model["MTG-Jamendo"]
    candidates = {
        model: sorted(set(human) & set(by_model[model])) for model in SOURCE_COUNTS
    }
    for model, descriptions in candidates.items():
        if len(descriptions) < SOURCE_COUNTS[model]:
            raise RuntimeError(
                f"insufficient exact-description matches for {model}: "
                f"{len(descriptions)} < {SOURCE_COUNTS[model]}"
            )
        rng.shuffle(descriptions)

    # Bipartite matching guarantees the requested provider quotas while using
    # every semantic condition only once. Slots with rarer candidate pools are
    # attempted first; seeded candidate order makes the result reproducible.
    slots = [
        (model, slot_number)
        for model, count in SOURCE_COUNTS.items()
        for slot_number in range(count)
    ]
    slots.sort(key=lambda slot: (len(candidates[slot[0]]), slot[0], slot[1]))
    matched_description: dict[str, tuple[str, int]] = {}
    slot_assignment: dict[tuple[str, int], str] = {}

    def augment(slot: tuple[str, int], visited: set[str]) -> bool:
        for description in candidates[slot[0]]:
            if description in visited:
                continue
            visited.add(description)
            incumbent = matched_description.get(description)
            if incumbent is None or augment(incumbent, visited):
                matched_description[description] = slot
                slot_assignment[slot] = description
                return True
        return False

    for slot in slots:
        if not augment(slot, set()):
            counts = {model: len(values) for model, values in candidates.items()}
            raise RuntimeError(f"cannot satisfy disjoint provider quotas; candidate counts={counts}")
    selected: dict[str, list[str]] = {
        model: [slot_assignment[(model, slot_number)] for slot_number in range(count)]
        for model, count in SOURCE_COUNTS.items()
    }

    pairs_by_model: dict[str, list[dict[str, object]]] = {}
    for model, descriptions in selected.items():
        current = []
        for description in descriptions:
            ai = by_model[model][description]
            real = human[description]
            condition_id = hashlib.sha256(description.encode("utf-8")).hexdigest()[:16]
            current.append(
                {
                    "model": model,
                    "condition_id": condition_id,
                    "description": description,
                    "human": real,
                    "ai": ai,
                }
            )
        rng.shuffle(current)
        pairs_by_model[model] = current

    groups: list[list[dict[str, object]]] = [[] for _ in range(5)]
    for model, allocation in GROUP_ALLOCATIONS.items():
        offset = 0
        for group_index, count in enumerate(allocation):
            groups[group_index].extend(pairs_by_model[model][offset : offset + count])
            offset += count
        if offset != len(pairs_by_model[model]):
            raise RuntimeError(f"allocation error for {model}")

    output = []
    serial = 0
    for group_index, group in enumerate(groups, 1):
        if len(group) != 10:
            raise RuntimeError(f"group {group_index} has {len(group)} pairs")
        rng.shuffle(group)
        for pair in group:
            serial += 1
            output.append({**pair, "group_id": f"group_{group_index:02d}", "pair_id": f"pair_{serial:03d}"})
    return output


def extract_audio_blob(parquet_path: Path, wanted_ids: set[str]) -> dict[str, tuple[bytes, str]]:
    found: dict[str, tuple[bytes, str]] = {}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=1, columns=["id", "audio"]):
        row = batch.to_pylist()[0]
        original_id = str(row["id"])
        if original_id not in wanted_ids:
            continue
        audio = row["audio"]
        if not isinstance(audio, dict) or not audio.get("bytes"):
            raise RuntimeError(f"embedded audio bytes missing for {original_id} in {parquet_path}")
        found[original_id] = (bytes(audio["bytes"]), str(audio.get("path") or "audio.bin"))
    missing = wanted_ids - set(found)
    if missing:
        raise RuntimeError(f"missing rows in {parquet_path}: {sorted(missing)}")
    return found


def suffix_for(blob: bytes, hinted_path: str) -> str:
    suffix = Path(hinted_path).suffix.lower()
    if suffix in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
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


def probe(path: Path, ffprobe: Path) -> dict[str, object]:
    command = [
        str(ffprobe), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name:format=duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = payload["streams"][0]
    return {
        "original_sample_rate": int(stream["sample_rate"]),
        "original_channels": int(stream["channels"]),
        "original_codec": str(stream["codec_name"]),
        "original_duration_s": float(payload["format"]["duration"]),
    }


def crop_start(duration: float, seed: int, original_id: str) -> float:
    available = max(0.0, duration - SECONDS)
    if available <= 0.05:
        return 0.0
    return round(random.Random(seed + int(original_id) * 1009).uniform(0.0, available), 6)


def standardize(source: Path, destination: Path, start: float, ffmpeg: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.flac")
    command = [
        str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{SECONDS:.3f}",
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-af", f"apad=whole_dur={SECONDS:.3f}",
        "-ac", "2", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
        "-compression_level", "8", "-y", str(temporary),
    ]
    subprocess.run(command, check=True)
    info = sf.info(temporary)
    if info.samplerate != SAMPLE_RATE or info.channels != 2 or info.frames != FRAMES:
        raise RuntimeError(f"invalid standardized output {temporary}: {info}")
    os.replace(temporary, destination)


def track_name(row: dict[str, object]) -> str:
    return f"{row['class_name']}_{row['source']}_{row['id']}"


def main() -> None:
    args = parse_args()
    parquet_paths = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"no parquet shards under {args.parquet_dir}")
    metadata = scan_metadata(parquet_paths)
    pairs = select_pairs(metadata, args.seed)

    selected_metadata: dict[str, dict[str, object]] = {}
    draft_rows: list[dict[str, object]] = []
    for pair in pairs:
        for label, key in ((0, "human"), (1, "ai")):
            source_row = pair[key]
            assert isinstance(source_row, dict)
            model = str(source_row["model"])
            if model in FORBIDDEN_CURRENT_FILTER_SOURCES:
                raise RuntimeError(f"forbidden current-filter source selected: {model}")
            original_id = str(source_row["original_id"])
            row = {
                "id": f"aime_{original_id}",
                "label": label,
                "class_name": "human" if label == 0 else "ai",
                "source": SOURCE_SLUGS[model],
                "model": model,
                "split": "external_test",
                "group_id": pair["group_id"],
                "pair_id": pair["pair_id"],
                "condition_id": pair["condition_id"],
                "prompt_index": source_row["prompt_index"],
                "matched_human_prompt_index": pair["human"]["prompt_index"],
                "matched_ai_prompt_index": pair["ai"]["prompt_index"],
                "description": source_row["description"],
                "hf_dataset": AIME_DATASET,
                "hf_main_revision": AIME_MAIN_REVISION,
                "hf_parquet_revision": AIME_PARQUET_REVISION,
                "hf_original_id": original_id,
                "source_parquet": Path(str(source_row["parquet_path"])).name,
                "source_row_number": source_row["row_number"],
                "selection_uses_audio_features": False,
                "selection_uses_detector_scores": False,
            }
            draft_rows.append(row)
            selected_metadata[original_id] = source_row

    by_parquet: dict[Path, set[str]] = defaultdict(set)
    for original_id, row in selected_metadata.items():
        by_parquet[Path(str(row["parquet_path"]))].add(original_id)
    blobs: dict[str, tuple[bytes, str]] = {}
    for number, (path, wanted) in enumerate(sorted(by_parquet.items()), 1):
        blobs.update(extract_audio_blob(path, wanted))
        print(f"extracted shard {number}/{len(by_parquet)}: {path.name}", flush=True)

    raw_dir = args.output_dir / "raw"
    clips_dir = args.output_dir / "clips"
    qc_rows = []
    final_rows = []
    raw_hashes: dict[str, str] = {}
    for number, row in enumerate(draft_rows, 1):
        original_id = str(row["hf_original_id"])
        blob, hinted_path = blobs[original_id]
        name = track_name(row)
        raw_path = raw_dir / f"{name}{suffix_for(blob, hinted_path)}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists() or raw_path.read_bytes() != blob:
            temporary = raw_path.with_suffix(raw_path.suffix + ".part")
            temporary.write_bytes(blob)
            os.replace(temporary, raw_path)
        raw_digest = sha256(raw_path)
        if raw_digest in raw_hashes and raw_hashes[raw_digest] != name:
            raise RuntimeError(f"exact duplicate audio: {name} and {raw_hashes[raw_digest]}")
        raw_hashes[raw_digest] = name
        audio_info = probe(raw_path, args.ffprobe)
        start = crop_start(float(audio_info["original_duration_s"]), args.seed, original_id)
        clip_path = clips_dir / f"{name}.flac"
        if not clip_path.exists():
            standardize(raw_path, clip_path, start, args.ffmpeg)
        else:
            info = sf.info(clip_path)
            if info.samplerate != SAMPLE_RATE or info.channels != 2 or info.frames != FRAMES:
                raise RuntimeError(f"invalid existing output {clip_path}: {info}")
        clip_digest = sha256(clip_path)
        eligible = int(int(audio_info["original_sample_rate"]) >= 40_000)
        final = {
            **row,
            "crop_start_s": start,
            "original_sample_rate": audio_info["original_sample_rate"],
            "original_channels": audio_info["original_channels"],
            "original_codec": audio_info["original_codec"],
            "original_duration_s": audio_info["original_duration_s"],
            "full_band_eligible": eligible,
            "raw_sha256": raw_digest,
            "standardized_sha256": clip_digest,
        }
        final_rows.append(final)
        qc_rows.append(
            {
                "track": name,
                "group_id": row["group_id"],
                "pair_id": row["pair_id"],
                "label": row["label"],
                "source": row["source"],
                "original_sample_rate": audio_info["original_sample_rate"],
                "original_channels": audio_info["original_channels"],
                "original_codec": audio_info["original_codec"],
                "original_duration_s": audio_info["original_duration_s"],
                "crop_start_s": start,
                "full_band_eligible": eligible,
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": raw_digest,
                "standardized_bytes": clip_path.stat().st_size,
                "standardized_sha256": clip_digest,
            }
        )
        if number == 1 or number % 10 == 0 or number == len(draft_rows):
            print(f"standardized {number}/{len(draft_rows)}", flush=True)

    rng = random.Random(args.seed + 77)
    ordered_rows = []
    for group_index in range(1, 6):
        group_id = f"group_{group_index:02d}"
        group = [row for row in final_rows if row["group_id"] == group_id]
        rng.shuffle(group)
        if Counter(int(row["label"]) for row in group) != Counter({0: 10, 1: 10}):
            raise RuntimeError(f"unbalanced {group_id}")
        ordered_rows.extend(group)
        write_jsonl(args.output_dir / "groups" / f"{group_id}.jsonl", group)
    write_jsonl(args.output_dir / "manifest.jsonl", ordered_rows)
    write_csv(args.output_dir / "audio_qc.csv", qc_rows)
    manifest_digest = sha256(args.output_dir / "manifest.jsonl")
    (args.output_dir / "manifest.sha256").write_text(
        f"{manifest_digest}  manifest.jsonl\n", encoding="utf-8"
    )
    summary = {
        "dataset": AIME_DATASET,
        "main_revision": AIME_MAIN_REVISION,
        "parquet_revision": AIME_PARQUET_REVISION,
        "seed": args.seed,
        "selection_basis": "model provenance, exact shared prompt-tag description, and seeded RNG only",
        "prompt_description_matched": True,
        "prompt_description_reused": False,
        "groups": 5,
        "pairs_per_group": 10,
        "tracks": len(ordered_rows),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in ordered_rows).items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in ordered_rows).items())),
        "native_sample_rates": dict(sorted(Counter(int(row["original_sample_rate"]) for row in ordered_rows).items())),
        "full_band_eligible_tracks": sum(int(row["full_band_eligible"]) for row in ordered_rows),
        "forbidden_current_filter_sources": sorted(FORBIDDEN_CURRENT_FILTER_SOURCES),
        "manifest_sha256": manifest_digest,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
