#!/usr/bin/env python3
"""Freeze a balanced, prompt-matched 500-item Muse inference manifest.

The Muse metadata is streamed because the two training JSONL files are large.
Selection is independent of file order: eligible records are ranked by a salted
SHA-256 key and the 250 smallest keys are retained for each language.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "bolshyC/Muse"
DATASET_REVISION = "b1bf3bf906daab3a896e14f6dea58cc295848452"
DATASET_LICENSE = "MIT"
DEFAULT_SELECTION_SALT = "open-model-spectral-thesis-v1"
DEFAULT_SPLIT_SALT = "open-model-spectral-locked-v1"
LANGUAGE_CODES = {"cn": "zh", "en": "en"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cn", type=Path, required=True)
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=250)
    parser.add_argument("--locked-per-language", type=int, default=50)
    parser.add_argument("--style-sim-min", type=float, default=0.4)
    parser.add_argument("--lyric-window-seconds", type=float, default=30.0)
    parser.add_argument("--selection-salt", default=DEFAULT_SELECTION_SALT)
    parser.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    return parser.parse_args()


def stable_hex(*parts: object) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    # Both generators accept ordinary signed 32-bit seeds.
    return int(stable_hex(*parts)[:8], 16) & 0x7FFF_FFFF


def canonical_section(value: object) -> str:
    name = re.sub(r"[\s_\-]+", " ", str(value).strip().lower())
    name = re.sub(r"\d+", "", name).strip()
    if "pre chorus" in name or "prechorus" in name:
        return "Prechorus"
    if "chorus" in name or "refrain" in name:
        return "Chorus"
    if "verse" in name:
        return "Verse"
    if "intro" in name:
        return "Intro"
    if "outro" in name:
        return "Outro"
    if "bridge" in name:
        return "Bridge"
    if "hook" in name:
        return "Hook"
    if "interlude" in name or "break" in name:
        return "Interlude"
    if "solo" in name:
        return "Solo"
    if "inst" in name:
        return "Inst"
    return "Verse"


def clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def build_windowed_lyrics(sections: Iterable[dict[str, Any]], window_s: float) -> str:
    parts: list[str] = []
    for section in sections:
        text = clean_text(section.get("text", ""))
        if not text:
            continue
        try:
            start_s = float(section.get("startS", 0.0))
        except (TypeError, ValueError):
            start_s = 0.0
        if start_s >= window_s:
            continue
        parts.append(f"[{canonical_section(section.get('section', 'Verse'))}]\n{text}")
    return "\n\n".join(parts).strip()


def source_duration(sections: Iterable[dict[str, Any]]) -> float:
    result = 0.0
    for section in sections:
        try:
            result = max(result, float(section.get("endS", 0.0)))
        except (TypeError, ValueError):
            continue
    return result


def comma_tags(style: str) -> list[str]:
    return [part.strip() for part in style.split(",") if part.strip()]


def candidate_from_record(
    record: dict[str, Any], language_group: str, args: argparse.Namespace
) -> tuple[dict[str, Any] | None, str]:
    try:
        style_sim = float(record.get("style_sim"))
    except (TypeError, ValueError):
        return None, "missing_style_sim"
    if style_sim < args.style_sim_min:
        return None, "style_sim"
    style = clean_text(record.get("style", ""))
    tags = comma_tags(style)
    if not tags:
        return None, "missing_style"
    common_prompt = ", ".join(tags)
    if len(common_prompt) > 512:
        return None, "caption_too_long"
    sections = record.get("sections") or []
    lyrics = build_windowed_lyrics(sections, args.lyric_window_seconds)
    if len(lyrics) < 40:
        return None, "insufficient_lyrics"
    if len(lyrics) > 4096:
        return None, "lyrics_too_long"
    duration_s = source_duration(sections)
    if duration_s < args.lyric_window_seconds:
        return None, "source_too_short"

    song_id = str(record.get("song_id", "")).strip()
    track_index = int(record.get("track_index", 0))
    if not song_id:
        return None, "missing_song_id"
    item_id = f"muse_{language_group}_{song_id}_{track_index}"
    selection_key = stable_hex(args.selection_salt, language_group, song_id, track_index)
    source_json = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    row = {
        "id": item_id,
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_license": DATASET_LICENSE,
        "source_split": "train",
        "source_language_group": language_group,
        "language": LANGUAGE_CODES[language_group],
        "source_song_id": song_id,
        "source_track_index": track_index,
        "source_audio_path": record.get("audio_path", ""),
        "source_duration_s": round(duration_s, 6),
        "style_sim": style_sim,
        "prompt_common": common_prompt,
        "heartmula_tags": ",".join(tags),
        "acestep_caption": common_prompt,
        "lyrics_30s": lyrics,
        "lyric_window_seconds": args.lyric_window_seconds,
        "seed": stable_seed(args.selection_salt, "seed", song_id, track_index),
        "selection_key": selection_key,
        "source_record_sha256": hashlib.sha256(source_json.encode("utf-8")).hexdigest(),
        "_source_record": record,
    }
    return row, "eligible"


def stream_select(
    path: Path, language_group: str, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # A max-heap implemented via the negative integer value keeps the smallest
    # deterministic hash keys without retaining the full metadata corpus.
    heap: list[tuple[int, str, dict[str, Any]]] = []
    counters: Counter[str] = Counter()
    input_hash = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in handle:
            input_hash.update(raw)
            if not raw.strip():
                continue
            counters["rows_seen"] += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                counters["invalid_json"] += 1
                continue
            row, reason = candidate_from_record(record, language_group, args)
            counters[reason] += 1
            if row is None:
                continue
            key_hex = str(row["selection_key"])
            key_int = int(key_hex, 16)
            entry = (-key_int, key_hex, row)
            if len(heap) < args.per_language:
                heapq.heappush(heap, entry)
            elif key_int < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    selected = [item[2] for item in heap]
    selected.sort(key=lambda row: str(row["selection_key"]))
    if len(selected) != args.per_language:
        raise SystemExit(
            f"{language_group}: selected {len(selected)}, expected {args.per_language}; "
            f"filter counts={dict(counters)}"
        )
    metadata = {
        "path": str(path.resolve()),
        "sha256": input_hash.hexdigest(),
        "bytes": path.stat().st_size,
        "filter_counts": dict(sorted(counters.items())),
    }
    return selected, metadata


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not (0 <= args.locked_per_language < args.per_language):
        raise SystemExit("locked-per-language must be in [0, per-language)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    for group, path in (("cn", args.cn), ("en", args.en)):
        selected, input_meta = stream_select(path, group, args)
        locked_ids = {
            str(row["id"])
            for row in sorted(
                selected,
                key=lambda row: stable_hex(args.split_salt, row["id"]),
            )[: args.locked_per_language]
        }
        for row in selected:
            row["split"] = "locked_test" if row["id"] in locked_ids else "development"
        all_rows.extend(selected)
        inputs[group] = input_meta

    all_rows.sort(key=lambda row: (str(row["source_language_group"]), str(row["id"])))
    source_rows = []
    manifest_rows = []
    for row in all_rows:
        source_rows.append(
            {
                "id": row["id"],
                "source_record_sha256": row["source_record_sha256"],
                "record": row.pop("_source_record"),
            }
        )
        manifest_rows.append(row)

    manifest = args.output_dir / "prompt_manifest.jsonl"
    selected_source = args.output_dir / "selected_source_rows.jsonl"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(selected_source, source_rows)
    manifest_hash = file_sha256(manifest)
    source_hash = file_sha256(selected_source)
    (args.output_dir / "prompt_manifest.sha256").write_text(
        f"{manifest_hash}  prompt_manifest.jsonl\n", encoding="utf-8"
    )

    counts = Counter((row["source_language_group"], row["split"]) for row in manifest_rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "note": "Only metadata conditions are used; Muse/Suno audio is not downloaded or analyzed.",
        },
        "selection": {
            "per_language": args.per_language,
            "locked_per_language": args.locked_per_language,
            "style_sim_min": args.style_sim_min,
            "lyric_window_seconds": args.lyric_window_seconds,
            "selection_salt": args.selection_salt,
            "split_salt": args.split_salt,
            "ranking": "smallest salted SHA-256 keys among eligible records",
        },
        "inputs": inputs,
        "artifacts": {
            "prompt_manifest_sha256": manifest_hash,
            "selected_source_rows_sha256": source_hash,
        },
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
    }
    metadata_path = args.output_dir / "prompt_manifest_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
