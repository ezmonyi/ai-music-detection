#!/usr/bin/env python3
"""Acquire and verify a deterministic MIR-1K subset for V/F0 validation only.

The script lists the two required directories at a pinned Hugging Face revision,
selects singer-balanced and song-round-robin clips, downloads only the selected
WAV/.pv pairs through ``hf download``, and writes hashes plus audio/label timing
diagnostics.  It never downloads an archive and never supplies classifier data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ID = "AnhP/Mir-1k-use-DJCM-training"
REVISION = "d6b7565ab4d85c0fa752eeb3ac63618c94620edd"
DEFAULT_SEED = "mir1k-v-pitch-validation-v1-20260907"
FRAME_PERIOD_SEC = 0.02
FIRST_FRAME_TIME_SEC = 0.02
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 2
EXPECTED_SAMPLE_WIDTH_BYTES = 2
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=here.parent / "external_validation" / "mir1k",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--hf-binary", default="hf")
    parser.add_argument("--execute", action="store_true",
                        help="Download selected pairs; without this flag only write the plan")
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_key(seed: str, *values: str) -> str:
    return hashlib.sha256("|".join((seed, *values)).encode("utf-8")).hexdigest()


def hub_tree_url(directory: str) -> str:
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in REPO_ID.split("/"))
    encoded_directory = "/".join(urllib.parse.quote(part, safe="") for part in directory.split("/"))
    return (
        f"https://huggingface.co/api/datasets/{encoded_repo}/tree/{REVISION}/"
        f"{encoded_directory}?recursive=true&expand=false&limit=1000"
    )


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "pitch-validation-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_inventory(directory: str, suffix: str) -> list[dict[str, Any]]:
    payload = fetch_json(hub_tree_url(directory))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Hub tree response for {directory}")
    rows = [row for row in payload if row.get("type") == "file" and row.get("path", "").endswith(suffix)]
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 {suffix} files at pinned revision, found {len(rows)}")
    return rows


def parse_stem(stem: str) -> tuple[str, str, str]:
    try:
        singer, song, clip = stem.rsplit("_", 2)
    except ValueError as error:
        raise ValueError(f"Unexpected MIR-1K clip name: {stem}") from error
    if not singer or not song.isdigit() or not clip.isdigit():
        raise ValueError(f"Unexpected MIR-1K clip name: {stem}")
    return singer, song, clip


def allocate_singer_targets(stems: list[str], count: int, seed: str) -> dict[str, int]:
    by_singer: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        singer, _, _ = parse_stem(stem)
        by_singer[singer].append(stem)
    singers = sorted(by_singer)
    if count < len(singers):
        raise ValueError(f"count must be at least the {len(singers)} singers")
    if count > len(stems):
        raise ValueError("count exceeds paired inventory")
    base, remainder = divmod(count, len(singers))
    targets = {singer: min(base, len(by_singer[singer])) for singer in singers}
    remaining = count - sum(targets.values())
    order = sorted(singers, key=lambda singer: stable_key(seed, "singer-extra", singer))
    cursor = 0
    while remaining:
        singer = order[cursor % len(order)]
        if targets[singer] < len(by_singer[singer]):
            targets[singer] += 1
            remaining -= 1
        cursor += 1
        if cursor > count * len(order) * 2:
            raise RuntimeError("Could not allocate singer targets")
    return targets


def choose_balanced(stems: list[str], count: int, seed: str) -> list[dict[str, Any]]:
    targets = allocate_singer_targets(stems, count, seed)
    by_singer_song: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for stem in stems:
        singer, song, _ = parse_stem(stem)
        by_singer_song[singer][song].append(stem)
    selected: list[dict[str, Any]] = []
    for singer in sorted(by_singer_song):
        songs = sorted(
            by_singer_song[singer], key=lambda song: stable_key(seed, "song", singer, song)
        )
        queues = {
            song: sorted(
                by_singer_song[singer][song],
                key=lambda stem: stable_key(seed, "clip", singer, song, stem),
            )
            for song in songs
        }
        target = targets[singer]
        round_index = 0
        singer_selected: list[str] = []
        while len(singer_selected) < target:
            progressed = False
            for song in songs:
                if round_index < len(queues[song]) and len(singer_selected) < target:
                    singer_selected.append(queues[song][round_index])
                    progressed = True
            if not progressed:
                raise RuntimeError(f"Could not fill target for singer {singer}")
            round_index += 1
        for singer_rank, stem in enumerate(singer_selected, 1):
            _, song, clip = parse_stem(stem)
            selected.append({
                "stem": stem, "singer": singer, "song_id": int(song), "clip_id": int(clip),
                "singer_target": target, "singer_selection_rank": singer_rank,
            })
    if len(selected) != count or len({row["stem"] for row in selected}) != count:
        raise AssertionError("Balanced selection did not produce exact unique count")
    return selected


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command[:8])}\n{result.stderr[-4000:]}"
        )
    return result


def verify_revision(hf_binary: str) -> tuple[dict[str, Any], str]:
    result = run_checked([
        hf_binary, "datasets", "info", REPO_ID, "--revision", REVISION,
        "--expand", "sha", "--format", "json",
    ])
    payload = json.loads(result.stdout)
    if payload.get("sha") != REVISION:
        raise RuntimeError(f"Hub resolved revision {payload.get('sha')}, expected {REVISION}")
    version = run_checked([hf_binary, "version", "--format", "json"]).stdout.strip()
    return payload, version


def download_selected(
    hf_binary: str, output_dir: Path, remote_paths: list[str], max_workers: int,
) -> tuple[list[str], str]:
    command = [
        hf_binary, "download", REPO_ID, *remote_paths,
        "--type", "dataset", "--revision", REVISION,
        "--local-dir", str(output_dir), "--max-workers", str(max_workers), "--format", "json",
    ]
    result = run_checked(command)
    return command, result.stdout.strip()


def load_pitch_labels(path: Path) -> list[float]:
    values: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(f"Invalid pitch value {path}:{line_number}: {text!r}") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Pitch labels must be finite nonnegative MIDI/zero values: {path}")
        values.append(value)
    if not values:
        raise ValueError(f"Pitch label file is empty: {path}")
    return values


def verify_pair(
    row: dict[str, Any], output_dir: Path, wav_inventory: dict[str, dict[str, Any]],
    pitch_inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stem = row["stem"]
    wav_remote = f"MIR-1K/Wavfile/{stem}.wav"
    pitch_remote = f"MIR-1K/PitchLabel/{stem}.pv"
    wav_path = output_dir / wav_remote
    pitch_path = output_dir / pitch_remote
    if not wav_path.is_file() or not pitch_path.is_file():
        raise FileNotFoundError(f"Selected pair is incomplete: {stem}")
    with wave.open(str(wav_path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frames = audio.getnframes()
        compression = audio.getcomptype()
    labels = load_pitch_labels(pitch_path)
    duration = frames / sample_rate
    label_last_time = len(labels) * FRAME_PERIOD_SEC
    delta = duration - label_last_time
    expected_label_count = math.floor(duration / FRAME_PERIOD_SEC) - 1
    wav_sha256 = sha256_file(wav_path)
    pitch_sha256 = sha256_file(pitch_path)
    pitch_git_sha1 = git_blob_sha1(pitch_path)
    wav_lfs = wav_inventory[wav_remote].get("lfs", {})
    passed = (
        channels == EXPECTED_CHANNELS
        and sample_rate == EXPECTED_SAMPLE_RATE
        and sample_width == EXPECTED_SAMPLE_WIDTH_BYTES
        and compression == "NONE"
        # The pinned MIR-1K originals consistently omit time zero and the final
        # partial/full analysis position: N=floor(duration/20 ms)-1.  Do not
        # substitute the unrelated MIREX09 readme's interpolated 10-ms grid.
        and len(labels) == expected_label_count
        and wav_path.stat().st_size == int(wav_inventory[wav_remote]["size"])
        and pitch_path.stat().st_size == int(pitch_inventory[pitch_remote]["size"])
        and (not wav_lfs.get("oid") or wav_sha256 == wav_lfs["oid"])
        and pitch_git_sha1 == pitch_inventory[pitch_remote].get("oid")
    )
    if not passed:
        raise RuntimeError(f"Audio/label verification failed for {stem}; duration delta={delta}")
    voiced = [value for value in labels if value > 0]
    return {
        **row,
        "repository": REPO_ID, "revision": REVISION,
        "wav_remote_path": wav_remote, "pitch_remote_path": pitch_remote,
        "wav_local_path": str(wav_path.resolve()), "pitch_local_path": str(pitch_path.resolve()),
        "wav_bytes": wav_path.stat().st_size, "pitch_bytes": pitch_path.stat().st_size,
        "wav_sha256": wav_sha256, "pitch_sha256": pitch_sha256,
        "hub_wav_lfs_sha256": wav_lfs.get("oid", ""),
        "hub_wav_git_oid": wav_inventory[wav_remote].get("oid", ""),
        "hub_pitch_git_oid": pitch_inventory[pitch_remote].get("oid", ""),
        "pitch_git_blob_sha1": pitch_git_sha1,
        "sample_rate_hz": sample_rate, "channels": channels,
        "channel_0_semantics": "accompaniment", "channel_1_semantics": "singing_voice",
        "sample_width_bytes": sample_width, "pcm_compression": compression,
        "audio_frames": frames, "audio_duration_sec": duration,
        "pitch_label_count": len(labels), "pitch_unit": "MIDI semitone; 0=unvoiced",
        "expected_pitch_label_count_from_audio": expected_label_count,
        "pitch_label_count_rule": "floor(audio_duration_sec / 0.02) - 1",
        "pitch_frame_period_sec": FRAME_PERIOD_SEC,
        "pitch_first_frame_time_sec": FIRST_FRAME_TIME_SEC,
        "pitch_last_frame_time_sec": label_last_time,
        "audio_minus_last_label_time_sec": delta,
        "voiced_label_count": len(voiced), "voiced_fraction": len(voiced) / len(labels),
        "voiced_midi_min": min(voiced) if voiced else "",
        "voiced_midi_max": max(voiced) if voiced else "",
        "verification_status": "passed",
        "use_scope": "external_V_F0_measurement_validation_only_never_classifier",
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    hf_path = shutil.which(args.hf_binary)
    if hf_path is None:
        raise RuntimeError(f"hf CLI not found: {args.hf_binary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_readme = args.output_dir / "MIR-1K" / "readme.txt"
    readme_hash_before = sha256_file(existing_readme) if existing_readme.is_file() else None

    repo_info, hf_version = verify_revision(hf_path)
    wav_rows = fetch_inventory("MIR-1K/Wavfile", ".wav")
    pitch_rows = fetch_inventory("MIR-1K/PitchLabel", ".pv")
    wav_inventory = {row["path"]: row for row in wav_rows}
    pitch_inventory = {row["path"]: row for row in pitch_rows}
    wav_stems = {Path(path).stem for path in wav_inventory}
    pitch_stems = {Path(path).stem for path in pitch_inventory}
    if wav_stems != pitch_stems:
        raise RuntimeError(
            f"Pinned mirror has unpaired identities: wav-only={len(wav_stems-pitch_stems)}, "
            f"pitch-only={len(pitch_stems-wav_stems)}"
        )
    selected = choose_balanced(sorted(wav_stems), args.count, args.seed)
    selected_paths = [
        path for row in selected for path in (
            f"MIR-1K/Wavfile/{row['stem']}.wav", f"MIR-1K/PitchLabel/{row['stem']}.pv"
        )
    ]
    plan = {
        "schema_version": SCHEMA_VERSION, "status": "frozen_selection",
        "repository": REPO_ID, "revision": REVISION, "resolved_revision": repo_info["sha"],
        "count": args.count, "seed": args.seed,
        "selection_algorithm": (
            "equal floor count per singer; deterministic hash-ordered remainder; "
            "within singer hash-ordered song round-robin and hash-ordered clips"
        ),
        "paired_inventory_count": len(wav_stems),
        "paired_inventory_id_set_sha256": canonical_hash(sorted(wav_stems)),
        "wav_tree_response_sha256": canonical_hash(wav_rows),
        "pitch_tree_response_sha256": canonical_hash(pitch_rows),
        "selected_id_set_sha256": canonical_hash(sorted(row["stem"] for row in selected)),
        "selected": selected,
        "archive_downloaded": False,
        "use_scope": "external_V_F0_measurement_validation_only_never_classifier",
    }
    (args.output_dir / "selection_plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not args.execute:
        print(json.dumps({"status": "planned", "count": args.count,
                          "output": str(args.output_dir / 'selection_plan.json')}, indent=2))
        return

    command, command_stdout = download_selected(
        hf_path, args.output_dir, selected_paths, args.max_workers
    )
    readme_hash_after = sha256_file(existing_readme) if existing_readme.is_file() else None
    if readme_hash_before != readme_hash_after:
        raise RuntimeError("Pre-existing MIR-1K/readme.txt changed during subset acquisition")
    selected_stems = {row["stem"] for row in selected}
    actual_wav_stems = {
        path.stem for path in (args.output_dir / "MIR-1K" / "Wavfile").glob("*.wav")
    }
    actual_pitch_stems = {
        path.stem for path in (args.output_dir / "MIR-1K" / "PitchLabel").glob("*.pv")
    }
    if actual_wav_stems != selected_stems or actual_pitch_stems != selected_stems:
        raise RuntimeError(
            "Local validation directories contain files outside or missing from the frozen selection: "
            f"wav_extra={sorted(actual_wav_stems-selected_stems)}, "
            f"pitch_extra={sorted(actual_pitch_stems-selected_stems)}, "
            f"wav_missing={sorted(selected_stems-actual_wav_stems)}, "
            f"pitch_missing={sorted(selected_stems-actual_pitch_stems)}"
        )
    verified = [verify_pair(row, args.output_dir, wav_inventory, pitch_inventory) for row in selected]
    write_csv(args.output_dir / "selection_manifest.csv", verified)
    selected_singer_counts: dict[str, int] = defaultdict(int)
    selected_song_counts: dict[str, set[int]] = defaultdict(set)
    for row in verified:
        selected_singer_counts[row["singer"]] += 1
        selected_song_counts[row["singer"]].add(int(row["song_id"]))
    summary = {
        **{key: value for key, value in plan.items() if key != "selected"},
        "status": "verified",
        "verified_pairs": len(verified),
        "verification_failures": 0,
        "selected_total_bytes": sum(int(row["wav_bytes"]) + int(row["pitch_bytes"]) for row in verified),
        "singer_count": len(selected_singer_counts),
        "per_singer_clip_counts": dict(sorted(selected_singer_counts.items())),
        "per_singer_distinct_song_counts": {
            singer: len(songs) for singer, songs in sorted(selected_song_counts.items())
        },
        "sample_rate_counts": {str(EXPECTED_SAMPLE_RATE): len(verified)},
        "channel_counts": {str(EXPECTED_CHANNELS): len(verified)},
        "pitch_frame_period_sec": FRAME_PERIOD_SEC,
        "pitch_first_frame_time_sec": FIRST_FRAME_TIME_SEC,
        "audio_minus_last_label_time_sec_min": min(
            float(row["audio_minus_last_label_time_sec"]) for row in verified
        ),
        "audio_minus_last_label_time_sec_max": max(
            float(row["audio_minus_last_label_time_sec"]) for row in verified
        ),
        "readme_preserved_sha256": readme_hash_after,
        "readme_identity_warning": (
            "This file describes MIREX09 (374 recordings), not MIR-1K; it is preserved "
            "as mirror evidence and is not used for MIR-1K timing."
        ),
        "license_status": (
            "unresolved: pinned mirror has no dataset card or explicit license file/tag; "
            "do not infer a license from public download access or from f0-mlf code's MIT license"
        ),
        "hf_cli": {"resolved_binary": hf_path, "version_output": hf_version},
        "download_command": command,
        "download_stdout": command_stdout,
        "manifest_csv": str((args.output_dir / "selection_manifest.csv").resolve()),
    }
    manifest_path = args.output_dir / "selection_summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "verified", "pairs": len(verified),
        "singers": len(selected_singer_counts), "bytes": summary["selected_total_bytes"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
