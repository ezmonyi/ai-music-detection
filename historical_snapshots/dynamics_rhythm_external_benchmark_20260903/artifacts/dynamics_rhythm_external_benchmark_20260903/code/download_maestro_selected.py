#!/usr/bin/env python3
"""Range-download only frozen MAESTRO members from the official archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip


AUDIO_ARCHIVE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MIDI_ARCHIVE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_member(names: set[str], relative: str) -> str:
    candidates = (relative, f"maestro-v3.0.0/{relative}")
    for candidate in candidates:
        if candidate in names:
            return candidate
    suffix_matches = [name for name in names if name.endswith("/" + relative)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise KeyError(f"Could not uniquely resolve {relative}; matches={suffix_matches[:5]}")


def fetch_archive(
    archive_url: str,
    relative_paths: list[str],
    output_root: Path,
    kind: str,
    workers: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with RemoteZip(archive_url) as archive:
        info_by_name = {info.filename: info for info in archive.infolist()}
        names = set(info_by_name)
        tasks = [
            (index, relative, resolve_member(names, relative))
            for index, relative in enumerate(relative_paths, start=1)
        ]

    def fetch_one(task: tuple[int, str, str]) -> tuple[int, dict[str, object]]:
        index, relative, member = task
        info = info_by_name[member]
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status = "existing_verified_size"
        if not output_path.exists() or output_path.stat().st_size != info.file_size:
            part = output_path.with_name(output_path.name + ".part")
            # RemoteZip's HTTP stream is stateful, so each worker owns its own
            # reader.  Four bounded workers cut request latency without sharing
            # a seekable stream or changing any selected member.
            with RemoteZip(archive_url) as worker_archive:
                with worker_archive.open(member) as source, part.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                    destination.flush()
                    os.fsync(destination.fileno())
            if part.stat().st_size != info.file_size:
                raise IOError(
                    f"Size mismatch for {relative}: {part.stat().st_size} != {info.file_size}"
                )
            os.replace(part, output_path)
            status = "downloaded"
        return index, {
            "kind": kind,
            "relative_path": relative,
            "archive_member": member,
            "archive_crc32": f"{info.CRC:08x}",
            "archive_file_size": info.file_size,
            "archive_compressed_size": info.compress_size,
            "local_path": str(output_path.resolve()),
            "local_size": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "status": status,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, task): task for task in tasks}
        for future in as_completed(futures):
            index, record = future.result()
            records.append(record)
            print(
                f"[{kind} {index:03d}/{len(relative_paths):03d}] "
                f"{record['status']} {record['relative_path']} "
                f"{int(record['archive_file_size']) / 1e6:.1f} MB",
                flush=True,
            )
    return sorted(records, key=lambda record: str(record["relative_path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with args.recording_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50 or any(row["status"] != "frozen_selected" for row in rows):
        raise RuntimeError("Recording manifest is not the frozen 50-recording selection")

    audio_paths = [row["audio_filename"] for row in rows]
    midi_paths = [row["midi_filename"] for row in rows]
    records = []
    if not 1 <= args.workers <= 8:
        raise RuntimeError("--workers must be between 1 and 8")
    records.extend(
        fetch_archive(AUDIO_ARCHIVE, audio_paths, args.output_root / "audio", "audio", args.workers)
    )
    records.extend(
        fetch_archive(MIDI_ARCHIVE, midi_paths, args.output_root / "midi", "midi", args.workers)
    )
    result = {
        "audio_archive": AUDIO_ARCHIVE,
        "midi_archive": MIDI_ARCHIVE,
        "n_audio": sum(record["kind"] == "audio" for record in records),
        "n_midi": sum(record["kind"] == "midi" for record in records),
        "total_local_bytes": sum(int(record["local_size"]) for record in records),
        "records": records,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
