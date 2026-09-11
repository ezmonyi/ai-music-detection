#!/usr/bin/env python3
"""Verify per-file MAESTRO transport bytes against official ZIP metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zlib
from pathlib import Path

from remotezip import RemoteZip

from download_maestro_selected import AUDIO_ARCHIVE, MIDI_ARCHIVE, resolve_member


def digests(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return sha.hexdigest(), f"{crc & 0xFFFFFFFF:08x}"


def verify_kind(
    archive_url: str,
    relative_paths: list[str],
    source_root: Path,
    kind: str,
) -> list[dict[str, object]]:
    with RemoteZip(archive_url) as archive:
        infos = {info.filename: info for info in archive.infolist()}
    names = set(infos)
    records = []
    for index, relative in enumerate(relative_paths, start=1):
        member = resolve_member(names, relative)
        info = infos[member]
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        sha256, crc32 = digests(path)
        size_match = path.stat().st_size == info.file_size
        crc_match = crc32 == f"{info.CRC:08x}"
        record = {
            "kind": kind,
            "relative_path": relative,
            "official_archive": archive_url,
            "official_archive_member": member,
            "official_size": info.file_size,
            "official_crc32": f"{info.CRC:08x}",
            "transport_size": path.stat().st_size,
            "transport_crc32": crc32,
            "transport_sha256": sha256,
            "size_match": size_match,
            "crc32_match": crc_match,
            "accepted": bool(size_match and crc_match),
        }
        if not record["accepted"]:
            raise RuntimeError(f"Official ZIP mismatch: {record}")
        records.append(record)
        print(f"[{kind} {index:03d}/{len(relative_paths):03d}] verified {relative}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--transport-repo", required=True)
    parser.add_argument("--transport-revision", required=True)
    args = parser.parse_args()
    with args.recording_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50 or any(row["status"] != "frozen_selected" for row in rows):
        raise RuntimeError("Recording manifest is not the frozen 50-recording selection")

    records = verify_kind(
        AUDIO_ARCHIVE,
        [row["audio_filename"] for row in rows],
        args.source_root,
        "audio",
    )
    records.extend(
        verify_kind(
            MIDI_ARCHIVE,
            [row["midi_filename"] for row in rows],
            args.source_root,
            "midi",
        )
    )
    result = {
        "transport_repo": args.transport_repo,
        "transport_revision": args.transport_revision,
        "identity_authority": "official MAESTRO v3.0.0 Google ZIP central-directory metadata",
        "n_files": len(records),
        "n_accepted": sum(bool(record["accepted"]) for record in records),
        "all_accepted": all(bool(record["accepted"]) for record in records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
