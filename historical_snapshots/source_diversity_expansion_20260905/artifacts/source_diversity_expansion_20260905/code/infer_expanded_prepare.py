#!/usr/bin/env python3
"""Normalize an incremental materialization manifest and freeze inference shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PATH_FIELDS = ("standardized_path", "audio_path", "view_10s_path", "source_audio_path")
SR_FIELDS = ("native_sr", "native_sample_rate_hz", "original_sample_rate")
DURATION_FIELDS = ("duration_sec", "duration_view", "duration", "view_10s_duration_s", "native_duration_s", "original_duration_s")
SR = 44_100


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_records(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    latest: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = str(record.get("item_id") or record.get("track") or "")
            if not item_id:
                raise RuntimeError(f"Missing item_id at {path}:{line_number}")
            latest[item_id] = record
    return list(latest.values())


def first(row: dict[str, object], names: tuple[str, ...], default: object = "") -> object:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return default


def normalize(row: dict[str, object]) -> dict[str, object] | None:
    if str(row.get("available", "1")).strip().lower() in {"0", "false", "no", "pending", ""}:
        return None
    if str(row.get("status", "success")).lower() not in {"success", "complete", "completed", "ok"}:
        return None
    item_id = str(first(row, ("item_id", "track", "id")))
    path = Path(str(first(row, PATH_FIELDS))).expanduser()
    if not item_id or str(path) in {"", "."}:
        raise RuntimeError(f"Missing id/path: {row}")
    label_raw = str(first(row, ("label", "class_name"))).lower()
    label = 1 if label_raw in {"1", "ai", "generated", "synthetic"} else 0 if label_raw in {"0", "human", "real"} else None
    if label is None:
        raise RuntimeError(f"Unknown label {label_raw!r} for {item_id}")
    source = str(first(row, ("source_id", "source", "model"), "unknown"))
    native_sr_raw = first(row, SR_FIELDS, "")
    duration_raw = first(row, DURATION_FIELDS, "")
    return {
        "item_id": item_id, "label": label, "source_id": source,
        "group_id": str(first(row, ("group_id", "condition_id"), item_id)),
        "standardized_path": str(path),
        "native_sr": int(float(native_sr_raw)) if native_sr_raw not in (None, "") else "",
        "duration": float(duration_raw) if duration_raw not in (None, "") else "",
        "audio_offset_s": float(row.get("audio_offset_s") or 0.0),
        "requires_crop": int(float(row.get("requires_crop") or 0)),
        "role": str(row.get("role", "")),
        "evaluation_allowed": str(row.get("evaluation_allowed", "")),
        "source_manifest_status": str(row.get("status", "success")),
    }


def exact_view(row: dict[str, object], root: Path, duration: float) -> dict[str, object]:
    import soundfile as sf
    source = Path(str(row["standardized_path"]))
    if not source.is_file():
        return row
    info = sf.info(source)
    needs_crop = bool(int(row["requires_crop"])) or abs(info.duration-duration) > 0.02 or float(row["audio_offset_s"]) > 0
    if not needs_crop:
        return row
    destination = root / str(row["source_id"]) / f"{row['item_id']}.flac"
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(round(duration * SR))
    if destination.exists():
        out = sf.info(destination)
        if out.samplerate == SR and out.channels == 2 and out.frames == expected:
            return {**row, "standardized_path": str(destination), "duration": duration, "requires_crop": 0}
        raise RuntimeError(f"Conflicting existing view: {destination}")
    with sf.SoundFile(source) as handle:
        handle.seek(int(round(float(row["audio_offset_s"]) * handle.samplerate)))
        audio = handle.read(int(round(duration * handle.samplerate)), dtype="float32", always_2d=True)
        source_sr = int(handle.samplerate)
    if source_sr != SR or audio.shape[1] != 2:
        raise RuntimeError(f"Crop input is not frozen 44.1k stereo: {source}")
    if len(audio) != expected:
        raise RuntimeError(f"Crop cannot supply exact {duration:g}s: {source} offset={row['audio_offset_s']}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".flac", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        sf.write(temporary, audio, SR, format="FLAC", subtype="PCM_16")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {**row, "standardized_path": str(destination), "duration": duration, "requires_crop": 0}


def allinone_complete(root: Path, item_id: str) -> bool:
    required = [
        root/"structure"/f"{item_id}.json", root/"spec"/f"{item_id}.npy",
        *(root/"demix"/"htdemucs"/item_id/f"{stem}.wav" for stem in ("bass", "drums", "other", "vocals")),
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--require-paths", action="store_true")
    parser.add_argument("--analysis-duration", type=float)
    parser.add_argument("--view-root", type=Path, help="materialize exact-duration crops here")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--skip-complete-root", type=Path, action="append", default=[],
                        help="omit rows with complete All-In-One outputs under this root")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="omit IDs already frozen in an earlier prepared CSV/JSONL wave")
    args = parser.parse_args()
    raw_rows = read_records(args.manifest)
    for expression in args.where:
        if "=" not in expression:
            raise ValueError(f"--where must be FIELD=VALUE: {expression}")
        field, value = expression.split("=", 1)
        raw_rows = [row for row in raw_rows if str(row.get(field, "")) == value]
    excluded_ids: set[str] = set()
    for path in args.exclude_manifest:
        for row in read_records(path):
            item_id = str(first(row, ("item_id", "track", "id")))
            if not item_id:
                raise RuntimeError(f"Missing item ID in exclusion manifest {path}")
            excluded_ids.add(item_id)
    if excluded_ids:
        raw_rows = [row for row in raw_rows if str(first(row, ("item_id", "track", "id"))) not in excluded_ids]
    rows = [value for raw in raw_rows if (value := normalize(raw)) is not None]
    if args.skip_complete_root:
        rows = [row for row in rows if not any(allinone_complete(root, str(row["item_id"])) for root in args.skip_complete_root)]
    if args.view_root:
        if not args.analysis_duration:
            raise ValueError("--view-root requires --analysis-duration")
        args.view_root.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(lambda row: exact_view(row, args.view_root, args.analysis_duration), rows))
    rows.sort(key=lambda row: (str(row["source_id"]), str(row["item_id"])))
    ids = [str(row["item_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise RuntimeError("Manifest is empty or item ids are not unique")
    missing = [str(row["standardized_path"]) for row in rows if not Path(str(row["standardized_path"])).is_file()]
    if args.require_paths and missing:
        raise RuntimeError(f"Missing {len(missing)} paths; first={missing[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical = args.output_dir / "inference_manifest.csv"
    with canonical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    shards: list[list[dict[str, object]]] = [[] for _ in range(args.n_shards)]
    for index, row in enumerate(rows):
        shards[index % args.n_shards].append(row)
    shard_meta = []
    for index, shard in enumerate(shards):
        body = "\n".join(str(row["standardized_path"]) for row in shard) + ("\n" if shard else "")
        path = args.output_dir / f"inference_shard_{index:02d}.txt"
        path.write_text(body, encoding="utf-8")
        shard_meta.append({"index": index, "rows": len(shard), "sha256": digest(body.encode())})
    payload = {
        "rows": len(rows), "sources": Counter(str(row["source_id"]) for row in rows),
        "labels": Counter(str(row["label"]) for row in rows), "missing_paths": len(missing),
        "excluded_ids": len(excluded_ids),
        "exclude_manifest_sha256": {
            str(path): digest(path.read_bytes()) for path in args.exclude_manifest
        },
        "manifest_sha256": digest(canonical.read_bytes()), "shards": shard_meta,
    }
    (args.output_dir / "inference_shards.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
