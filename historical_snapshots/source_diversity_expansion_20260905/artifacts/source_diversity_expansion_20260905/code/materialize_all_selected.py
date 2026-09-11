#!/usr/bin/env python3
"""Resume-safe materialization of every row in the frozen v2 audio manifest.

The frozen CSV is read-only. Original/native bytes are retained, then two derived
views are emitted: a canonical 10 s/44.1 kHz/stereo/PCM16 FLAC view and an
unpadded, at-most-60 s view that contains the frozen 10 s selection.

Supported locators:
* direct HTTP(S) audio files (including pinned Hugging Face resolve URLs)
* hf:// pinned archives with a per-row ``container_member``
* Parquet shards containing embedded Hugging Face Audio values (DEAM)

``materialization_events.jsonl`` is append-only. The canonical, de-duplicated
``materialization_manifest.jsonl`` is written atomically at checkpoints and exit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any


SEED = "ai-human-source-diversity-v2-catalogue-first-20260904"
SCHEMA_VERSION = "materialization-v3-native-10s-max60"
MANIFEST_COLUMNS = [
    "schema_version", "item_id", "status", "attempt", "timestamp_utc",
    "label", "source_id", "source_type", "role", "source_revision",
    "source_path", "source_locator", "container_member", "group_id",
    "artist_or_creator", "title", "genre_or_tags", "provenance_grade",
    "license", "evaluation_allowed", "notes",
    "retrieval_kind", "retrieval_url", "cache_path",
    "native_path", "native_bytes", "native_sha256", "native_duration_s",
    "native_sample_rate_hz", "native_channels", "native_codec",
    "crop_start_s", "view_10s_path", "view_10s_bytes", "view_10s_sha256",
    "view_10s_duration_s", "view_10s_sample_rate_hz", "view_10s_channels",
    "view_10s_codec", "view_10s_padded_s",
    "long_crop_start_s", "view_max60s_path", "view_max60s_bytes",
    "view_max60s_sha256", "view_max60s_duration_s",
    "view_max60s_sample_rate_hz", "view_max60s_channels",
    "view_max60s_codec", "view_max60s_padded_s",
    "eligible_through_8khz", "eligible_through_10khz",
    "eligible_through_20khz", "error_type", "error_message",
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(value: str) -> float:
    digest = hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def choose_start(item: dict[str, str], duration: float) -> float:
    frozen = item["window_start_s"]
    if frozen not in {"", "deterministic_after_probe"}:
        return max(0.0, min(float(frozen), max(0.0, duration - 10.0)))
    if duration <= 10.0:
        return 0.0
    source_key = item["source_id"].removeprefix("human_")
    identity = item["source_path"] if source_key != "hindustani_raag_hf" else item["group_id"]
    return (duration - 10.0) * (0.1 + 0.8 * stable_fraction(f"{source_key}|{identity}"))


def choose_long_start(crop_start: float, duration: float) -> tuple[float, float]:
    """Return an unpadded <=60 s window containing the frozen 10 s window."""
    long_duration = min(60.0, duration)
    if duration <= 60.0:
        return 0.0, long_duration
    return max(0.0, min(crop_start - 25.0, duration - 60.0)), 60.0


def hf_uri_to_url(locator: str) -> str:
    # hf://datasets/org/repo@revision/path -> pinned resolve URL
    prefix = "hf://datasets/"
    if not locator.startswith(prefix):
        return locator
    rest = locator[len(prefix):]
    repo, revision_and_path = rest.rsplit("@", 1)
    revision, path = revision_and_path.split("/", 1)
    quoted = "/".join(urllib.parse.quote(x, safe="") for x in path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quoted}"


def locator_alternates(url: str) -> list[str]:
    values = [url]
    if "huggingface.co/" in url:
        values.append(url.replace("https://huggingface.co/", "https://hf.co/", 1))
        sep = "&" if "?" in url else "?"
        values.append(url + sep + "download=true")
    return list(dict.fromkeys(values))


def download(url: str, destination: Path, curl: str) -> str:
    """Download with resumable partials, atomic publication, and safe alternates."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return url
    partial = destination.with_name(destination.name + ".part")
    errors: list[str] = []
    for candidate in locator_alternates(url):
        command = [
            curl, "-L", "--fail", "--retry", "6", "--retry-all-errors",
            "--retry-delay", "3", "--connect-timeout", "30", "--speed-time", "90",
            "--speed-limit", "1024", "-C", "-", "--silent", "--show-error",
            "-o", str(partial), candidate,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and partial.exists() and partial.stat().st_size > 0:
            os.replace(partial, destination)
            return candidate
        errors.append(f"{candidate}: rc={result.returncode}: {result.stderr[-500:]}")
        # Some servers reject range resumes. Keep the partial for other transient
        # errors, but retry from zero after curl explicitly reports range failure.
        if result.returncode == 33:
            partial.unlink(missing_ok=True)
    raise RuntimeError("all locator attempts failed: " + " | ".join(errors))


def probe(ffprobe: str, path: Path) -> dict[str, Any]:
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=sample_rate,channels,codec_name",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if not value.get("streams"):
        raise ValueError("no audio stream")
    stream = value["streams"][0]
    duration = value.get("format", {}).get("duration") or stream.get("duration")
    if duration is None:
        raise ValueError("audio duration unavailable")
    return {
        "duration_s": float(duration), "sample_rate_hz": int(stream["sample_rate"]),
        "channels": int(stream["channels"]), "codec": stream["codec_name"],
    }


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def clean_suffix(value: str, fallback: str = ".audio") -> str:
    suffix = PurePosixPath(urllib.parse.unquote(value)).suffix.lower()
    if not suffix or len(suffix) > 8:
        return fallback
    return suffix


def existing_native_path(item: dict[str, str], output_root: Path) -> Path | None:
    source_root = output_root / "native" / item["source_id"]
    if item["container_member"]:
        candidate = source_root / (item["item_id"] + clean_suffix(item["container_member"]))
        return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None
    base = item["source_locator"].split("#", 1)[0]
    if base.lower().endswith(".parquet"):
        matches = [p for p in source_root.glob(item["item_id"] + ".*")
                   if p.is_file() and ".part" not in p.name and p.stat().st_size > 0]
        return matches[0] if len(matches) == 1 else None
    candidate = source_root / (item["item_id"] + clean_suffix(item["source_path"]))
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def annotate_existing_native(item: dict[str, str], output_root: Path) -> bool:
    native = existing_native_path(item, output_root)
    if native is None:
        return False
    item["_native_path"] = str(native)
    base = item["source_locator"].split("#", 1)[0]
    if item["container_member"]:
        item["_retrieval_kind"] = "archive_member"
        archive_url = hf_uri_to_url(item["source_locator"])
        item["_retrieval_url"] = archive_url
        item["_cache_path"] = str((output_root / "cache" / "archives" /
                                   Path(urllib.parse.urlparse(archive_url).path).name).resolve())
    elif base.lower().endswith(".parquet"):
        item["_retrieval_kind"] = "parquet_embedded_audio"
        item["_retrieval_url"] = base
        item["_cache_path"] = str((output_root / "cache" / "parquet" /
                                   Path(urllib.parse.urlparse(base).path).name).resolve())
    else:
        item["_retrieval_kind"] = "direct_audio"
        item["_retrieval_url"] = item["source_locator"]
    return True


def ensure_archive_member(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            info = bundle.getmember(member)
            source = bundle.extractfile(info)
            if source is None:
                raise FileNotFoundError(f"not a regular archive member: {member}")
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise ValueError(f"empty archive member: {member}")
    os.replace(temporary, destination)


def extract_archive_group(archive: Path, items: list[dict[str, str]], native_root: Path) -> None:
    """Extract all selected native members in one archive scan.

    In particular, re-opening an 8 GB gzip tar for every selected track would
    repeatedly decompress the prefix and is prohibitively slow. A single ordered
    scan also avoids concurrent seeks on one compressed stream.
    """
    wanted: dict[str, tuple[dict[str, str], Path]] = {}
    for item in items:
        destination = native_root / item["source_id"] / (
            item["item_id"] + clean_suffix(item["container_member"])
        )
        if destination.exists() and destination.stat().st_size > 0:
            item["_native_path"] = str(destination)
        else:
            wanted[item["container_member"]] = (item, destination)
    if not wanted:
        return
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            archive_names = set(bundle.namelist())
            for member, (item, destination) in wanted.items():
                matches = [name for name in archive_names
                           if name == member or name.endswith("/" + member)]
                if len(matches) != 1:
                    raise KeyError(f"archive member resolution for {member!r}: {matches[:20]}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".part")
                with bundle.open(matches[0]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                os.replace(temporary, destination)
                item["_native_path"] = str(destination)
        return
    remaining = set(wanted)
    with tarfile.open(archive, "r:*") as bundle:
        for info in bundle:
            matches = [member for member in remaining
                       if info.name == member or info.name.endswith("/" + member)]
            if not matches:
                continue
            if len(matches) != 1:
                raise KeyError(f"ambiguous archive member {info.name!r}: {matches[:20]}")
            member = matches[0]
            item, destination = wanted[member]
            source = bundle.extractfile(info)
            if source is None:
                raise FileNotFoundError(f"not a regular archive member: {info.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part")
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
            os.replace(temporary, destination)
            item["_native_path"] = str(destination)
            remaining.remove(member)
    if remaining:
        raise KeyError(f"archive members absent ({len(remaining)}): {sorted(remaining)[:20]}")


def parquet_audio_value(value: Any) -> tuple[bytes, str]:
    """Normalize common Arrow representations of datasets.Audio."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), ".audio"
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path") or ""
        if raw is None:
            raise ValueError(f"embedded Audio has no bytes: keys={sorted(value)}")
        return bytes(raw), clean_suffix(path)
    raise TypeError(f"unsupported embedded Audio value: {type(value).__name__}")


def extract_deam_shard(shard: Path, items: list[dict[str, str]], native_root: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for embedded Parquet audio") from exc
    names = pq.read_schema(shard).names
    id_name = next((n for n in names if n.lower() in {"song_id", "songid", "id"}), None)
    audio_name = next((n for n in names if n.lower() in {"audio", "song", "music"}), None)
    if id_name is None or audio_name is None:
        raise ValueError(f"cannot identify id/audio columns: {names}")
    # The annotation arrays are irrelevant here and can be sizeable. Read only
    # the pinned identity and embedded original-audio columns.
    table = pq.read_table(shard, columns=[id_name, audio_name])
    wanted = {str(x["group_id"]): x for x in items}
    found: set[str] = set()
    ids = table[id_name]
    audio = table[audio_name]
    for index in range(table.num_rows):
        key = str(ids[index].as_py())
        item = wanted.get(key)
        if item is None:
            continue
        raw, suffix = parquet_audio_value(audio[index])
        destination = native_root / item["source_id"] / f"{item['item_id']}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_bytes(raw)
        os.replace(temporary, destination)
        item["_native_path"] = str(destination)
        found.add(key)
    missing = sorted(set(wanted) - found)
    if missing:
        raise KeyError(f"song ids absent from shard: {missing[:20]}")


def make_views(item: dict[str, str], native: Path, output_root: Path,
               ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    native_meta = probe(ffprobe, native)
    duration = float(native_meta["duration_s"])
    crop_start = choose_start(item, duration)
    long_start, long_duration = choose_long_start(crop_start, duration)
    view10 = output_root / "views_10s" / item["source_id"] / f"{item['item_id']}.flac"
    view60 = output_root / "views_max60s" / item["source_id"] / f"{item['item_id']}.flac"
    view10.parent.mkdir(parents=True, exist_ok=True)
    view60.parent.mkdir(parents=True, exist_ok=True)

    if not view10.exists():
        temporary = view10.with_name(view10.name + ".part.flac")
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{crop_start:.6f}",
            "-i", str(native), "-t", "10.000", "-af", "apad=pad_dur=10",
            "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", "-c:a", "flac",
            "-threads", "1", str(temporary),
        ], check=True)
        os.replace(temporary, view10)
    if not view60.exists():
        temporary = view60.with_name(view60.name + ".part.flac")
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{long_start:.6f}",
            "-i", str(native), "-t", f"{long_duration:.6f}", "-ar", "44100", "-ac", "2",
            "-sample_fmt", "s16", "-c:a", "flac", "-threads", "1", str(temporary),
        ], check=True)
        os.replace(temporary, view60)

    ten_meta = probe(ffprobe, view10)
    long_meta = probe(ffprobe, view60)
    if abs(float(ten_meta["duration_s"]) - 10.0) > 0.03:
        raise ValueError(f"10 s view duration invalid: {ten_meta}")
    if float(long_meta["duration_s"]) > 60.03:
        raise ValueError(f"long view exceeds 60 s: {long_meta}")
    if float(long_meta["duration_s"]) - duration > 0.03:
        raise ValueError(f"long view padded beyond native duration: native={duration}, view={long_meta}")
    for metadata in (ten_meta, long_meta):
        if metadata["sample_rate_hz"] != 44100 or metadata["channels"] != 2:
            raise ValueError(f"standardized view format invalid: {metadata}")
    return {
        "native_path": str(native.resolve()), "native_bytes": native.stat().st_size,
        "native_sha256": sha256(native),
        **{f"native_{k}": v for k, v in native_meta.items()},
        "crop_start_s": round(crop_start, 6),
        "view_10s_path": str(view10.resolve()), "view_10s_bytes": view10.stat().st_size,
        "view_10s_sha256": sha256(view10),
        **{f"view_10s_{k}": v for k, v in ten_meta.items()},
        "view_10s_padded_s": round(max(0.0, 10.0 - max(0.0, duration - crop_start)), 6),
        "long_crop_start_s": round(long_start, 6),
        "view_max60s_path": str(view60.resolve()), "view_max60s_bytes": view60.stat().st_size,
        "view_max60s_sha256": sha256(view60),
        **{f"view_max60s_{k}": v for k, v in long_meta.items()},
        "view_max60s_padded_s": 0.0,
        "eligible_through_8khz": int(int(native_meta["sample_rate_hz"]) >= 16000),
        "eligible_through_10khz": int(int(native_meta["sample_rate_hz"]) >= 20000),
        "eligible_through_20khz": int(int(native_meta["sample_rate_hz"]) >= 40000),
    }


def base_record(item: dict[str, str], attempt: int) -> dict[str, Any]:
    record = {key: item.get(key, "") for key in MANIFEST_COLUMNS}
    record.update({
        "schema_version": SCHEMA_VERSION, "item_id": item["item_id"],
        "attempt": attempt, "timestamp_utc": now_utc(), "status": "pending",
        "error_type": "", "error_message": "",
    })
    return record


def process_item(item: dict[str, str], args: argparse.Namespace, attempts: dict[str, int]) -> dict[str, Any]:
    record = base_record(item, attempts.get(item["item_id"], 0) + 1)
    try:
        source_locator = item["source_locator"]
        cache_path = ""
        retrieval_url = ""
        if item.get("_native_path"):
            native = Path(item["_native_path"])
            retrieval_kind = item.get("_retrieval_kind", "prepared_native")
            retrieval_url = item.get("_retrieval_url", source_locator.split("#", 1)[0])
            cache_path = item.get("_cache_path", "")
        elif item["container_member"]:
            archive_url = hf_uri_to_url(source_locator)
            archive = Path(item["_cache_path"])
            native = args.output_root / "native" / item["source_id"] / (
                item["item_id"] + clean_suffix(item["container_member"])
            )
            ensure_archive_member(archive, item["container_member"], native)
            retrieval_kind = "archive_member"
            retrieval_url = archive_url
            cache_path = str(archive.resolve())
        else:
            suffix = clean_suffix(item["source_path"])
            native = args.output_root / "native" / item["source_id"] / f"{item['item_id']}{suffix}"
            retrieval_url = download(source_locator, native, args.curl)
            retrieval_kind = "direct_audio"
        record.update({"retrieval_kind": retrieval_kind, "retrieval_url": retrieval_url,
                       "cache_path": cache_path})
        record.update(make_views(item, native, args.output_root, args.ffmpeg, args.ffprobe))
        record["status"] = "success"
    except Exception as exc:  # per-item isolation is a core requirement
        record.update({"status": "error", "error_type": type(exc).__name__,
                       "error_message": str(exc)[-4000:]})
    return record


def valid_success(record: dict[str, Any]) -> bool:
    if record.get("status") != "success":
        return False
    return all(Path(str(record.get(key, ""))).is_file() for key in
               ("native_path", "view_10s_path", "view_max60s_path"))


def write_snapshot(args: argparse.Namespace, latest: dict[str, dict[str, Any]], total: int) -> None:
    ordered = [latest[key] for key in sorted(latest)]
    atomic_write_jsonl(args.output_root / "materialization_manifest.jsonl", ordered)
    success = [x for x in ordered if valid_success(x)]
    errors = [x for x in ordered if x.get("status") == "error"]
    summary = {
        "schema_version": SCHEMA_VERSION, "updated_utc": now_utc(),
        "frozen_total_items": total, "records": len(ordered),
        "successful_items": len(success), "error_items": len(errors),
        "pending_items": total - len(success),
        "success_source_counts": dict(sorted(Counter(x["source_id"] for x in success).items())),
        "error_source_counts": dict(sorted(Counter(x["source_id"] for x in errors).items())),
        "native_bytes": sum(int(x.get("native_bytes") or 0) for x in success),
        "view_10s_bytes": sum(int(x.get("view_10s_bytes") or 0) for x in success),
        "view_max60s_bytes": sum(int(x.get("view_max60s_bytes") or 0) for x in success),
        "manifest_columns": MANIFEST_COLUMNS,
    }
    atomic_write_json(args.output_root / "materialization_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--curl", default="curl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--existing-native-only", action="store_true",
                        help="Snapshot atomically published native files; never download/extract")
    parser.add_argument("--prepare-native-only", action="store_true",
                        help="Extract archive/Parquet native bytes but do not make views or ledger records")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    assert len(all_rows) == 2641, f"expected frozen 2641 rows, got {len(all_rows)}"
    if args.source:
        selected = set(args.source)
        rows = [x for x in all_rows if x["source_id"] in selected]
    else:
        rows = all_rows

    events_path = args.output_root / "materialization_events.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = defaultdict(int)
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                latest[event["item_id"]] = event
                attempts[event["item_id"]] = max(attempts[event["item_id"]], int(event.get("attempt", 0)))

    pending = [x for x in rows if not valid_success(latest.get(x["item_id"], {}))]
    if args.existing_native_only and args.prepare_native_only:
        parser.error("--existing-native-only and --prepare-native-only are mutually exclusive")
    if args.prepare_native_only:
        direct = [x["item_id"] for x in pending if not x["container_member"]
                  and not x["source_locator"].split("#", 1)[0].lower().endswith(".parquet")]
        if direct:
            parser.error("--prepare-native-only accepts only archive/Parquet sources")
    ready_at_start = None
    if args.existing_native_only:
        ready = []
        for item in pending:
            if annotate_existing_native(item, args.output_root):
                ready.append(item)
        pending = ready
        ready_at_start = len(ready)

    # Round-robin sources so downstream work gets broad source coverage early
    # instead of waiting behind hundreds of large files from one catalogue. This
    # changes only execution order, never the frozen selection/window identity.
    source_queues: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for item in sorted(pending, key=lambda x: (x["source_id"], x["item_id"])):
        source_queues[item["source_id"]].append(item)
    pending = []
    while source_queues:
        for source_id in sorted(list(source_queues)):
            pending.append(source_queues[source_id].popleft())
            if not source_queues[source_id]:
                del source_queues[source_id]
    total_pending_before_limit = len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]
    print(json.dumps({"phase": "start", "selected": len(rows), "pending_run": len(pending),
                      "pending_before_limit": total_pending_before_limit,
                      "already_success": len(rows) - len([x for x in rows if not valid_success(latest.get(x["item_id"], {}))]),
                      "existing_native_snapshot_items": ready_at_start}), flush=True)

    # Fetch shared archives once, retaining the archives as provenance/cache.
    archive_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in pending:
        if item["container_member"] and not args.existing_native_only:
            archive_groups[item["source_locator"]].append(item)
    for locator, items in archive_groups.items():
        url = hf_uri_to_url(locator)
        archive_name = Path(urllib.parse.urlparse(url).path).name
        cache = args.output_root / "cache" / "archives" / archive_name
        try:
            used_url = download(url, cache, args.curl)
            extract_archive_group(cache, items, args.output_root / "native")
            print(json.dumps({"phase": "archive_ready", "url": used_url, "path": str(cache),
                              "bytes": cache.stat().st_size, "members_selected": len(items)}), flush=True)
            for item in items:
                item["_cache_path"] = str(cache)
                item["_retrieval_kind"] = "archive_member"
                item["_retrieval_url"] = used_url
        except Exception as exc:
            for item in items:
                event = base_record(item, attempts.get(item["item_id"], 0) + 1)
                event.update({"status": "error", "retrieval_kind": "archive_member",
                              "retrieval_url": url, "cache_path": str(cache),
                              "error_type": type(exc).__name__, "error_message": str(exc)[-4000:]})
                latest[item["item_id"]] = event
                attempts[item["item_id"]] = int(event["attempt"])
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            pending = [x for x in pending if x not in items]

    # Download each Parquet shard once and extract selected embedded original bytes.
    parquet_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in pending:
        base = item["source_locator"].split("#", 1)[0]
        if base.lower().endswith(".parquet") and not args.existing_native_only:
            parquet_groups[base].append(item)
    for url, items in parquet_groups.items():
        cache = args.output_root / "cache" / "parquet" / Path(urllib.parse.urlparse(url).path).name
        try:
            used_url = download(url, cache, args.curl)
            extract_deam_shard(cache, items, args.output_root / "native")
            print(json.dumps({"phase": "parquet_ready", "url": used_url, "path": str(cache),
                              "bytes": cache.stat().st_size, "items_extracted": len(items)}), flush=True)
            for item in items:
                item["_cache_path"] = str(cache)
                item["_retrieval_kind"] = "parquet_embedded_audio"
                item["_retrieval_url"] = used_url
        except Exception as exc:
            for item in items:
                event = base_record(item, attempts.get(item["item_id"], 0) + 1)
                event.update({"status": "error", "retrieval_kind": "parquet_embedded_audio",
                              "retrieval_url": url, "cache_path": str(cache),
                              "error_type": type(exc).__name__, "error_message": str(exc)[-4000:]})
                latest[item["item_id"]] = event
                attempts[item["item_id"]] = int(event["attempt"])
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            pending = [x for x in pending if x not in items]

    if args.prepare_native_only:
        native_ready = sum(existing_native_path(item, args.output_root) is not None for item in rows)
        print(json.dumps({"phase": "native_prepare_complete", "selected": len(rows),
                          "native_ready": native_ready}), flush=True)
        return

    lock = threading.Lock()
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, item, args, attempts): item for item in pending}
        for future in as_completed(futures):
            event = future.result()
            with lock:
                processed += 1
                latest[event["item_id"]] = event
                attempts[event["item_id"]] = int(event["attempt"])
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                if processed % args.checkpoint_every == 0 or event["status"] == "error":
                    write_snapshot(args, latest, len(all_rows))
                print(json.dumps({"phase": "item", "processed": processed, "pending_run": len(pending),
                                  "item_id": event["item_id"], "source_id": event["source_id"],
                                  "status": event["status"], "error": event["error_message"][:300]}),
                      flush=True)
    write_snapshot(args, latest, len(all_rows))
    summary = json.loads((args.output_root / "materialization_summary.json").read_text())
    print(json.dumps({"phase": "complete", **summary}), flush=True)
    if summary["error_items"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
