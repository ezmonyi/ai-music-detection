#!/usr/bin/env python3
"""Audit released breath labels and retrieve only matching original ZIP members.

Annotation source is a third-party, model-prefilled, human-reviewed layer, NOT
the original VocalSet paper's ground truth. No model fitting or AI labels here.
"""
import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import time
import urllib.request
import wave
import zipfile
import zlib

REPO = "Ewakaa/Vocalset-Breath"
REVISION = "86be3b2eb6884983ff328c35390e88cc1694db87"
URL = "https://zenodo.org/api/records/1193957/files/VocalSet.zip/content"
SIZE = 2077087366
MD5 = "c44f60d34b8724b9a6f6d15e0a3158a9"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fetch_range(start, end):
    last = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}",
                                                          "User-Agent": "audio-phenomena-research/1"})
            with urllib.request.urlopen(request, timeout=40) as response:
                expected = f"bytes {start}-{end}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not honor exact byte range")
                raw = response.read(end - start + 2)
            if len(raw) != end - start + 1:
                raise ValueError("Short/oversized range")
            return raw
        except Exception as error:
            last = error
            if attempt < 5:
                print(json.dumps({"retry": attempt + 1, "error": repr(error)}), flush=True)
                time.sleep(min(40, 2 ** (attempt + 1)))
    raise RuntimeError(f"Failed byte range {start}-{end}") from last


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.cache = []

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        if not 0 <= self.position <= SIZE:
            raise ValueError("Invalid archive seek")
        return self.position

    def tell(self):
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = SIZE - self.position
        n = min(n, SIZE - self.position)
        if n == 0:
            return b""
        if n > 4 * 1024 * 1024:
            raise ValueError("Unexpectedly large ZIP index read")
        for start, raw in self.cache:
            if start <= self.position and self.position + n <= start + len(raw):
                out = raw[self.position - start:self.position - start + n]
                self.position += n
                return out
        start = self.position
        raw = fetch_range(start, start + n - 1)
        self.cache.append((start, raw))
        self.position += n
        return raw


def read_member(info):
    # One request for local header, one for compressed data. Retain CRC and full
    # decoded member SHA-256; do not pretend selected ranges verify whole ZIP MD5.
    raw = fetch_range(info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<4s5H3I2H", raw)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1:
        raise ValueError("Invalid/encrypted local ZIP header")
    start = info.header_offset + 30 + fields[-2] + fields[-1]
    packed = fetch_range(start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        audio = zlib.decompress(packed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        audio = packed
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(audio) != info.file_size or zlib.crc32(audio) & 0xffffffff != info.CRC:
        raise ValueError("Selected ZIP member failed length/CRC")
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    records, invalid, ambiguous = [], [], []
    counts, confidence_counts = collections.Counter(), collections.Counter()
    for path in sorted((args.labels_dir / "labels/vocalset").glob("*.breath.json")):
        raw = path.read_bytes()
        label = json.loads(raw)
        name = label["audio_file"]
        match = re.match(r"^([fm]\d+)_", name)
        if not match:
            ambiguous.append({"filename": name, "reason": "No unambiguous singer prefix"})
            continue
        errors = []
        for field in ("breath_events", "silent_breaths", "uncertain", "hard_negatives", "exhales"):
            for event in label.get(field, []):
                start, end = event["start_sec"], event["end_sec"]
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= label["duration_sec"] + 0.001):
                    errors.append(field + ": invalid time interval")
                counts[field] += 1
                confidence_counts[field + ":" + event.get("confidence", "missing")] += 1
        if errors:
            invalid.append({"filename": name, "errors": errors})
            continue
        records.append({"filename": name, "singer": match[1], "duration_sec": label["duration_sec"],
                        "annotation_path": str(path.resolve()), "annotation_sha256": digest(raw),
                        "high_medium_breaths": sum(e.get("confidence") in ("high", "medium") for e in label["breath_events"]),
                        "prefill": label.get("pre_fill_source"), "labeler": label.get("labeler")})
    audit = {"repo": REPO, "revision": REVISION, "script_sha256": digest(Path(__file__).read_bytes()),
             "selected_labels": len(records), "invalid_labels": invalid, "ambiguous_singer_labels": ambiguous,
             "event_counts_valid_singer_candidates": dict(counts), "confidence_counts": dict(confidence_counts),
             "singer_counts": dict(collections.Counter(r["singer"] for r in records)),
             "strict_positive_events": sum(r["high_medium_breaths"] for r in records),
             "labels": records, "whole_archive_hash_verified": False, "archive_reported_md5": MD5,
             "archive_url": URL, "archive_bytes": SIZE,
             "limitations": ["Third-party single-primary-labeler layer, model-prefilled; no independent IAA results found in release",
                             "Silent-breath field is not a positive (README erratum); low-confidence breath must be ignored",
                             "Singer-less filenames excluded, not guessed; this is not AI/human classification"]}
    atomic(args.output_dir / "annotation_audit.json", audit)
    if not args.download:
        print(json.dumps({k: v for k, v in audit.items() if k != "labels"}))
        return
    with zipfile.ZipFile(RemoteZip()) as archive:
        infos = archive.infolist()
    index = collections.defaultdict(list)
    for info in infos:
        if info.filename.lower().endswith(".wav") and "__MACOSX" not in info.filename:
            index[Path(info.filename).name].append(info)
    receipts, errors = [], []
    for number, row in enumerate(records, 1):
        candidates = index[row["filename"]]
        if len(candidates) != 1:
            errors.append({"filename": row["filename"], "error": "ZIP filename is absent or ambiguous", "matches": len(candidates)})
            continue
        info = candidates[0]
        target = args.output_dir / "audio" / row["filename"]
        try:
            audio = target.read_bytes() if target.exists() else read_member(info)
            if len(audio) != info.file_size or zlib.crc32(audio) & 0xffffffff != info.CRC:
                raise ValueError("Resumed member fails original ZIP CRC/length")
            with wave.open(io.BytesIO(audio)) as wav:
                sr, frames, channels = wav.getframerate(), wav.getnframes(), wav.getnchannels()
                duration = frames / sr
                if abs(duration - row["duration_sec"]) > 0.01:
                    raise ValueError(f"Annotation/audio duration mismatch: {row['duration_sec']} vs {duration}")
                if len(wav.readframes(frames)) != frames * channels * wav.getsampwidth():
                    raise ValueError("Truncated PCM data")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".wav.tmp")
                temporary.write_bytes(audio)
                temporary.replace(target)
            receipt = {**row, "audio_path": str(target.resolve()), "audio_sha256": digest(audio),
                       "archive_member": info.filename, "member_crc32": f"{info.CRC:08x}",
                       "member_bytes": info.file_size, "sample_rate": sr, "frames": frames, "channels": channels,
                       "duration_verified": duration}
            receipts.append(receipt)
            atomic(args.output_dir / "receipts" / (row["filename"] + ".json"), receipt)
        except Exception as error:
            errors.append({"filename": row["filename"], "error": repr(error)})
        if number % 10 == 0:
            print(json.dumps({"examined": number, "selected": len(records), "downloaded_verified": len(receipts), "errors": len(errors)}), flush=True)
    summary = {"status": "complete_accounting", "selected": len(records), "downloaded_verified": len(receipts),
               "errors": errors, "manifest": receipts, "whole_archive_hash_verified": False,
               "annotation_audit_sha256": digest((args.output_dir / "annotation_audit.json").read_bytes())}
    atomic(args.output_dir / "acquisition_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}), flush=True)


if __name__ == "__main__":
    main()
