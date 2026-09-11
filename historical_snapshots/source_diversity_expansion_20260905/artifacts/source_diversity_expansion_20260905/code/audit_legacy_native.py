#!/usr/bin/env python3
"""Read-only physical audit of the 1,000 selected local FMA/Suno originals."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    assert len(rows) == 1000
    assert Counter(r["source"] for r in rows) == {"fma_medium": 500, "humair_suno": 250, "suno_unknown": 250}

    def check(row):
        path = (args.workspace / row["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        info = json.loads(subprocess.check_output([
            args.ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
            "format=duration:stream=sample_rate,channels,codec_name", "-of", "json", str(path),
        ], text=True))
        stream = info["streams"][0]
        duration = float(info["format"]["duration"])
        sample_rate = int(stream["sample_rate"])
        if sample_rate != int(row["sample_rate"]) or abs(duration - float(row["duration"])) > 1e-5:
            raise ValueError("Legacy native metadata mismatch: " + str(path))
        return {
            "id": f"{row['class_name']}_{row['source']}_{row['id']}",
            "source_id": row["source"], "native_path": str(path), "native_bytes": path.stat().st_size,
            "native_sha256": digest(path), "native_duration_s": duration,
            "native_sample_rate_hz": sample_rate, "native_channels": int(stream["channels"]),
            "native_codec": stream["codec_name"], "status": "verified",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verified = list(pool.map(check, rows))
    payload = {
        "status": "passed", "passed": True, "expected": 1000, "verified": len(verified),
        "manifest_sha256": digest(args.manifest), "native_bytes": sum(r["native_bytes"] for r in verified),
        "sources": dict(Counter(r["source_id"] for r in verified)),
        "unique_native_sha256": len({r["native_sha256"] for r in verified}), "details": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}))


if __name__ == "__main__":
    main()
