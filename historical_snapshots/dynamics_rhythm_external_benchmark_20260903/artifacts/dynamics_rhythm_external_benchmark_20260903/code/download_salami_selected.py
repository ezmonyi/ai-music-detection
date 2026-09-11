#!/usr/bin/env python3
"""Download the frozen, reachable SALAMI Internet Archive audio subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


USER_AGENT = "Mozilla/5.0 SALAMI-research-audit/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, retries: int = 6) -> tuple[str, int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    final_url = url
    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                final_url = response.geturl()
                append = existing > 0 and response.status == 206
                mode = "ab" if append else "wb"
                if existing and not append:
                    existing = 0
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if part.stat().st_size <= 1024:
                raise IOError("downloaded file is implausibly small")
            os.replace(part, output)
            return final_url, output.stat().st_size, "downloaded"
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(f"Failed {url} after {retries} attempts: {error}") from error
            time.sleep(min(5 * attempt, 20))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    with args.selected_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not 30 <= len(rows) <= 50 or any(row["status"] != "frozen_selected" for row in rows):
        raise RuntimeError("Manifest must contain 30--50 frozen selected tracks")

    records = []
    for index, row in enumerate(rows, start=1):
        song_id = row["song_id"]
        source_url = row["url"]
        # The availability audit records the exact Internet Archive derivative
        # that answered the probe.  Reuse it here: some compute hosts cannot
        # reach archive.org's current redirect target even though the frozen
        # derivative remains reachable from the audit host.
        download_url = row.get("final_url", "") or source_url
        suffix = Path(urlparse(download_url).path).suffix.lower()
        if suffix not in {".mp3", ".flac", ".wav", ".ogg", ".m4a"}:
            suffix = ".audio"
        output = args.output_root / f"{song_id}{suffix}"
        if output.exists() and output.stat().st_size > 1024:
            final_url, size, status = download_url, output.stat().st_size, "existing"
        else:
            final_url, size, status = download(download_url, output)
        record = {
            "selection_rank": int(row["selection_rank"]),
            "song_id": song_id,
            "source_url": source_url,
            "final_url": final_url,
            "local_path": str(output.resolve()),
            "bytes": size,
            "sha256": sha256_file(output),
            "status": status,
        }
        records.append(record)
        print(
            f"[{index:03d}/{len(rows):03d}] {song_id} {status} {size / 1e6:.1f} MB",
            flush=True,
        )

    result = {
        "n_tracks": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "records": records,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_tracks": result["n_tracks"], "total_bytes": result["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
