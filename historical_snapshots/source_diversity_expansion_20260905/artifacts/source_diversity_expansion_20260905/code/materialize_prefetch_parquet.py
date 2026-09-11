#!/usr/bin/env python3
"""Prefetch unique frozen Parquet shards without touching materialization ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fetch(url: str, root: Path) -> dict[str, object]:
    destination = root / Path(urllib.parse.urlparse(url).path).name
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and destination.stat().st_size:
        return {"status": "exists", "url": url, "path": str(destination),
                "bytes": destination.stat().st_size}
    result = subprocess.run([
        "curl", "-L", "--fail", "--retry", "12", "--retry-all-errors",
        "--retry-delay", "5", "--connect-timeout", "30", "-C", "-",
        "--silent", "--show-error", "-o", str(partial), url,
    ], capture_output=True, text=True)
    if result.returncode:
        return {"status": "error", "url": url, "path": str(destination),
                "error": result.stderr[-1000:]}
    partial.replace(destination)
    return {"status": "success", "url": url, "path": str(destination),
            "bytes": destination.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        urls = sorted({row["source_locator"].split("#", 1)[0]
                       for row in csv.DictReader(handle)
                       if row["source_locator"].split("#", 1)[0].lower().endswith(".parquet")})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, url, args.output_root) for url in urls]
        failed = 0
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "error":
                failed += 1
            print(json.dumps({"progress": f"{index}/{len(urls)}", **result},
                             ensure_ascii=False), flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
