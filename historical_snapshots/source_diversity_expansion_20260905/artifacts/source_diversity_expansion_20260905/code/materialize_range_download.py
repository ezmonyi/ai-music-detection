#!/usr/bin/env python3
"""Resumable bounded-range downloader for large pinned Hugging Face LFS files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_verification(destination: Path, value: dict[str, object]) -> None:
    sidecar = destination.with_name(destination.name + ".verified.json")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=sidecar.parent,
                                     prefix=f".{sidecar.name}.", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, sidecar)


def get_metadata(url: str) -> tuple[int, str, str]:
    # The Hub response carries the Git-LFS SHA-256 in X-Linked-ETag. The final
    # Xet CDN ETag is a content-addressed transport key and is not the file hash.
    hub = requests.head(url, allow_redirects=False, timeout=(30, 60))
    hub.raise_for_status()
    if hub.is_redirect:
        resolved_url = hub.headers["location"]
        total = int(hub.headers["x-linked-size"])
        etag = hub.headers.get("x-linked-etag", "").strip('"')
    else:
        resolved_url = url
        total = int(hub.headers["content-length"])
        etag = hub.headers.get("etag", "").strip('"')
    expected_sha256 = etag.lower() if len(etag) == 64 and all(c in "0123456789abcdefABCDEF" for c in etag) else ""
    return total, expected_sha256, resolved_url


def fetch_range(source_url: str, path: Path, start: int, end: int,
                attempts: int) -> dict[str, object]:
    expected = end - start + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            return {"status": "exists", "path": str(path), "bytes": have}
        if have > expected:
            path.rename(path.with_name(path.name + f".oversize.{int(time.time())}"))
            have = 0
        request_start = start + have
        try:
            with requests.get(source_url, headers={"Range": f"bytes={request_start}-{end}"},
                              stream=True, allow_redirects=True, timeout=(30, 60)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range status {response.status_code}, expected 206")
                content_range = response.headers.get("content-range", "")
                if not content_range.startswith(f"bytes {request_start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")
                mode = "ab" if have else "wb"
                with path.open(mode) as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual = path.stat().st_size
            if actual == expected:
                return {"status": "success", "path": str(path), "bytes": actual}
            raise IOError(f"short range after response: {actual}/{expected}")
        except Exception as exc:
            if attempt == attempts:
                return {"status": "error", "path": str(path), "error": str(exc),
                        "bytes": path.stat().st_size if path.exists() else 0}
            time.sleep(min(30, 2 * attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--attempts", type=int, default=12)
    args = parser.parse_args()
    total, expected_sha256, resolved_url = get_metadata(args.url)
    if args.destination.exists() and args.destination.stat().st_size == total:
        digest = file_sha256(args.destination)
        if expected_sha256 and digest != expected_sha256:
            raise SystemExit(f"existing file checksum mismatch: {digest} != {expected_sha256}")
        result = {"status": "exists_verified", "url": args.url,
                  "path": str(args.destination), "bytes": total, "sha256": digest,
                  "expected_sha256": expected_sha256,
                  "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_verification(args.destination, result)
        print(json.dumps(result), flush=True)
        return
    chunk_bytes = args.chunk_mib * 1024 * 1024
    ranges = [(start, min(total - 1, start + chunk_bytes - 1))
              for start in range(0, total, chunk_bytes)]
    part_root = args.destination.with_name(args.destination.name + ".ranges")
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (start, end) in enumerate(ranges):
            path = part_root / f"part-{index:05d}-{start}-{end}"
            futures.append(executor.submit(fetch_range, resolved_url, path, start, end, args.attempts))
        failed = []
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps({"progress": f"{index}/{len(ranges)}", **result}), flush=True)
            if result["status"] == "error":
                failed.append(result)
    if failed:
        raise SystemExit(f"{len(failed)} ranges failed; rerun is resumable")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    assembling = args.destination.with_name(args.destination.name + ".assembling")
    digest = hashlib.sha256()
    written = 0
    with assembling.open("wb") as target:
        for index, (start, end) in enumerate(ranges):
            part = part_root / f"part-{index:05d}-{start}-{end}"
            if part.stat().st_size != end - start + 1:
                raise SystemExit(f"bad range length: {part}")
            with part.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    actual_sha256 = digest.hexdigest()
    if written != total:
        raise SystemExit(f"assembled length mismatch: {written} != {total}")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise SystemExit(f"assembled checksum mismatch: {actual_sha256} != {expected_sha256}")
    os.replace(assembling, args.destination)
    result = {"status": "success_verified", "url": args.url,
              "resolved_url": resolved_url, "path": str(args.destination),
              "bytes": written, "sha256": actual_sha256,
              "expected_sha256": expected_sha256,
              "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_verification(args.destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
