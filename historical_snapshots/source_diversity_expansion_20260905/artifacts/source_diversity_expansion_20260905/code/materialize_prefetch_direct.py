#!/usr/bin/env python3
"""Network-only, resume-safe prefetch of frozen direct-audio native bytes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests


def suffix(value: str) -> str:
    result = PurePosixPath(urllib.parse.unquote(value)).suffix.lower()
    return result if result and len(result) <= 8 else ".audio"


def metadata(url: str) -> tuple[int, str]:
    response = requests.head(url, allow_redirects=False, timeout=(30, 60))
    response.raise_for_status()
    if response.is_redirect:
        total = int(response.headers["x-linked-size"])
        linked = response.headers.get("x-linked-etag", "").strip('"')
    else:
        total = int(response.headers["content-length"])
        linked = response.headers.get("etag", "").strip('"')
    expected_sha256 = linked.lower() if len(linked) == 64 else ""
    return total, expected_sha256


def fetch(item: dict[str, str], root: Path, attempts: int) -> dict[str, object]:
    destination = root / "native" / item["source_id"] / (item["item_id"] + suffix(item["source_path"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"item_id": item["item_id"], "source_id": item["source_id"],
                                 "native_path": str(destination), "timestamp": time.time()}
    try:
        total, expected_sha256 = metadata(item["source_locator"])
        if destination.exists() and destination.stat().st_size == total:
            return {**record, "status": "exists", "native_bytes": total,
                    "expected_sha256": expected_sha256}
        partial = destination.with_name(destination.name + ".part")
        for attempt in range(1, attempts + 1):
            have = partial.stat().st_size if partial.exists() else 0
            if have == total:
                os.replace(partial, destination)
                return {**record, "status": "success", "native_bytes": total,
                        "expected_sha256": expected_sha256, "attempt": attempt}
            if have > total:
                partial.rename(partial.with_name(partial.name + f".oversize.{int(time.time())}"))
                have = 0
            try:
                headers = {"Range": f"bytes={have}-{total - 1}"} if have else {}
                with requests.get(item["source_locator"], headers=headers, stream=True,
                                  allow_redirects=True, timeout=(30, 60)) as response:
                    expected_status = 206 if have else (200, 206)
                    if response.status_code not in ((expected_status,) if isinstance(expected_status, int) else expected_status):
                        raise RuntimeError(f"unexpected HTTP status {response.status_code} at offset {have}")
                    if have:
                        content_range = response.headers.get("content-range", "")
                        if not content_range.startswith(f"bytes {have}-"):
                            raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    with partial.open("ab" if have else "wb") as handle:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if partial.stat().st_size == total:
                    os.replace(partial, destination)
                    return {**record, "status": "success", "native_bytes": total,
                            "expected_sha256": expected_sha256, "attempt": attempt}
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(min(30, attempt * 2))
        raise RuntimeError("attempt loop exhausted")
    except Exception as exc:
        return {**record, "status": "error", "error_type": type(exc).__name__,
                "error_message": str(exc)[-1000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=12)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if not row["container_member"]
                and not row["source_locator"].split("#", 1)[0].lower().endswith(".parquet")]
    success_ids = set()
    ledger = args.data_root / "materialization_events.jsonl"
    if ledger.exists():
        latest = {}
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                latest[record["item_id"]] = record
        success_ids = {key for key, value in latest.items() if value.get("status") == "success"}
    queues: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for item in rows:
        if item["item_id"] not in success_ids:
            queues[item["source_id"]].append(item)
    pending = []
    while queues:
        for source in sorted(list(queues)):
            pending.append(queues[source].popleft())
            if not queues[source]:
                del queues[source]
    events = args.data_root / "direct_prefetch_events.jsonl"
    print(json.dumps({"phase": "start", "pending": len(pending), "workers": args.workers}), flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, item, args.data_root, args.attempts) for item in pending]
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            failed += record["status"] == "error"
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({"progress": f"{index}/{len(pending)}", **record}, ensure_ascii=False), flush=True)
    raise SystemExit(2 if failed else 0)


if __name__ == "__main__":
    main()
