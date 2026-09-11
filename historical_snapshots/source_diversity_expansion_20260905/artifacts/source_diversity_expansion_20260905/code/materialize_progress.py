#!/usr/bin/env python3
"""Write stage-separated live progress for the frozen materialization run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


ARCHIVES = {
    "human_magnatagatune": "mp3.zip",
    "human_medleydb": "medleydb.tar.gz",
    "human_moisesdb": "moisesdb.tar.gz",
    "human_gtzan": "genres.tar.gz",
}


def ids_in(root: Path, wanted: set[str]) -> int:
    if not root.is_dir():
        return 0
    found = set()
    for path in root.iterdir():
        if not path.is_file() or ".part" in path.name:
            continue
        name = path.name
        for suffix in (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac", ".audio"):
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)]
                break
        if name in wanted:
            found.add(name)
    return len(found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.frozen.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    latest = {}
    events = args.data_root / "materialization_events.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            if line:
                event = json.loads(line)
                latest[event["item_id"]] = event
    stages = {}
    parquet_root = args.data_root / "cache" / "parquet"
    parquet_final = len(list(parquet_root.glob("*.parquet"))) if parquet_root.is_dir() else 0
    parquet_partial = len(list(parquet_root.glob("*.part"))) if parquet_root.is_dir() else 0
    for source, source_rows in sorted(by_source.items()):
        wanted = {x["item_id"] for x in source_rows}
        success = [latest[x] for x in wanted if x in latest and latest[x].get("status") == "success"]
        failed = [latest[x] for x in wanted if x in latest and latest[x].get("status") == "error"]
        record = {
            "selected_items": len(wanted),
            "native_ready_items": ids_in(args.data_root / "native" / source, wanted),
            "view_10s_ready_items": ids_in(args.data_root / "views_10s" / source, wanted),
            "view_max60s_ready_items": ids_in(args.data_root / "views_max60s" / source, wanted),
            "ledger_success_items": len(success), "ledger_error_items": len(failed),
        }
        if source in ARCHIVES:
            archive = args.data_root / "cache" / "archives" / ARCHIVES[source]
            record.update({
                "retrieval_container_kind": "archive",
                "archive_download_complete": archive.is_file(),
                "archive_bytes": archive.stat().st_size if archive.is_file() else 0,
                "archive_verified_sidecar": archive.with_name(archive.name + ".verified.json").is_file(),
            })
        elif source == "human_deam":
            record.update({"retrieval_container_kind": "parquet_embedded",
                           "parquet_shards_complete": parquet_final,
                           "parquet_shards_partial": parquet_partial,
                           "parquet_shards_expected": 100})
        else:
            record["retrieval_container_kind"] = "direct_audio"
        stages[source] = record
    success_values = [x for x in latest.values() if x.get("status") == "success"]
    output = {
        "progress_version": "materialization-progress-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_items": len(rows), "ledger_success_items": len(success_values),
        "ledger_error_items": sum(x.get("status") == "error" for x in latest.values()),
        "ledger_success_by_source": dict(sorted(Counter(x["source_id"] for x in success_values).items())),
        "source_stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent,
                                     prefix=f".{args.output.name}.", delete=False) as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
