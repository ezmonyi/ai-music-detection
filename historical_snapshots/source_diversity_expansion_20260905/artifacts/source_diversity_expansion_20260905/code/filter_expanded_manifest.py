#!/usr/bin/env python3
"""Atomically select metadata rows for inference or feature-extraction waves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def parse_rule(value: str) -> tuple[str, set[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("rule must be FIELD=VALUE[,VALUE...]")
    field, raw = value.split("=", 1)
    choices = {part for part in raw.split(",") if part != ""}
    if not field or not choices:
        raise argparse.ArgumentTypeError("rule must contain a field and at least one value")
    return field, choices


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include", type=parse_rule, action="append", default=[])
    parser.add_argument("--exclude", type=parse_rule, action="append", default=[])
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader); fields = list(reader.fieldnames or [])
    if not fields:
        raise RuntimeError("input has no CSV header")
    id_field = next((name for name in ("item_id", "track", "id") if name in fields), None)
    if id_field is None:
        raise RuntimeError("input has no item_id/track/id column")
    ids = [row[id_field] for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("input IDs must be nonempty and unique")
    for field, _ in [*args.include, *args.exclude]:
        if field not in fields:
            raise RuntimeError(f"rule field absent from input: {field}")
    excluded_ids: set[str] = set()
    for path in args.exclude_manifest:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                identifier = row.get("item_id") or row.get("track") or row.get("id") or ""
                if identifier:
                    excluded_ids.add(identifier)
    selected = [
        row for row in rows
        if all(row[field] in values for field, values in args.include)
        and all(row[field] not in values for field, values in args.exclude)
        and row[id_field] not in excluded_ids
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({
        "input": str(args.input), "input_sha256": sha256(args.input), "input_rows": len(rows),
        "output": str(args.output), "output_sha256": sha256(args.output), "output_rows": len(selected),
        "include": [{field: sorted(values)} for field, values in args.include],
        "exclude": [{field: sorted(values)} for field, values in args.exclude],
        "excluded_manifest_rows": len(excluded_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
