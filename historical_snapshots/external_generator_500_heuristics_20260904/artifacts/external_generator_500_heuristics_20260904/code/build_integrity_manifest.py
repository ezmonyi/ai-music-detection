#!/usr/bin/env python3
"""Hash lightweight experiment artifacts while excluding large audio/intermediates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_PARTS = {"audio", "demix", "spec"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path == args.output or any(part in EXCLUDED_PARTS for part in path.relative_to(args.root).parts):
            continue
        files.append({"path": str(path.relative_to(args.root)), "bytes": path.stat().st_size, "sha256": digest(path)})
    payload = {"file_count": len(files), "files": files}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(files), "manifest_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()

