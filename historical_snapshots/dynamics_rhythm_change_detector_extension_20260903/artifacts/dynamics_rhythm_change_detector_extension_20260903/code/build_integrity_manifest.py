#!/usr/bin/env python3
"""Write stable SHA-256 records for the lightweight experiment archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    files = [
        path for path in sorted(args.root.rglob("*"))
        if path.is_file()
        and path.resolve() != output
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    lines = [f"{digest(path)}  {path.relative_to(args.root)}" for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} hashes to {args.output}")


if __name__ == "__main__":
    main()
