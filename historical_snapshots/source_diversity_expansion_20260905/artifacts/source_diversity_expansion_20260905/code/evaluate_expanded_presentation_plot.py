#!/usr/bin/env python3
"""Presentation-only rendering wrapper with a verified star-capable font.

This does not replace or mutate the formal plot artifacts.  It delegates all
table logic to the hash-frozen formal plotter and changes only its font lookup
so the frozen-leader star renders on macOS instead of as a missing glyph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import ImageFont

import evaluate_expanded_plot as formal_plot


REGULAR_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def presentation_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = BOLD_FONT if bold else REGULAR_FONT
    if not path.is_file():
        raise FileNotFoundError(f"Required presentation font is absent: {path}")
    selected = ImageFont.truetype(str(path), size=size)
    if not bold:
        star = bytes(selected.getmask("★"))
        missing = bytes(selected.getmask("\U0010ffff"))
        if not star or star == missing:
            raise RuntimeError(f"Presentation font does not provide a distinct star glyph: {path}")
    return selected


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--audit", type=Path, required=True)
    known, remaining = wrapper.parse_known_args()
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--cv-summary", type=Path, required=True)
    base_parser.add_argument("--frozen-selection", type=Path, required=True)
    base_parser.add_argument("--output", type=Path, required=True)
    base_parser.add_argument("--panel", choices=("composite", "auc", "ba"), default="composite")
    base_args = base_parser.parse_args(remaining)
    if base_args.panel != "composite":
        raise ValueError("The presentation wrapper is reserved for the composite overview")

    # Exercise the glyph check before delegating, then replace only the formal
    # plotter's font resolver for this process.
    presentation_font(17)
    formal_plot.font = presentation_font
    sys.argv = [sys.argv[0], *remaining]
    formal_plot.main()

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "presentation-only font substitution; formal plots and numeric tables unchanged",
        "panel": base_args.panel,
        "wrapper": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "formal_plotter": {
            "path": str(Path(formal_plot.__file__).resolve()),
            "sha256": sha256(Path(formal_plot.__file__)),
        },
        "font": {
            "regular_path": str(REGULAR_FONT), "regular_sha256": sha256(REGULAR_FONT),
            "bold_path": str(BOLD_FONT), "bold_sha256": sha256(BOLD_FONT),
            "star_glyph_distinct_from_missing_glyph": True,
        },
        "inputs": {
            "cv_summary": {"path": str(base_args.cv_summary.resolve()), "sha256": sha256(base_args.cv_summary)},
            "frozen_selection": {
                "path": str(base_args.frozen_selection.resolve()),
                "sha256": sha256(base_args.frozen_selection),
            },
        },
        "output": {"path": str(base_args.output.resolve()), "sha256": sha256(base_args.output)},
    }
    known.audit.parent.mkdir(parents=True, exist_ok=True)
    known.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
