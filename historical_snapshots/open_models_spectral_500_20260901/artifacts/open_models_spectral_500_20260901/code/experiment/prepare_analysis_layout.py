#!/usr/bin/env python3
"""Build symlinked Human-vs-generator layouts for the existing analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


GENERATORS = ("heartmula", "acestep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-clips", type=Path, required=True)
    parser.add_argument("--baseline-stems", type=Path, required=True)
    parser.add_argument("--standardized-root", type=Path, required=True)
    parser.add_argument("--generated-stems-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=GENERATORS,
        default=list(GENERATORS),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def track_name(row: dict[str, object]) -> str:
    return f"{row['class_name']}_{row['source']}_{row['id']}"


def ensure_symlink(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {target}")
    os.symlink(source.resolve(), target)


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompt_manifest)
    baseline = read_jsonl(args.baseline_manifest)
    humans = [row for row in baseline if int(row["label"]) == 0]
    if len(humans) != 500:
        raise SystemExit(f"expected 500 baseline human tracks, found {len(humans)}")
    if len(prompts) != 500:
        raise SystemExit(f"expected 500 prompts, found {len(prompts)}")
    metadata_path = args.output_dir / "layout_metadata.json"
    if metadata_path.exists():
        summary = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    for generator in args.generators:
        model_rows = [
            {
                "id": row["id"],
                "label": 1,
                "class_name": "ai",
                "source": generator,
                "split": row["split"],
                "language": row["language"],
                "prompt_id": row["id"],
                "prompt_seed": row["seed"],
            }
            for row in prompts
        ]
        rows = sorted(
            [*humans, *model_rows],
            key=lambda row: (int(row["label"]), str(row["source"]), str(row["id"])),
        )
        pair_root = args.output_dir / f"human_vs_{generator}"
        clips = pair_root / "clips"
        stems = pair_root / "stems" / "htdemucs"
        clips.mkdir(parents=True, exist_ok=True)
        stems.mkdir(parents=True, exist_ok=True)
        manifest = pair_root / "manifest.jsonl"
        write_jsonl(manifest, rows)

        for row in humans:
            name = track_name(row)
            ensure_symlink(args.baseline_clips / f"{name}.flac", clips / f"{name}.flac")
            ensure_symlink(args.baseline_stems / "htdemucs" / name, stems / name)
        for row in model_rows:
            name = track_name(row)
            ensure_symlink(
                args.standardized_root / generator / f"{name}.flac",
                clips / f"{name}.flac",
            )
            ensure_symlink(
                args.generated_stems_root / generator / "htdemucs" / name,
                stems / name,
            )

        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (pair_root / "manifest.sha256").write_text(
            f"{digest}  manifest.jsonl\n", encoding="utf-8"
        )
        summary[generator] = {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": digest,
            "tracks": len(rows),
            "human": len(humans),
            "ai": len(model_rows),
            "clip_links": len(list(clips.glob("*.flac"))),
            "stem_links": len(list(stems.iterdir())),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
