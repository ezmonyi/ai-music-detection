#!/usr/bin/env python3
"""Export small, pinned metadata snapshots used by the v2 source manifest.

The audio archives are deliberately not downloaded here.  This script records only
the fields needed to select independent works/artists before any audio or detector
feature is inspected.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source_metadata"


def run_json(arguments: list[str]) -> object:
    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def hf_sql(parquet_url: str, columns: str) -> list[dict[str, object]]:
    query = f"SELECT {columns} FROM read_parquet('{parquet_url}')"
    value = run_json(["hf", "datasets", "sql", query, "--format", "json"])
    assert isinstance(value, list)
    return value


def repo_audio_paths(repo_id: str, revision: str, predicate) -> list[str]:
    value = run_json([
        "hf", "datasets", "list", repo_id, "--recursive", "--revision", revision,
        "--format", "json",
    ])
    assert isinstance(value, list)
    paths = sorted(
        item["path"] for item in value
        if isinstance(item, dict) and isinstance(item.get("path"), str)
        and predicate(item["path"])
    )
    return paths


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    write_json(
        "mtg_jamendo_records.json",
        hf_sql(
            "https://huggingface.co/datasets/seungheondoh/cmd-mtg_jamendo-metadata/"
            "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "id, title, artist, license, audio_path, duration, sample_rate, channels, "
            "genre, instrument, original_id",
        ),
    )
    write_json(
        "moisesdb_records.json",
        hf_sql(
            "https://huggingface.co/datasets/seungheondoh/cmd-moisesdb-metadata/"
            "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "id, type, medleydb_id, license, genre, instrument, audio_path",
        ),
    )
    write_json(
        "medleydb_records.json",
        hf_sql(
            "https://huggingface.co/datasets/seungheondoh/cmd-medleydb-metadata/"
            "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "id, type, medleydb_id, license, genre, instrument, audio_path",
        ),
    )
    write_json(
        "urmp_mix_paths.json",
        repo_audio_paths(
            "DreamyWanderer/URMP-Reduced",
            "3f2da70e1f46a8806d15b02e6586cfc7ae5bf782",
            lambda path: "/AuMix_" in path and path.endswith(".wav"),
        ),
    )
    write_json(
        "hindustani_raag_paths.json",
        repo_audio_paths(
            "neerajaabhyankar/hindustani-raag-small",
            "326caef0bc01da44ad46e4d9c65a5146da6bcc5b",
            lambda path: path.endswith(".mp3") and "_chunk" in path,
        ),
    )

    print(json.dumps({
        "mtg_jamendo": len(json.loads((OUT / "mtg_jamendo_records.json").read_text())),
        "moisesdb": len(json.loads((OUT / "moisesdb_records.json").read_text())),
        "medleydb": len(json.loads((OUT / "medleydb_records.json").read_text())),
        "urmp_mixes": len(json.loads((OUT / "urmp_mix_paths.json").read_text())),
        "hindustani_clips": len(json.loads((OUT / "hindustani_raag_paths.json").read_text())),
    }, indent=2))


if __name__ == "__main__":
    main()
