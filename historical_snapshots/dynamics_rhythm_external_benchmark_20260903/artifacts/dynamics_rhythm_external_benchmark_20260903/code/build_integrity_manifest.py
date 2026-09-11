#!/usr/bin/env python3
"""Build the lightweight benchmark metadata and SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_PARTS = {"data", "checkpoints", "venv", "__pycache__"}
EXCLUDED_NAMES = {".DS_Store", "LIGHTWEIGHT_SHA256.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lightweight_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        and path.name not in EXCLUDED_NAMES
        and not path.name.endswith(".pyc")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution-host", default="5090-5")
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 5090, index 0")
    args = parser.parse_args()

    root = args.root.resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    direct_results = [
        "results/rhythm_benchmark/rhythm_benchmark_summary.json",
        "results/maestro_dynamics/summary.json",
        "results/maestro_waveform_dynamics/summary.json",
        "results/salami_structure/summary.json",
        "results/admission_decisions.csv",
        "results/aggregate_summary.json",
        "figures/external_benchmark_dashboard.png",
    ]
    metadata = {
        "snapshot_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_date": "2026-09-03",
        "execution_host": args.execution_host,
        "gpu_restriction": "CUDA_VISIBLE_DEVICES=0",
        "gpu": args.gpu,
        "remote_archive_root": "/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903",
        "source_revisions": {
            "beat_this": "b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c",
            "salami_data_public": "8e4f95d18a3ab628c53011fa5a43e9d3be27965d",
            "all_in_one_infer": "3c93b4ae389328544dd5955af7497030cb1bca3a",
            "maestro_hf_transport": "67e586b1dcf5925c94a6729fcd5eb7f23d504fac",
        },
        "sample_counts": {
            "gtzan_scored_tracks_per_seed": 993,
            "gtzan_explicit_no_downbeat_exclusions": 6,
            "maestro_recordings": 50,
            "maestro_30s_windows": 200,
            "maestro_oracle_event_measurements": 65049,
            "salami_tracks": 44,
            "salami_annotation_files": 68,
        },
        "decision": {
            "admitted_detector_features": 0,
            "ai_human_ablation_run": False,
            "reason": "No waveform-deployable candidate passed its frozen external admission gate.",
        },
        "direct_result_sha256": {
            relative: sha256(root / relative) for relative in direct_results
        },
    }
    metadata_path = reports / "RUN_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    files = lightweight_files(root)
    manifest_path = reports / "LIGHTWEIGHT_SHA256.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")

    print(json.dumps({
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "hashed_files": len(files),
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
