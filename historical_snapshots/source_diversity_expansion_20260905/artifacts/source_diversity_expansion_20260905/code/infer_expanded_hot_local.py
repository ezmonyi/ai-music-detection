#!/usr/bin/env python3
"""Run one frozen All-In-One shard via task-owned local scratch, then publish."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STEMS = ("bass", "drums", "other", "vocals")
RESERVE_BYTES = 5 * (1 << 30)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def mark(timing: dict[str, object], name: str, start: float) -> None:
    timing[name] = {"seconds": time.monotonic() - start, "completed_utc": datetime.now(timezone.utc).isoformat()}


def gpu_idle(gpu: int) -> None:
    used, utilization = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if int(used.strip()) > 1024 or int(utilization.strip()) > 5:
        raise RuntimeError(f"Refusing busy GPU {gpu}: {used} MiB, {utilization}%")


def verify_reserve(path: Path, required_working_bytes: int = 0) -> int:
    free = shutil.disk_usage(path).free
    if free - required_working_bytes < RESERVE_BYTES:
        raise RuntimeError(
            f"Local scratch reserve would be violated: free={free}, "
            f"working_estimate={required_working_bytes}, reserve={RESERVE_BYTES}"
        )
    return free


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--nfs-output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clear-local-after-verify", action="store_true")
    args = parser.parse_args()

    scratch = args.scratch_root.resolve()
    data_root = Path("/data").resolve()
    if scratch == data_root or data_root not in scratch.parents:
        raise RuntimeError("--scratch-root must be a task-specific child of /data")
    if scratch.exists() and any(scratch.iterdir()):
        raise RuntimeError(f"Scratch root is not empty: {scratch}")
    scratch.mkdir(parents=True, exist_ok=True)

    shard = args.prepared_dir / f"inference_shard_{args.shard_index:02d}.txt"
    manifest = args.prepared_dir / "inference_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path_to_row = {row["standardized_path"]: row for row in rows}
    source_paths = [Path(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
    if not source_paths:
        raise RuntimeError(f"Empty shard: {shard}")
    selected = []
    for source in source_paths:
        row = path_to_row.get(str(source))
        if row is None:
            raise RuntimeError(f"Shard path absent from canonical manifest: {source}")
        if not source.is_file():
            raise RuntimeError(f"Missing source: {source}")
        selected.append((row["item_id"], source))
    if len({item_id for item_id, _ in selected}) != len(selected):
        raise RuntimeError("Duplicate item IDs in shard")

    input_bytes = sum(path.stat().st_size for _, path in selected)
    # Udio measurement was roughly 8x retained output/input. Reserve 9x input
    # plus the mandatory 5 GiB headroom before beginning an actual shard.
    free_before = verify_reserve(scratch, input_bytes * 9)
    local_inputs = scratch / "inputs"
    local_output = scratch / "output"
    local_inputs.mkdir(); local_output.mkdir()
    timing: dict[str, object] = {}
    input_hashes: dict[str, dict[str, str]] = {}

    started = time.monotonic()
    local_paths = []
    for item_id, source in selected:
        destination = local_inputs / f"{item_id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        source_hash, local_hash = digest(source), digest(destination)
        if source_hash != local_hash:
            raise RuntimeError(f"Input staging hash mismatch: {item_id}")
        input_hashes[item_id] = {"source": source_hash, "local": local_hash}
        local_paths.append(destination)
    mark(timing, "input_stage_and_hash", started)
    free_after_stage = verify_reserve(scratch, input_bytes * 8)

    gpu_idle(args.gpu)
    executable = args.runtime_root / "venv/bin/all-in-one-infer"
    command = [
        str(executable), *(str(path) for path in local_paths),
        "-o", str(local_output / "structure"), "-m", "harmonix-all", "-d", "cuda", "-k",
        "--demix-dir", str(local_output / "demix"), "--spec-dir", str(local_output / "spec"),
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log_path = scratch / "allinone.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    mark(timing, "compute", started)
    if completed.returncode:
        raise RuntimeError(f"All-In-One failed with {completed.returncode}; see {log_path}")

    expected_relatives = []
    for item_id, _ in selected:
        expected_relatives.extend([
            Path("structure") / f"{item_id}.json",
            Path("spec") / f"{item_id}.npy",
            *(Path("demix/htdemucs") / item_id / f"{stem}.wav" for stem in STEMS),
        ])
    missing = [str(path) for relative in expected_relatives if not (path := local_output / relative).is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"Local inference missing {len(missing)} outputs; first={missing[0]}")
    free_after_compute = verify_reserve(scratch)

    started = time.monotonic()
    args.nfs_output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-a", "--ignore-existing", f"{local_output}/", f"{args.nfs_output_root}/",
    ], check=True)
    mark(timing, "nfs_publish", started)

    started = time.monotonic()
    output_hashes: dict[str, str] = {}
    for relative in expected_relatives:
        local_path = local_output / relative
        nfs_path = args.nfs_output_root / relative
        if not nfs_path.is_file():
            raise RuntimeError(f"Published output missing: {nfs_path}")
        local_hash, nfs_hash = digest(local_path), digest(nfs_path)
        if local_hash != nfs_hash:
            raise RuntimeError(f"Published output hash mismatch: {relative}")
        output_hashes[str(relative)] = local_hash
    mark(timing, "publication_hash_verify", started)

    payload = {
        "schema_version": 1,
        "status": "verified",
        "rows": len(selected),
        "item_ids": [item_id for item_id, _ in selected],
        "prepared_manifest_sha256": digest(manifest),
        "shard_sha256": digest(shard),
        "input_bytes": input_bytes,
        "free_bytes": {"before": free_before, "after_stage": free_after_stage, "after_compute": free_after_compute},
        "allinone_options": {
            "model": "harmonix-all", "device": "cuda", "keep_byproducts": True,
            "demucs_model": "htdemucs", "demucs_shifts": 1,
            "demucs_shift_rng": "python_stdlib_random_unseeded_established_default",
            "demucs_overlap": 0.25, "demucs_fp16": False,
            "reproducibility_boundary": "exact_from_frozen_published_stems_not_bitwise_demucs_regeneration",
        },
        "command": command,
        "timing": timing,
        "input_hashes": input_hashes,
        "published_output_hashes": output_hashes,
        "log_path": str(log_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"item_ids", "input_hashes", "published_output_hashes", "command"}}, indent=2))

    if args.clear_local_after_verify:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
