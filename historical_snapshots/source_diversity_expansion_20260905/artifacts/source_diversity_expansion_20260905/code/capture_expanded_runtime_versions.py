#!/usr/bin/env python3
"""Capture non-secret package versions and bind them into the inference contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "numpy", "scipy", "librosa", "soundfile", "demucs", "demucs-infer", "all-in-one-infer",
    "beat_this", "torch", "torchaudio",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload: dict[str, object] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_sha256": sha256(Path(sys.executable)),
        "platform": platform.platform(),
        "packages": versions,
        "collection_method": "importlib.metadata.version; no pip freeze or direct_url metadata",
    }
    try:
        import torch
        payload["torch_runtime"] = {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        payload["torch_runtime_error"] = f"{type(exc).__name__}:{exc}"
    atomic_json(args.output_json, payload)
    if args.contract:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        contract["library_runtime_snapshot"] = payload
        contract["library_runtime_snapshot_path"] = str(args.output_json)
        contract["library_runtime_snapshot_sha256"] = sha256(args.output_json)
        atomic_json(args.contract, contract)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
