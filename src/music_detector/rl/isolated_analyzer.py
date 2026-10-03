"""Strict subprocess bridge between independently pinned training/analysis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .rewards import FEATURE_IMPLEMENTATION_SHA256


class IsolatedAnalyzer:
    def __init__(self, config) -> None:
        self.config = config
        python = Path(config.analyzer_python)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RuntimeError("Isolated analysis Python is missing or not executable")
        self._worker_sha256 = hashlib.sha256(
            Path(__file__).with_name("analyzer_worker.py").read_bytes()).hexdigest()
        self.receipt = self._call("status")["receipt"]

    def _call(self, action: str, path: Path | None = None) -> dict:
        cfg = self.config
        request = {"action": action, "families": cfg.families,
                   "device": cfg.device, "seed": cfg.analyzer_seed,
                   "bundle_sha256": cfg.bundle_sha256,
                   "cache_dir": cfg.analyzer_cache_dir}
        if path is not None:
            request["path"] = str(path.resolve())
        env = dict(os.environ)
        env.update(MUSIC_DETECTOR_NEURAL_CACHE=cfg.analyzer_cache_dir,
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        completed = subprocess.run(
            [cfg.analyzer_python, "-m", "music_detector.rl.analyzer_worker"],
            input=json.dumps(request, allow_nan=False), capture_output=True,
            text=True, timeout=cfg.analyzer_timeout_s, env=env, check=False)
        if completed.returncode:
            raise RuntimeError("Isolated frozen analyzer failed: " + completed.stderr[-3000:])
        try:
            response = json.loads(completed.stdout)
            receipt = response["receipt"]
            if not isinstance(receipt, dict):
                raise ValueError("Receipt is not an object")
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError("Malformed isolated analysis response") from error
        expected = {"backend": "isolated_frozen_analyzer_v1",
                    "families": cfg.families, "device": cfg.device,
                    "seed": cfg.analyzer_seed, "bundle_sha256": cfg.bundle_sha256,
                    "feature_implementation_sha256": dict(FEATURE_IMPLEMENTATION_SHA256),
                    "worker_sha256": self._worker_sha256}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Isolated analysis release/protocol receipt mismatch")
        if set(cfg.families) & {"S", "D", "R", "P"} and not receipt.get("runtime", {}).get("ready"):
            raise RuntimeError("Isolated analysis runtime is not ready")
        if hasattr(self, "receipt") and receipt != self.receipt:
            raise RuntimeError("Isolated analysis runtime changed after preflight")
        return response

    def __call__(self, path, families, *, device):
        if list(families) != self.config.families or device != self.config.device:
            raise ValueError("Analysis request differs from preflight configuration")
        return self._call("analyze", Path(path))["result"]
