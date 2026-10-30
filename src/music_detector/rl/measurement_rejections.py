"""Opt-in adapter for one observed BeatThis input-measurement failure.

This module is not registered in any production CLI. Original fatal defaults,
frozen worker/extractor files, successful measurements and model receipts stay
unchanged. The adapter fabricates no score: it can only re-raise a verified,
known input error as ValueError for the existing invalid-reward path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


POLICY = "beat_this_downbeat_subset_input_rejection_v1"
ORIGINAL_WORKER_SHA256 = "80b184cbc6cc85411a1664062390bd44b4afb514ff28db495d689d0a06df2105"
ORIGINAL_BRIDGE_SHA256 = "ad3bb9949e96ef58c27b7925ea8ffeecdf93cad06a6811a2b43f2929f2d5c824"
_FOOTER = "ValueError: Not all downbeats are beats."


class KnownBeatMeasurementRejected(ValueError):
    """No valid R measurement was produced; this is never an admissible score."""


def known_beat_input_failure(error: BaseException) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    text = str(error).replace("\\", "/")
    if not text.startswith("Isolated frozen analyzer failed: "):
        return False
    marker = "Traceback (most recent call last):"
    if marker not in text:
        return False
    tail = text.rsplit(marker, 1)[1]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines or lines[-1] != _FOOTER:
        return False
    frames = (
        r'File "[^"\n]*/music_detector/rl/analyzer_worker\.py", line \d+, in execute',
        r'File "[^"\n]*/music_detector/neural\.py", line \d+, in _track_beats',
        r'File "[^"\n]*/beat_this/utils\.py", line \d+, in save_beat_tsv',
        r'File "[^"\n]*/beat_this/utils\.py", line \d+, in infer_beat_numbers',
    )
    return all(re.search(pattern, tail) is not None for pattern in frames)


def _check_frozen_sources() -> None:
    for name, expected in (("analyzer_worker.py", ORIGINAL_WORKER_SHA256),
                           ("isolated_analyzer.py", ORIGINAL_BRIDGE_SHA256)):
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Known-input adapter requires the original frozen " + name)


class _MeasurementErrorAdapter:
    def __init__(self, delegate):
        _check_frozen_sources()
        self._delegate = delegate
        self._receipt = deepcopy(delegate.receipt)
        if self._receipt.get("worker_sha256") != ORIGINAL_WORKER_SHA256:
            raise RuntimeError("Known-input adapter worker identity mismatch")
        self.error_policy_receipt = {
            "policy": POLICY, "worker_sha256": ORIGINAL_WORKER_SHA256,
            "adapter_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_bridge_sha256": ORIGINAL_BRIDGE_SHA256,
            "original_analysis_receipt_sha256": hashlib.sha256(
                json.dumps(self._receipt, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "valid_measurements_passed_through": True,
            "unknown_runtime_errors_remain_fatal": True,
            "known_rejection_produces_no_features_or_score": True,
            "fresh_original_worker_status_required_after_known_failure": True,
        }

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def _unchanged(self):
        _check_frozen_sources()
        if self._delegate.receipt != self._receipt:
            raise RuntimeError("Analyzer receipt changed under the known-input adapter")

    def __call__(self, path, families, *, device):
        self._unchanged()
        try:
            result = self._delegate(path, families, device=device)
        except RuntimeError as error:
            if not known_beat_input_failure(error):
                raise
            self._unchanged()
            # The failing subprocess did not return a receipt. Re-check the
            # original worker's readiness/identity rather than classifying
            # every subprocess failure as missingness.
            status = self._delegate._call("status")
            self._unchanged()
            if status.get("receipt") != self._receipt:
                raise RuntimeError("Post-failure frozen analyzer status changed") from error
            raise KnownBeatMeasurementRejected(
                "Known BeatThis input measurement rejected; no feature or score was fabricated. "
                + str(error)) from error
        self._unchanged()
        return result


def create_measurement_safe_analyzer(config, *, policy: str):
    """Explicit optional factory, not a replacement of the original defaults.

Record ``error_policy_receipt`` in a separate immutable sidecar before using
this adapter in a future experiment. The present failed cloud run never used it.
"""
    if policy != POLICY:
        raise ValueError("An explicit supported measurement-error policy is required")
    from .isolated_analyzer import IsolatedAnalyzer
    return _MeasurementErrorAdapter(IsolatedAnalyzer(config))
