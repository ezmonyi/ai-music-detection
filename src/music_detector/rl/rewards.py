"""A conservative, frozen-artifact reward bridge for online music RL.

This module deliberately keeps the training objective narrower than the
detector UI.  A completion receives an absolute, bounded negative frozen raw
artifact score.  A same-seed baseline is used for cheap quality gates and for
an optional diagnostic raw-score delta; the delta is *never* the training
reward.  At initialization a candidate and its baseline can be identical, so
using that delta as the objective would give every GRPO group zero advantage.

The default analyzer is the existing :func:`music_detector.analysis.analyze`
entry point.  Tests and light-weight callers may inject an analyzer with the
same ``(path, families, **kwargs)`` call shape.  No model setup, download, or
fallback is performed by this bridge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from ..scoring import FAMILIES


FAMILY_ORDER = ("S", "D", "R", "P", "F", "H", "SC")
_NEURAL_FAMILIES = frozenset(("S", "D", "R", "P"))

# These hashes are the released implementation boundary.  The reward must not
# silently change because a feature extractor or calibration transform changed
# in the working tree.  The deployment bundle hash is supplied by the caller so
# deployments can make the expected release explicit in configuration.
FEATURE_IMPLEMENTATION_SHA256 = {
    "analysis.py": "94de5fa278e7e8393df22036bd2fdbaf3fc88661ac588afbd8d02cf996e9bf1c",
    "audio.py": "7739c67e3fef1afe9ec899601335f7fbf696e36d1437abf7029499c1afb2cd74",
    "calibration.py": "e96865300877a5394d9af35131c1148d0b4b430783cb63a32bfb6636aa806c18",
    "scoring.py": "d1382ff0d56978d01307d5483267d5e66e499e9b4d446ec3040331e4e55e2a9f",
    "phase_features.py": "f1598ecb513ae67431571e586e436840f80114ac2b6d04fce3a46e8eeb0a5ac5",
    "musical_features.py": "72d54ceb5266e2ff0d1ce5bdbcfa7e8c93611784a03e40d3255672020f8083",
    "stereo_candidate_v1.py": "132bd5fe225258b72b913238bd456e3b18279b6b5a664b2191b0292aede9481c",
    "neural.py": "370a234d1a569481521cb2128b66769d0a04ad9f3f8e1d57bb347138290d69b1",
    "neural_features.py": "60f2131d012b9da56303a22f9892c5f12dbbc5a6987b0b621c3564252dd9e7c0",
    "expanded_feature_definitions.py": "8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00",
}
_ASSET_SHA256 = {
    "deployment_models.json": "05dbe17a012bfec5391dd7c0b203549929e26d1f4612874332000403f71aaf53",
    "neural_runtime.json": "9ffcb47f043225f7678d9764ba8f52c5209866c7dd629f4a3633995c7b271536",
    "demucs_frequency_bias.npz": "bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e",
}
_CALIBRATION_PROTOCOL_SHA256 = "2567e86df60658dae99f97a3aca7d3d7f117ac0652286c3c935dbbb04cc520d3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class RewardGuards:
    """Engineering validity limits, not claims about musical quality.

    The limits are intentionally conservative and visible in every result.
    They reject common reward-gaming shortcuts (silence, tail-only audio,
    clipping, low-pass output, and duplicated channels) before a detector
    score is consulted.  They should be revalidated against the target model's
    native-output distribution before a production run.
    """

    minimum_duration_seconds: float = 30.0
    minimum_sample_rate_hz: int = 24_000
    minimum_rms_dbfs: float = -60.0
    maximum_rms_drift_db: float = 12.0
    minimum_active_fraction: float = 0.10
    maximum_active_fraction_drift: float = 0.45
    maximum_peak: float = 1.000001
    maximum_clipped_fraction: float = 1.0e-5
    bandwidth_low_hz: float = 8_000.0
    bandwidth_high_hz: float = 20_000.0
    minimum_high_band_fraction: float = 1.0e-4
    maximum_bandwidth_collapse_db: float = 12.0
    minimum_side_energy_fraction: float = 1.0e-5
    maximum_stereo_collapse_db: float = 12.0
    score_scale: float = 2.0
    invalid_reward: float = -1.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_duration_seconds",
            "minimum_rms_dbfs",
            "maximum_rms_drift_db",
            "minimum_active_fraction",
            "maximum_active_fraction_drift",
            "maximum_peak",
            "maximum_clipped_fraction",
            "bandwidth_low_hz",
            "bandwidth_high_hz",
            "minimum_high_band_fraction",
            "maximum_bandwidth_collapse_db",
            "minimum_side_energy_fraction",
            "maximum_stereo_collapse_db",
            "score_scale",
            "invalid_reward",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not _finite_number(value):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.minimum_sample_rate_hz, (bool, np.bool_)) or not isinstance(
            self.minimum_sample_rate_hz, (int, np.integer)
        ):
            raise ValueError("minimum_sample_rate_hz must be an integer")
        if self.minimum_duration_seconds <= 0:
            raise ValueError("minimum_duration_seconds must be positive")
        if self.minimum_sample_rate_hz <= 0:
            raise ValueError("minimum_sample_rate_hz must be positive")
        if not (0 < self.minimum_active_fraction <= 1):
            raise ValueError("minimum_active_fraction must be in (0, 1]")
        if self.maximum_active_fraction_drift < 0:
            raise ValueError("maximum_active_fraction_drift must be nonnegative")
        if self.maximum_rms_drift_db < 0:
            raise ValueError("maximum_rms_drift_db must be nonnegative")
        if self.maximum_peak <= 0 or self.maximum_clipped_fraction < 0:
            raise ValueError("invalid peak/clipping guard")
        if self.bandwidth_low_hz <= 0 or self.bandwidth_high_hz <= self.bandwidth_low_hz:
            raise ValueError("invalid bandwidth band")
        if self.minimum_high_band_fraction < 0 or self.minimum_side_energy_fraction < 0:
            raise ValueError("energy fractions must be nonnegative")
        if self.maximum_bandwidth_collapse_db < 0 or self.maximum_stereo_collapse_db < 0:
            raise ValueError("collapse limits must be nonnegative")
        if self.score_scale <= 0 or not math.isfinite(self.score_scale):
            raise ValueError("score_scale must be finite and positive")
        if self.invalid_reward > -1.0:
            raise ValueError("invalid_reward must be no better than every valid reward (<= -1)")


@dataclass(frozen=True)
class RewardResult:
    reward: float
    valid: bool
    diagnostics: dict[str, Any]


class _GateFailure(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


Analyzer = Callable[..., Mapping[str, Any]]


class ArtifactReward:
    """Score an audio completion against a frozen detector release.

    Parameters
    ----------
    families:
        One or more predictive families in canonical ``S,D,R,P,F,H,SC``
        order.  ``BC`` and diagnostic/missingness-only modes are not admitted.
    bundle_sha256:
        Expected SHA-256 of ``src/music_detector/assets/deployment_models.json``.
        A mismatch is a setup/configuration error and raises at construction.
    device:
        Device forwarded to the default analyzer.  No silent fallback occurs.
    guard:
        :class:`RewardGuards` or a mapping of its fields.
    analyzer:
        Optional injected analyzer for tests.  It receives the temporary WAV
        path, selected families, and ``device=...``; its result must contain
        ``feature_values`` with every selected model column finite.
    """

    def __init__(
        self,
        families: Sequence[str],
        bundle_sha256: str,
        device: str = "cpu",
        guard: RewardGuards | Mapping[str, Any] | None = None,
        analyzer: Analyzer | None = None,
    ) -> None:
        requested = list(families)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("families must be a nonempty set")
        if any(family not in FAMILY_ORDER for family in requested):
            raise ValueError("reward families must be predictive S,D,R,P,F,H,SC; BC is research-only")
        selected = [family for family in FAMILY_ORDER if family in requested]
        if not isinstance(bundle_sha256, str) or len(bundle_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in bundle_sha256
        ):
            raise ValueError("bundle_sha256 must be a lowercase SHA-256 hex digest")

        if guard is None:
            checked_guard = RewardGuards()
        elif isinstance(guard, RewardGuards):
            checked_guard = guard
        elif isinstance(guard, Mapping):
            allowed = {field.name for field in fields(RewardGuards)}
            unknown = set(guard) - allowed
            if unknown:
                raise ValueError("unknown reward guard fields: " + ", ".join(sorted(unknown)))
            checked_guard = RewardGuards(**dict(guard))
        else:
            raise TypeError("guard must be RewardGuards, a mapping, or None")

        self.families = tuple(selected)
        self.bundle_sha256 = bundle_sha256
        self.device = device
        self.guard = checked_guard
        self._analyzer = self._default_analyzer if analyzer is None else analyzer
        self._custom_analyzer = analyzer is not None

        package_root = Path(__file__).resolve().parents[1]
        self._package_root = package_root
        self._bundle_path = package_root / "assets" / "deployment_models.json"
        self._verify_release_boundary()

        import json

        self._bundle = json.loads(self._bundle_path.read_text(encoding="utf-8"))
        if self._bundle.get("protocol_sha256") != _CALIBRATION_PROTOCOL_SHA256:
            raise RuntimeError("frozen calibration protocol identity mismatch")
        key = "+".join(self.families)
        if key not in self._bundle.get("models", {}):
            raise RuntimeError(f"frozen deployment bundle lacks model combination {key}")
        envelope = self._bundle["models"][key]
        self._model_sha256 = envelope.get("sha256")
        if not isinstance(self._model_sha256, str):
            raise RuntimeError("frozen deployment model has no envelope hash")

        # The default analyzer has an explicit dependency/weight setup check.
        # An injected analyzer is deliberately exempt so unit tests can use
        # synthetic feature values without downloading analysis weights.
        if not self._custom_analyzer and self._neural_selected:
            from ..neural import runtime_status

            status = runtime_status(families=list(self.families))
            if not status["ready"]:
                raise RuntimeError("analysis runtime is not ready: " + "; ".join(status["errors"]))

    @property
    def _neural_selected(self) -> bool:
        return bool(set(self.families) & _NEURAL_FAMILIES)

    def _verify_release_boundary(self) -> None:
        actual_bundle = _sha256(self._bundle_path)
        if actual_bundle != self.bundle_sha256:
            raise RuntimeError(
                "deployment bundle checksum mismatch: "
                f"expected {self.bundle_sha256}, found {actual_bundle}"
            )

        algorithms = self._package_root / "algorithms"
        module_paths = {
            "analysis.py": self._package_root / "analysis.py",
            "audio.py": self._package_root / "audio.py",
            "calibration.py": self._package_root / "calibration.py",
            "scoring.py": self._package_root / "scoring.py",
        }
        if "F" in self.families:
            module_paths["phase_features.py"] = algorithms / "phase_features.py"
        if "H" in self.families:
            module_paths["musical_features.py"] = algorithms / "musical_features.py"
        if "SC" in self.families:
            module_paths["stereo_candidate_v1.py"] = algorithms / "stereo_candidate_v1.py"
        if self._neural_selected:
            module_paths.update(
                {
                    "neural.py": self._package_root / "neural.py",
                    "neural_features.py": self._package_root / "neural_features.py",
                    "expanded_feature_definitions.py": algorithms / "expanded_feature_definitions.py",
                }
            )
        for name, expected in module_paths.items():
            if not name in FEATURE_IMPLEMENTATION_SHA256:
                raise RuntimeError(f"no frozen implementation hash for {name}")
            path = module_paths[name]
            if not path.is_file():
                raise RuntimeError(f"missing frozen implementation file: {path}")
            actual = _sha256(path)
            if actual != FEATURE_IMPLEMENTATION_SHA256[name]:
                raise RuntimeError(f"feature implementation checksum mismatch: {name}")

        assets = self._package_root / "assets"
        asset_paths = {"deployment_models.json": self._bundle_path}
        if self._neural_selected:
            asset_paths.update(
                {
                    "neural_runtime.json": assets / "neural_runtime.json",
                    "demucs_frequency_bias.npz": assets / "demucs_frequency_bias.npz",
                }
            )
        for name, path in asset_paths.items():
            expected = _ASSET_SHA256[name]
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"frozen analysis asset checksum mismatch: {name}")

    @staticmethod
    def _default_analyzer(path: Path, families: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        from ..analysis import analyze

        return analyze(path, list(families), **kwargs)

    @staticmethod
    def _as_stereo(audio: Any, label: str) -> np.ndarray:
        values = np.asarray(audio)
        if values.ndim != 2 or 2 not in values.shape:
            raise _GateFailure("invalid_shape", f"{label} must have shape [2,N] or [N,2]")
        if not np.issubdtype(values.dtype, np.floating):
            raise _GateFailure("invalid_dtype", f"{label} must contain floating-point samples")
        if values.shape[0] == 2 and values.shape[1] != 2:
            values = values.T
        elif values.shape[1] == 2:
            values = values
        else:
            raise _GateFailure("ambiguous_shape", f"{label} channel axis is ambiguous")
        values = np.asarray(values, dtype=np.float64)
        if values.shape[1] != 2:
            raise _GateFailure("invalid_channels", f"{label} must be native stereo")
        if not np.isfinite(values).all():
            raise _GateFailure("nonfinite_audio", f"{label} contains non-finite samples")
        return np.ascontiguousarray(values)

    def _quality(self, values: np.ndarray, sample_rate: int, label: str) -> dict[str, float | int]:
        duration = values.shape[0] / sample_rate
        if duration < self.guard.minimum_duration_seconds:
            raise _GateFailure("short_audio", f"{label} is shorter than the Native30 contract", duration_seconds=duration)
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > self.guard.maximum_peak:
            raise _GateFailure("clipping_peak", f"{label} exceeds the allowed peak", peak=peak)
        clipped_fraction = float(np.mean(np.max(np.abs(values), axis=1) >= 1.0 - 1.0e-6))
        if clipped_fraction > self.guard.maximum_clipped_fraction:
            raise _GateFailure(
                "clipping_fraction",
                f"{label} contains too many full-scale samples",
                clipped_fraction=clipped_fraction,
            )

        mono = values.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        rms_dbfs = 20.0 * math.log10(max(rms, np.finfo(float).tiny))
        if rms_dbfs < self.guard.minimum_rms_dbfs:
            raise _GateFailure("silence", f"{label} is below the RMS floor", rms_dbfs=rms_dbfs)

        frame_length = min(4096, values.shape[0])
        hop = max(1, frame_length // 2)
        frame_count = 1 + max(0, (values.shape[0] - frame_length) // hop)
        frames = np.lib.stride_tricks.as_strided(
            mono,
            shape=(frame_count, frame_length),
            strides=(hop * mono.strides[0], mono.strides[0]),
            writeable=False,
        )
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        active_floor = max(10.0 ** (self.guard.minimum_rms_dbfs / 20.0), float(frame_rms.max()) * 0.01)
        active_fraction = float(np.mean(frame_rms >= active_floor))
        if active_fraction < self.guard.minimum_active_fraction:
            raise _GateFailure(
                "mostly_silent",
                f"{label} has too little active time",
                active_fraction=active_fraction,
            )

        side = 0.5 * (values[:, 0] - values[:, 1])
        mid = 0.5 * (values[:, 0] + values[:, 1])
        side_energy = float(np.mean(np.square(side, dtype=np.float64)))
        mid_energy = float(np.mean(np.square(mid, dtype=np.float64)))
        side_fraction = side_energy / max(mid_energy + side_energy, np.finfo(float).tiny)
        if side_fraction < self.guard.minimum_side_energy_fraction:
            raise _GateFailure(
                "stereo_collapse",
                f"{label} is effectively mono",
                side_energy_fraction=side_fraction,
            )

        if sample_rate < self.guard.minimum_sample_rate_hz:
            raise _GateFailure(
                "sample_rate",
                f"{label} cannot support the bandwidth guard at this sample rate",
                sample_rate_hz=sample_rate,
            )
        frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
        spectrum = np.fft.rfft(frames * np.hanning(frame_length), axis=1)
        power = np.mean(np.abs(spectrum) ** 2, axis=0)
        total_mask = (frequencies >= 80.0) & (frequencies < min(self.guard.bandwidth_high_hz, sample_rate / 2.0))
        high_mask = (frequencies >= self.guard.bandwidth_low_hz) & (frequencies < min(self.guard.bandwidth_high_hz, sample_rate / 2.0))
        if not np.any(high_mask) or not np.any(total_mask):
            raise _GateFailure("bandwidth_unassessable", f"{label} has no measurable high band")
        high_fraction = float(power[high_mask].sum() / max(power[total_mask].sum(), np.finfo(float).tiny))
        if high_fraction < self.guard.minimum_high_band_fraction:
            raise _GateFailure(
                "lowpass",
                f"{label} has insufficient high-band energy",
                high_band_fraction=high_fraction,
            )
        return {
            "duration_seconds": float(duration),
            "rms_dbfs": float(rms_dbfs),
            "active_fraction": float(active_fraction),
            "peak": float(peak),
            "clipped_fraction": float(clipped_fraction),
            "side_energy_fraction": float(side_fraction),
            "high_band_fraction": float(high_fraction),
            "sample_rate_hz": int(sample_rate),
        }

    def _paired_gates(self, candidate: np.ndarray, baseline: np.ndarray, sample_rate: int) -> dict[str, Any]:
        try:
            candidate_quality = self._quality(candidate, sample_rate, "candidate")
        except _GateFailure:
            raise
        try:
            baseline_quality = self._quality(baseline, sample_rate, "baseline")
        except _GateFailure as error:
            raise _GateFailure(
                "baseline_ineligible",
                f"baseline is not eligible: {error}",
                baseline_failure=error.code,
                baseline_message=str(error),
                **error.details,
            ) from error
        rms_delta = abs(float(candidate_quality["rms_dbfs"]) - float(baseline_quality["rms_dbfs"]))
        if rms_delta > self.guard.maximum_rms_drift_db:
            raise _GateFailure("rms_drift", "candidate RMS drifts too far from its paired baseline", rms_delta_db=rms_delta)
        active_delta = abs(float(candidate_quality["active_fraction"]) - float(baseline_quality["active_fraction"]))
        if active_delta > self.guard.maximum_active_fraction_drift:
            raise _GateFailure("activity_drift", "candidate active-time coverage drifts too far", active_delta=active_delta)

        band_db = 10.0 * math.log10(
            max(float(candidate_quality["high_band_fraction"]), np.finfo(float).tiny)
            / max(float(baseline_quality["high_band_fraction"]), np.finfo(float).tiny)
        )
        if band_db < -self.guard.maximum_bandwidth_collapse_db:
            raise _GateFailure("bandwidth_collapse", "candidate high-band energy collapsed against baseline", bandwidth_delta_db=band_db)
        stereo_db = 10.0 * math.log10(
            max(float(candidate_quality["side_energy_fraction"]), np.finfo(float).tiny)
            / max(float(baseline_quality["side_energy_fraction"]), np.finfo(float).tiny)
        )
        if stereo_db < -self.guard.maximum_stereo_collapse_db:
            raise _GateFailure("stereo_collapse", "candidate stereo side energy collapsed against baseline", stereo_delta_db=stereo_db)
        return {
            "candidate": candidate_quality,
            "baseline": baseline_quality,
            "rms_delta_db": float(rms_delta),
            "active_delta": float(active_delta),
            "bandwidth_delta_db": float(band_db),
            "stereo_delta_db": float(stereo_db),
        }

    def _write_and_analyze(self, values: np.ndarray, sample_rate: int, directory: Path, name: str) -> Mapping[str, Any]:
        path = directory / f"{name}.wav"
        sf.write(path, np.asarray(values, dtype=np.float32), sample_rate, subtype="FLOAT")
        result = self._analyzer(path, list(self.families), device=self.device)
        if not isinstance(result, Mapping):
            raise ValueError("analyzer result must be a mapping")
        return result

    def _extract_raw_score(self, result: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        feature_values = result.get("feature_values")
        if not isinstance(feature_values, Mapping):
            raise ValueError("analyzer result lacks feature_values")
        columns = [column for family in self.families for column in FAMILIES[family]]
        if set(feature_values) != set(columns):
            raise ValueError("analyzer feature_values do not exactly match selected model columns")
        if not all(_finite_number(feature_values[column]) for column in columns):
            raise ValueError("selected artifact family is incomplete or non-finite")

        cards = result.get("features")
        if isinstance(cards, list):
            allowed_status = {"ok", "eligible"}
            by_id = {card.get("id"): card for card in cards if isinstance(card, Mapping)}
            for family in self.families:
                status = by_id.get(family, {}).get("status")
                if status is not None and status not in allowed_status:
                    raise ValueError(f"artifact family {family} is not quality-eligible: {status}")

        from ..calibration import predict_deployment

        normalized_features = {column: float(feature_values[column]) for column in columns}
        scored = predict_deployment(self._bundle, normalized_features, list(self.families))
        if scored.get("missing_features"):
            raise ValueError("deployment scoring unexpectedly used missingness")
        raw_score = scored.get("raw_score")
        if not _finite_number(raw_score):
            raise ValueError("deployment raw score is non-finite")
        return float(raw_score), {
            "model_sha256": self._model_sha256,
            "calibration_protocol_sha256": _CALIBRATION_PROTOCOL_SHA256,
            "prediction": {
                "probability": scored.get("probability"),
                "decision": scored.get("decision"),
            },
            "feature_values": normalized_features,
            "vocal_activity": result.get("provenance", {}).get(
                "neural_measurement_metadata", {}).get("vocal_activity"),
            "isolated_analyzer_receipt": result.get("isolated_analyzer_receipt"),
        }

    def score(
        self,
        audio: Any,
        sample_rate: int,
        baseline_audio: Any,
    ) -> RewardResult:
        """Return one bounded reward and auditable diagnostics.

        ``audio`` and ``baseline_audio`` may be ``[2,N]`` or ``[N,2]``; both
        must resolve to the same native-stereo shape.  Invalid candidate audio
        returns ``invalid_reward`` and does not invoke the analyzer.  Frozen
        setup/checksum failures are raised at construction (or propagated for a
        runtime that disappears after construction).
        """
        diagnostics: dict[str, Any] = {
            "reward_kind": "bounded_negative_frozen_raw_score",
            "delta_scope": "paired_baseline_diagnostic_only",
            "families": list(self.families),
            "device": self.device,
            "bundle_sha256": self.bundle_sha256,
            "model_sha256": self._model_sha256,
            "guard_config": asdict(self.guard),
            "feature_implementation_sha256": dict(FEATURE_IMPLEMENTATION_SHA256),
        }
        try:
            if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
                raise _GateFailure("sample_rate", "sample_rate must be a positive integer")
            candidate = self._as_stereo(audio, "candidate")
            baseline = self._as_stereo(baseline_audio, "baseline")
            if candidate.shape != baseline.shape:
                raise _GateFailure("shape_mismatch", "candidate and baseline must have identical shapes")
            paired = self._paired_gates(candidate, baseline, int(sample_rate))
            diagnostics["quality"] = paired
        except _GateFailure as error:
            diagnostics.update({"validity": "invalid", "failure": error.code, "message": str(error), **error.details})
            return RewardResult(float(self.guard.invalid_reward), False, diagnostics)

        with tempfile.TemporaryDirectory(prefix="music-artifact-reward-") as temporary:
            directory = Path(temporary)
            try:
                candidate_result = self._write_and_analyze(candidate, int(sample_rate), directory, "candidate")
                raw_score, score_details = self._extract_raw_score(candidate_result)
            except (ValueError, sf.SoundFileError) as error:
                diagnostics.update({"validity": "invalid", "failure": "analysis", "message": str(error)})
                return RewardResult(float(self.guard.invalid_reward), False, diagnostics)
            try:
                baseline_result = self._write_and_analyze(baseline, int(sample_rate), directory, "baseline")
                baseline_raw_score, baseline_details = self._extract_raw_score(baseline_result)
            except (ValueError, sf.SoundFileError) as error:
                diagnostics.update({"validity": "invalid", "failure": "baseline_analysis", "message": str(error)})
                return RewardResult(float(self.guard.invalid_reward), False, diagnostics)

        reward = -math.tanh(raw_score / self.guard.score_scale)
        diagnostics.update(
            {
                "validity": "valid",
                "raw_score": float(raw_score),
                "baseline_raw_score": float(baseline_raw_score),
                "raw_score_delta": float(raw_score - baseline_raw_score),
                "reward": float(reward),
                "score": score_details,
                "baseline_score": baseline_details,
            }
        )
        return RewardResult(float(reward), True, diagnostics)


__all__ = ["ArtifactReward", "RewardGuards", "RewardResult", "FEATURE_IMPLEMENTATION_SHA256"]
