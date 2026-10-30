"""Standard-library-only regressions; no model, GPU or paid node is used."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from music_detector.rl import measurement_rejections as module


TRACE = '''Isolated frozen analyzer failed: progress output
Traceback (most recent call last):
  File "/root/source/src/music_detector/rl/analyzer_worker.py", line 53, in execute
    result = analyze(path)
  File "/root/source/src/music_detector/neural.py", line 195, in _track_beats
    save_beat_tsv(beats, downbeats, output)
  File "/root/runtime/beat_this/utils.py", line 94, in save_beat_tsv
    numbers = infer_beat_numbers(beats, downbeats)
  File "/root/runtime/beat_this/utils.py", line 41, in infer_beat_numbers
    raise ValueError("Not all downbeats are beats.")
ValueError: Not all downbeats are beats.
'''


class Delegate:
    def __init__(self, error=None):
        self.error = error
        self.receipt = {"worker_sha256": module.ORIGINAL_WORKER_SHA256,
                        "runtime": {"ready": True, "versions": {"torch": "fixture"}}}
        self.result = {"feature_values": {"fixture": 1.25}}
        self.status_calls = 0
        self.status_error = None
        self.return_changed_status = False

    def __call__(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.result

    def _call(self, action):
        assert action == "status"
        self.status_calls += 1
        if self.status_error:
            raise self.status_error
        receipt = deepcopy(self.receipt)
        if self.return_changed_status:
            receipt["runtime"]["ready"] = False
        return {"receipt": receipt}


class MeasurementRecoveryTests(unittest.TestCase):
    def test_exact_observed_trace_is_recognized(self):
        self.assertTrue(module.known_beat_input_failure(RuntimeError(TRACE)))

    def test_message_without_worker_and_library_frames_is_not_whitelisted(self):
        self.assertFalse(module.known_beat_input_failure(RuntimeError("Isolated frozen analyzer failed: ValueError: Not all downbeats are beats.")))
        self.assertFalse(module.known_beat_input_failure(RuntimeError(TRACE.replace("in _track_beats", "in unknown"))))

    def test_cuda_malformed_and_other_value_errors_remain_fatal(self):
        for message in ("CUDA out of memory", "Malformed isolated analysis response",
                        "Isolated analysis release/protocol receipt mismatch",
                        TRACE.replace("ValueError: Not all downbeats are beats.", "RuntimeError: CUDA out of memory"),
                        TRACE.replace("ValueError: Not all downbeats are beats.", "ValueError: Unknown schema")):
            self.assertFalse(module.known_beat_input_failure(RuntimeError(message)))

    def test_success_returns_original_measurement_without_status_probe(self):
        delegate = Delegate();adapter = module._MeasurementErrorAdapter(delegate)
        self.assertIs(adapter("x.wav", ["R"], device="cuda:0"), delegate.result)
        self.assertEqual(delegate.status_calls, 0)
        self.assertEqual(adapter.receipt, delegate.receipt)

    def test_known_input_error_becomes_value_error_only_after_fresh_status(self):
        delegate = Delegate(RuntimeError(TRACE));adapter = module._MeasurementErrorAdapter(delegate)
        with self.assertRaises(module.KnownBeatMeasurementRejected) as caught:
            adapter("x.wav", ["R"], device="cuda:0")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(delegate.status_calls, 1)

    def test_unknown_subprocess_error_is_unchanged(self):
        failure = RuntimeError("Isolated frozen analyzer failed: missing weights")
        delegate = Delegate(failure);adapter = module._MeasurementErrorAdapter(delegate)
        with self.assertRaises(RuntimeError) as caught:adapter("x.wav", ["R"], device="cuda:0")
        self.assertIs(caught.exception, failure);self.assertEqual(delegate.status_calls, 0)

    def test_changed_post_failure_receipt_is_fatal(self):
        delegate = Delegate(RuntimeError(TRACE));delegate.return_changed_status = True
        adapter = module._MeasurementErrorAdapter(delegate)
        with self.assertRaisesRegex(RuntimeError, "status changed"):
            adapter("x.wav", ["R"], device="cuda:0")

    def test_failed_post_failure_status_is_fatal(self):
        delegate = Delegate(RuntimeError(TRACE));delegate.status_error = RuntimeError("runtime not ready")
        adapter = module._MeasurementErrorAdapter(delegate)
        with self.assertRaisesRegex(RuntimeError, "not ready"):adapter("x.wav", ["R"], device="cuda:0")

    def test_receipt_mutation_cannot_be_hidden(self):
        delegate = Delegate();adapter = module._MeasurementErrorAdapter(delegate)
        delegate.receipt["runtime"]["ready"] = False
        with self.assertRaisesRegex(RuntimeError, "receipt changed"):adapter("x.wav", ["R"], device="cuda:0")

    def test_mutating_returned_receipt_does_not_mutate_original(self):
        delegate = Delegate();adapter = module._MeasurementErrorAdapter(delegate)
        receipt = adapter.receipt;receipt["runtime"]["ready"] = False
        self.assertTrue(adapter.receipt["runtime"]["ready"])

    def test_frozen_source_drift_remains_fatal(self):
        adapter = module._MeasurementErrorAdapter(Delegate(RuntimeError(TRACE)))
        with patch.object(module, "_check_frozen_sources", side_effect=RuntimeError("source drift")):
            with self.assertRaisesRegex(RuntimeError, "source drift"):adapter("x.wav", ["R"], device="cuda:0")

    def test_factory_requires_explicit_known_policy_before_importing_runtime(self):
        with self.assertRaisesRegex(ValueError, "explicit supported"):
            module.create_measurement_safe_analyzer(None, policy="ignore all failures")


if __name__ == "__main__":unittest.main()
