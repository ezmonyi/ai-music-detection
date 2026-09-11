"""Synthetic-only tests for audit_native30_bc_v1.

These tests never open a frozen cohort input, never call the producer, and do
not create a result COMMIT.  The expensive 3,830-row audit is intentionally
left to the caller after the live producer publishes a complete COMMIT.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import audit_native30_bc_v1 as a


class NumericalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small enough to make direct primitive tests fast; the view test uses
        # one deterministic exact Native30-shaped stereo array below.
        rng = np.random.default_rng(17)
        cls.spectra = (rng.normal(size=(a.FRAMES, 513)) +
                       1j * rng.normal(size=(a.FRAMES, 513))).astype(np.complex128)

    def test_grid_is_exact_full228_and_target_order(self):
        self.assertEqual(a.GRID.shape, (228, 3))
        self.assertEqual(a.GRID[a.TARGET_INDEX].tolist(), list(a.TARGET))
        self.assertEqual(a.GRID[-1].tolist(), [128, 128, 256])
        self.assertTrue(np.array_equal(a.GRID, np.asarray([
            (x, y, x + y) for x in range(8, 193, 8)
            for y in range(x, 193, 8) if x + y <= 256], dtype=np.int64)))

    def test_independent_primitive_uses_f_contiguous_normalized_columns_and_parity(self):
        actual = a.independent_primitive(self.spectra, (32, 48))
        columns = self.spectra[:, [32, 48, 80]]
        scales = np.maximum(np.max(np.abs(columns.real), axis=0),
                            np.max(np.abs(columns.imag), axis=0))
        normalized = np.empty(columns.shape, dtype=np.complex128, order="C")
        normalized.real = columns.real / scales
        normalized.imag = columns.imag / scales
        u, v = normalized[:, 0] * normalized[:, 1], normalized[:, 2]
        z = complex(np.sum(u * np.conj(v)))
        e1, e2 = float(np.sum(np.abs(u) ** 2)), float(np.sum(np.abs(v) ** 2))
        expected_score = (abs(z) / np.sqrt(e1) / np.sqrt(e2)) ** 2
        self.assertTrue(np.isclose(actual["normalized_sums"]["product_energy_sum"], e1,
                                   atol=a.ATOL, rtol=a.RTOL))
        self.assertTrue(np.isclose(actual["normalized_sums"]["sum_frequency_energy_sum"], e2,
                                   atol=a.ATOL, rtol=a.RTOL))
        self.assertTrue(np.isclose(actual["squared_bicoherence"], expected_score,
                                   atol=a.ATOL, rtol=a.RTOL))

    def test_independent_resample_is_bit_exact_to_pinned_scipy_reference(self):
        native = np.zeros(1323000, dtype=np.float64)
        native[0] = 0.25
        native[44100 * 11 - 1] = -0.125
        native[44100 * 19] = 0.0625
        actual = a.independent_resample(native)
        expected = resample_poly(native, 160, 441, window=("kaiser", 5.0),
                                 padtype="constant", cval=0.0)
        np.testing.assert_array_equal(actual, expected)

    def test_replay_pool_retains_all_grid_cells_and_zero_nulls(self):
        wave = np.zeros(128000, dtype=np.float64)
        metadata, arrays = a.replay_pool(wave, "unsupported_zero_background_rms")
        self.assertEqual(len(metadata["pools"]), 2)
        self.assertEqual(len(metadata["pools"][0]["cells"]), 228)
        self.assertEqual(arrays["spectra"].shape, (2, 247, 513))
        self.assertEqual(arrays["eligible_mask"].shape, (2, 228))
        self.assertEqual(a.reduce_scalar(metadata)["status"], "no_eligible_target_pools")
        self.assertEqual(a.reduce_scalar(metadata)["eligibility_mask"], [False, False])
        self.assertTrue(np.isnan(arrays["bin_energy_fraction"]).all())

    def test_wrong_order_view_is_rejected(self):
        stereo = np.zeros((1323000, 2), dtype=np.float64)
        stereo[44100 * 11 - 1, 0] = 1.0
        _, view, _ = a.reconstruct_view(stereo)
        wrong = copy.deepcopy(view)
        wrong["resampling"]["crop_start"] += 1
        with self.assertRaisesRegex(ValueError, "analysis view"):
            a.validate_view(wrong, view)

    def test_null_and_failure_are_distinct(self):
        wave = np.zeros(128000, dtype=np.float64)
        metadata, _ = a.replay_pool(wave, "unsupported_zero_background_rms")
        scalar = a.reduce_scalar(metadata)
        self.assertIsNone(scalar["median_squared_bicoherence"])
        failed = copy.deepcopy(metadata)
        failed["pools"][0]["status"] = "processing_failure"
        with self.assertRaises(ValueError):
            a.reduce_scalar(failed)

    def test_adversarial_array_mutation_is_rejected(self):
        wave = np.zeros(128000, dtype=np.float64)
        metadata, arrays = a.replay_pool(wave, "unsupported_zero_background_rms")
        mutated = {k: np.array(v, copy=True) for k, v in arrays.items()}
        mutated["eligible_mask"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "array eligible_mask"):
            a._metadata_and_array_parity(metadata, mutated, metadata, arrays)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_source_decode_requires_exact_float_native30_and_hash(self):
        values = np.zeros((1323000, 2), dtype=np.float32)
        values[13, 0] = np.float32(0.25)
        path = self.root / "synthetic.wav"
        sf.write(path, values, 44100, subtype="FLOAT")
        row = {"input": a.file_binding(path),
               "waveform_float32_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()}
        stereo, evidence = a.read_native_source(row)
        self.assertEqual(stereo.dtype, np.dtype("float64"))
        self.assertEqual(stereo.shape, (1323000, 2))
        self.assertEqual(evidence["waveform_float32_sha256"], row["waveform_float32_sha256"])
        path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))
        with self.assertRaises(ValueError):
            a.read_native_source(row)

    def test_symlink_and_output_overlap_fail_closed(self):
        target = self.root / "target.json"
        target.write_text("{}\n")
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            a.file_binding(link)
        result = self.root / "result"
        result.mkdir()
        (result / "writer.lock").touch()
        with self.assertRaises(ValueError):
            a._expected_inventory(result, [])

    def test_terminal_inventory_rejects_retained_failure_or_orphan(self):
        result = self.root / "result"
        for folder in ("items", "metadata", "arrays"):
            (result / folder).mkdir(parents=True)
        (result / "writer.lock").touch()
        (result / "contract.json").write_text("{}\n")
        (result / "manifest.json").write_text("{}\n")
        (result / "COMMIT.json").write_text("{}\n")
        (result / "failures").mkdir()
        (result / "failures" / "bad.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "complete result inventory"):
            a._expected_inventory(result, [])

    def test_audit_never_publishes_when_result_has_no_complete_commit(self):
        # Freeze/COMMIT validation is deliberately first.  This synthetic
        # missing result confirms the public entry point cannot create output
        # while the producer is still running.
        result = self.root / "result"
        result.mkdir()
        output = self.root / "audit.json"
        with self.assertRaises(ValueError):
            a.audit(result, self.root / "missing-freeze.json", "0" * 64,
                    result / "COMMIT.json", "0" * 64, output)
        self.assertFalse(output.exists())

    def test_runtime_manifest_uses_real_binding_dict_shape(self):
        # Exercise the frozen schema shape directly: module/package entries
        # and libsndfile_binary are {path, bytes, sha256} dictionaries.
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            tmp = Path(tmp)
            module_file = tmp / "module.so"
            package_file = tmp / "package.py"
            sndfile_file = tmp / "libsndfile.so"
            for path, payload in ((module_file, b"module"),
                                  (package_file, b"package"),
                                  (sndfile_file, b"sndfile")):
                path.write_bytes(payload)
            runtime = {
                "python": a.platform.python_version(),
                "numpy": a.np.__version__, "scipy": a.scipy.__version__,
                "soundfile": a.sf.__version__, "executable":
                a.file_binding(Path(a.sys.executable).resolve()),
                "import_origins": {}, "module_files":
                {"synthetic": a.file_binding(module_file)}, "package_files":
                {str(package_file): a.file_binding(package_file)},
                "libsndfile_binary": a.file_binding(sndfile_file),
                "libsndfile_resolution": {}, "thread_environment": {},
            }
            a._validate_runtime(runtime)

            malformed = copy.deepcopy(runtime)
            malformed["module_files"] = {"synthetic": str(module_file)}
            with self.assertRaisesRegex(ValueError, "runtime module binding: synthetic shape"):
                a._validate_runtime(malformed)

    def test_execution_contract_is_parent_authorized_envelope(self):
        prepared = {"version": "prepared-sentinel"}
        freeze_entry = {"path": "/tmp/freeze.json", "bytes": 1, "sha256": "a" * 64}
        prepared_entry = {"path": "/tmp/prepared.json", "bytes": 2, "sha256": "b" * 64}
        contract = a._execution_contract(freeze_entry, prepared_entry, prepared)
        self.assertEqual(contract["prepared"], prepared)
        self.assertEqual(contract["parent_freeze"], freeze_entry)
        self.assertEqual(contract["prepared_contract"], prepared_entry)
        self.assertEqual(contract["status"], "parent_authorized_BC_measurement_only")
        self.assertEqual(contract["classifier_fits"], 0)
        self.assertFalse(contract["classifier_admission"])
        self.assertNotEqual(a.value_hash(contract), a.value_hash(prepared))

    def test_auditor_has_no_producer_imports_or_resample_poly_call(self):
        source = Path(a.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import run_native30_bc_v2", source)
        self.assertNotIn("resample_poly(", source)
        self.assertNotIn("bicoherence_audio_v1 import", source)
        self.assertNotIn("bicoherence_primitive_v2 import", source)


if __name__ == "__main__":
    unittest.main()
