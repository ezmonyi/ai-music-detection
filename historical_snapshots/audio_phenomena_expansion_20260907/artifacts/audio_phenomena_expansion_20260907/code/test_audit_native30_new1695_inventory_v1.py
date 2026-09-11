"""Narrow mutation tests; not a synthetic full-1695 audit fixture."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import audit_native30_new1695_inventory_v1 as a


H = "a" * 64
P = "b" * 64


def documents():
    excluded = [{"id": f"m{i}"} for i in range(a.MONO)]
    contract = {"status": "frozen_before_any_audio_processing", "scope": "new1695_native_stereo_DSP_only",
                "expected_count": a.COUNT, "classifier_fits": 0, "cohort_admitted": False,
                "feature_extraction_authorized": False, "excluded_native_mono": excluded}
    manifest = {"version": "materialize_native30_new1695_v1", "status": "all1695_DSP_materialized_not_cohort_admitted",
                "count": a.COUNT, "contract_sha256": H, "classifier_fits": 0, "cohort_admitted": False,
                "feature_extraction_authorized": False, "excluded_native_mono": excluded}
    commit = {"status": "committed_new1695_DSP_only", "completed": a.COUNT, "contract_sha256": H,
              "products": {"contract.json": {"bytes": 1, "sha256": P}}}
    return commit, contract, manifest


def record_fixtures():
    runtime = {key: "v" for key in ("python", "numpy", "scipy", "soundfile", "libsndfile")}
    row = {"id": "x", "source_group": "S", "group_id": "g", "component_id": "c", "label": "1", "role": "development",
           "execution_native_path": "/native.flac", "native_evidence": {"sample_rate_hz": 48000, "channels": 2, "sha256": H}}
    actual = 1440000
    audit = {"status": "verified_DSP_only_not_cohort_admission", "classifier_admitted": False, "configuration": {"dsp": 1},
             "source_path": "/native.flac", "native_rate_hz": 48000, "native_channels": 2,
             "sequential_decode": {"actual_frames": actual, "empty_eof_observed": True, "read_calls_including_empty_eof": 23,
                                   "float64_pcm_sha256": H, "header": {"sample_rate_hz": 48000, "channels": 2,
                                                                                "frames": actual, "format": "FLAC", "subtype": "PCM_16"}},
             "header_minus_actual_frames": 0, "region_kind": "full_native_sequential_decode",
             "approved_region_float64_sha256": None,
             "coordinates": {"region_start_frame": 0, "region_frames": actual, "crop_start_frame": 0,
                             "crop_frames": actual, "crop_end_frame_exclusive": actual},
             "sequential_passes_identical": True, "native_crop_float64_sha256": P,
             "output_rate_hz": a.RATE, "output_channels": 2, "output_frames": a.FRAMES,
             "sample_count": a.FRAMES * 2, "output_waveform_float32_sha256": P, "runtime": runtime}
    receipt = {"status": "materialized_DSP_only", "contract_sha256": H, "row": row, "row_sha256": a.vh(row),
               **{key: row[key] for key in ("id", "source_group", "group_id", "component_id", "label", "role")},
               "waveform_bit_exact_roundtrip": True}
    contract = {"configuration": {"dsp": 1}, "runtime": dict(runtime)}
    return receipt, row, audit, contract


class InventoryPreconditions(unittest.TestCase):
    def test_canonical_and_strict_json(self):
        self.assertEqual(a.vh({"a": 1, "b": 2}), a.vh({"b": 2, "a": 1}))
        with self.assertRaises(ValueError): a.canonical({"a": float("nan")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"): a.load(path)
            path.write_text('{"a":NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite JSON"): a.load(path)

    def test_batch_document_status_and_scope(self):
        commit, contract, manifest = documents()
        a.validate_batch_documents(commit, contract, manifest, H)
        mutations = ((commit, "status", "partial"), (commit, "completed", a.COUNT - 1),
                     (contract, "scope", "cohort"), (contract, "cohort_admitted", True),
                     (manifest, "status", "partial"), (manifest, "feature_extraction_authorized", True))
        for original, key, bad in mutations:
            c, k, m = copy.deepcopy((commit, contract, manifest))
            target = c if original is commit else k if original is contract else m
            target[key] = bad
            with self.assertRaises(ValueError): a.validate_batch_documents(c, k, m, H)

    def test_full_inventory_rejects_extra_directory_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("COMMIT.json", "writer.lock", "contract.json", "manifest.json"):
                (root / name).write_text("x")
            for name in ("items", "audio", "failures", "runs"):
                (root / name).mkdir()
            products = {"contract.json": {"bytes": 1, "sha256": H}, "manifest.json": {"bytes": 1, "sha256": H}}
            a.validate_full_inventory(root, products)
            (root / "extra").mkdir()
            with self.assertRaisesRegex(ValueError, "root inventory"): a.validate_full_inventory(root, products)
            (root / "extra").rmdir()
            (root / "audio" / "bad").symlink_to(root / "contract.json")
            with self.assertRaisesRegex(ValueError, "nested/nonregular"): a.validate_full_inventory(root, products)

    def test_independent_pilot_receipt_is_required(self):
        products = {f"audio/id{i}.wav": {"bytes": 1, "sha256": H} for i in range(8)}
        commit = {"status": "committed_DSP_pilot_only", "products": products}
        audit = {"status": "passed_independent_pilot_DSP_replay", "producer_imported": False,
                 "cohort_admitted": False, "classifier_fits": 0, "bindings": {"commit": a.PILOT_COMMIT},
                 "records": [{"id": f"id{i}", "float32_bit_exact": True, "output_frames": a.FRAMES} for i in range(8)]}
        replay = a.validate_pilot_replay(commit, audit, {f"id{i}": None for i in range(8)})
        self.assertEqual(len(replay), 8)
        for key, bad in (("status", "unknown"), ("cohort_admitted", True)):
            changed = copy.deepcopy(audit); changed[key] = bad
            with self.assertRaises(ValueError): a.validate_pilot_replay(commit, changed, {f"id{i}": None for i in range(8)})
        changed = copy.deepcopy(audit); changed["records"][0]["float32_bit_exact"] = False
        with self.assertRaises(ValueError): a.validate_pilot_replay(commit, changed, {f"id{i}": None for i in range(8)})

    def test_receipt_metadata_accepts_complete_join(self):
        receipt, row, audit, contract = record_fixtures()
        rate, actual, coordinates = a.validate_receipt_metadata(receipt, row, audit, contract, H)
        self.assertEqual((rate, actual, coordinates["crop_start_frame"]), (48000, 1440000, 0))

    def test_receipt_metadata_rejects_lineage_and_output_mutations(self):
        paths = (("receipt", ("source_group",), "other"), ("audit", ("source_path",), "/other.flac"),
                 ("audit", ("native_rate_hz",), 44100), ("audit", ("sequential_decode", "header", "channels"), 1),
                 ("audit", ("coordinates", "crop_start_frame"), 1), ("audit", ("output_frames",), a.FRAMES - 1),
                 ("audit", ("runtime", "numpy"), "other"))
        for target_name, keys, bad in paths:
            receipt, row, audit, contract = copy.deepcopy(record_fixtures())
            target = receipt if target_name == "receipt" else audit
            for key in keys[:-1]: target = target[key]
            target[keys[-1]] = bad
            with self.assertRaises(ValueError): a.validate_receipt_metadata(receipt, row, audit, contract, H)

    def test_binding_detects_end_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x"; path.write_bytes(b"a")
            binding = a.bind_file(path); path.write_bytes(b"b")
            with self.assertRaises(ValueError): a.recheck({"x": binding})

    def test_exclusive_publication_and_directory_sync_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            a.write_exclusive(path, {"status": "draft"})
            self.assertEqual(json.loads(path.read_text()), {"status": "draft"})
            with self.assertRaises(FileExistsError): a.write_exclusive(path, {"status": "changed"})


if __name__ == "__main__":
    unittest.main()
