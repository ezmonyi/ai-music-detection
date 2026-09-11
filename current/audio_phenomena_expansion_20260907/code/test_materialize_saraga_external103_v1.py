"""Small explicit synthetic fixtures; tempfile honors TMPDIR (including NFS)."""
import copy
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import materialize_saraga_external103_v1 as m


class FakeBackend:
    """Control-flow backend; not numerical evidence. Separate real WAV test below."""
    CONFIG = {"synthetic_test_only": True}
    np = types.SimpleNamespace(__version__="synthetic")
    scipy = types.SimpleNamespace(__version__="synthetic")
    def __init__(self):
        self.calls = []
        self.fail = None
        self.sf = types.SimpleNamespace(__version__="synthetic", __libsndfile_version__="synthetic", write=self.write)
    def write(self, path, data, rate, **kwargs):
        Path(path).write_bytes(data)
    def _sample_sha(self, data, dtype):
        return m.util.sha(data)
    def _raw_hashes(self, path):
        raw = Path(path).read_bytes()
        return {"sha256": m.util.sha(raw), "md5": hashlib.md5(raw).hexdigest(), "bytes": len(raw)}, {}
    def verify_center_interval(self, path, envelope):
        self.calls.append(str(path))
        if self.fail and self.fail in str(path):
            raise ValueError("Synthetic requested failure")
        r = envelope["record"]
        n = r["measurement"]
        count = 60 * n["sample_rate"]
        start = (n["actual_frames"] - count) // 2
        proof = {"status": "passed_interval_proof_not_classifier_admission", "physical_record_sha256": m.util.seal(r),
                 "physical_contract_sha256": r["contract_sha256"], "source_audio_path": str(path), "mbid": r["item"]["mbid"],
                 "crop_start_frame": start, "crop_frames": count, "crop_end_frame_exclusive": start + count,
                 "seek_proof": {"exact_bytes_equal": True, "exact_array_equal": True}, "real_empty_read_observed": True,
                 "configuration": self.CONFIG, "standardized_float32_sha256": m.util.sha(b"synthetic_pcm"),
                 "raw_hashes_after": n["raw_hashes_after_decode"], "observed_actual_frames": n["actual_frames"],
                 "physical_header_frames": n["header_frames"], "native_sample_rate_hz": n["sample_rate"],
                 "native_float64_sha256": m.util.sha(b"synthetic_native64"), "native_float32_sha256": m.util.sha(b"synthetic_native32")}
        return None, b"synthetic_pcm", proof


def fake_readback(path, helper, sample_sha):
    raw = Path(path).read_bytes()
    m.util.require(m.util.sha(raw) == sample_sha, "Synthetic readback identity mismatch")
    return {"file_sha256": m.util.sha(raw), "bytes": len(raw), "float32_sha256": sample_sha, "readback_verified": True}


def fixture(root):
    raw_root = root / "synthetic_raw"
    raw_root.mkdir()
    rows, inventory = [], {}
    for index, rate in enumerate((44100, 48000)):
        mbid = f"synthetic_record_{index}"
        path = raw_root / (mbid + ".mp3")
        path.write_bytes(f"SYNTHETIC TEST ONLY {index}".encode())
        raw = path.read_bytes()
        hashes = {"bytes": len(raw), "sha256": m.util.sha(raw), "md5": hashlib.md5(raw).hexdigest()}
        frames = rate * 61 + 5
        count, start = 60 * rate, (frames - 60 * rate) // 2
        record = {"contract_sha256": "a" * 64, "item": {"mbid": mbid}, "measurement": {
            "sample_rate": rate, "actual_frames": frames, "header_frames": frames + 7, "channels": 2,
            "raw_hashes_before_decode": hashes, "raw_hashes_after_decode": hashes}}
        relative = f"{m.util.PHYSICAL}/receipts/{mbid}.json"
        receipt = root / relative
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_bytes(m.util.canonical({"record": record, "record_sha256": m.util.seal(record)}))
        inventory[relative] = {"sha256": m.file_sha(receipt)}
        rows.append({"item_id": "saraga_hindustani_" + mbid, "mbid": mbid, "label": "0", "source_id": m.util.SOURCE,
                     "role": m.util.ROLE, "group_id": "synthetic_group_0", "source_audio_path": str(path),
                     "acquisition_raw_sha256": hashes["sha256"], "native_sample_rate_hz": str(rate), "native_channels": "2",
                     "actual_eof_frames": str(frames), "soundfile_header_frames": str(frames + 7),
                     "crop_start_frame": str(start), "crop_frames": str(count), "crop_end_frame_exclusive": str(start + count),
                     "source_offset_seconds": repr(start / rate), "classifier_admission": "False", "evaluation_allowed": "False"})
    draft = {"status": "draft", "synthetic_test_only": True}
    directory = root / "synthetic_selection"
    directory.mkdir()
    files = {"metadata.csv": m.csv_bytes(rows), "input_inventory.json": m.util.canonical(inventory), "selection_draft.json": m.util.canonical(draft)}
    for name, raw in files.items():
        (directory / name).write_bytes(raw)
    selection = {"status": "frozen", "authorized_stage": "saraga_external103_cohort_selection_only", "synthetic_test_only": True,
                 "cohort_identities_and_intervals_frozen": True, "real_audio_measurement_accepted": False,
                 "selection_draft_directory": "synthetic_selection", "selection_draft": draft,
                 "selection_files_sha256": {name: m.util.sha(raw) for name, raw in files.items()}}
    freeze = root / m.SELECTION
    freeze.parent.mkdir()
    freeze.write_bytes(m.util.canonical(selection))
    return {"selection_sha256": m.file_sha(freeze), "raw_root": str(raw_root)}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="saraga-batch-synthetic-")
        self.root = Path(self.tmp.name).resolve()
        self.synthetic = fixture(self.root)
        self.output = self.root / "synthetic_output"
        self.backend = FakeBackend()
        self.kwargs = {"_synthetic": self.synthetic, "_backend": self.backend}
    def tearDown(self):
        self.tmp.cleanup()
    def freeze_measurement(self, change=None):
        draft = m.create_draft(self.root, self.output, self.root / "draft", **self.kwargs)
        evidence = self.root / "synthetic_review.json"
        evidence.write_bytes(b'{"synthetic_test_only":true,"approved":true}\n')
        wrapper = {"status": "frozen", "authorized_stage": m.STAGE, "contract": draft["contract"],
                   "contract_sha256": draft["contract_sha256"], "independent_review": {"approved": True, "reviewer": "root",
                   "evidence": evidence.name, "evidence_sha256": m.file_sha(evidence)}}
        if change:
            change(wrapper)
        frozen = self.root / "measurement_frozen.json"
        frozen.write_bytes(m.util.canonical(wrapper))
        return frozen, m.file_sha(frozen)
    def launch(self, frozen, digest, resume=False):
        with mock.patch.object(m, "verify_written", side_effect=fake_readback):
            return m.run(self.root, frozen, digest, resume=resume, **self.kwargs)
    def test_draft_no_audio_or_output(self):
        with mock.patch.object(self.backend, "verify_center_interval", side_effect=AssertionError("No audio")):
            draft = m.create_draft(self.root, self.output, self.root / "draft", **self.kwargs)
        self.assertEqual(draft["status"], "draft")
        self.assertFalse(self.output.exists())
        self.assertFalse(draft["audio_opened"])
    def test_all_success_and_resume_hash_identity(self):
        frozen, digest = self.freeze_measurement()
        result = self.launch(frozen, digest)
        self.assertEqual((result["passed"], result["failed"]), (2, 0))
        m.verify_publication(self.output / "final")
        native = m.util.csv_rows((self.output / "final/native_metadata_60s.csv").read_bytes())
        self.assertEqual([r["crop_frames"] for r in native], ["2646000", "2880000"])
        manifest = m.util.csv_rows((self.output / "final/inference_manifest.csv").read_bytes())
        self.assertEqual([r["native_sample_rate_hz"] for r in manifest], ["44100", "48000"])
        self.assertTrue(all(r["standardized_sr"] == "44100" for r in manifest))
        self.assertTrue(all(r["role"] == m.util.ROLE and r["label"] == "0" for r in native))
        self.assertEqual(self.launch(frozen, digest, resume=True), result)
        self.assertEqual(len(self.backend.calls), 2)
    def test_selection_freeze_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "measurement freeze"):
            self.launch(self.root / m.SELECTION, self.synthetic["selection_sha256"])
    def test_draft_cannot_launch(self):
        m.create_draft(self.root, self.output, self.root / "draft", **self.kwargs)
        path = self.root / "draft/measurement_contract_draft.json"
        with self.assertRaisesRegex(ValueError, "measurement freeze"):
            self.launch(path, m.file_sha(path))
    def test_independent_root_review_required(self):
        frozen, digest = self.freeze_measurement(lambda x: x["independent_review"].update(reviewer="self"))
        with self.assertRaisesRegex(ValueError, "root review"):
            self.launch(frozen, digest)
    def test_stale_freeze_pin(self):
        frozen, digest = self.freeze_measurement()
        with self.assertRaisesRegex(ValueError, "freeze file pin"):
            self.launch(frozen, "0" * 64)
    def test_runtime_drift_before_audio(self):
        frozen, digest = self.freeze_measurement()
        with mock.patch.object(m, "runtime", return_value={"changed": True}), self.assertRaisesRegex(ValueError, "runtime mismatch"):
            self.launch(frozen, digest)
        self.assertEqual(self.backend.calls, [])
    def test_failure_continues_without_partial_final(self):
        frozen, digest = self.freeze_measurement()
        self.backend.fail = "record_0"
        result = self.launch(frozen, digest)
        self.assertEqual((result["passed"], result["failed"]), (1, 1))
        self.assertFalse((self.output / "final").exists())
        self.assertEqual(len(self.backend.calls), 2)
        self.backend.fail = None
        self.assertEqual(self.launch(frozen, digest, resume=True)["failed"], 1)
        self.assertEqual(len(self.backend.calls), 2)
    def test_resume_corrupted_wav_rejected(self):
        frozen, digest = self.freeze_measurement()
        self.launch(frozen, digest)
        wav = next((self.output / "items").glob("*/audio.wav"))
        wav.write_bytes(b"tamper")
        self.assertEqual(self.launch(frozen, digest, resume=True)["failed"], 1)
    def test_final_sweep_catches_source_changed_after_earlier_item(self):
        frozen, digest = self.freeze_measurement()
        original = self.backend.verify_center_interval
        def changing(path, envelope):
            result = original(path, envelope)
            if "record_1" in str(path):
                (self.root / "synthetic_raw/synthetic_record_0.mp3").write_bytes(b"changed after earlier proof")
            return result
        with mock.patch.object(self.backend, "verify_center_interval", side_effect=changing):
            result = self.launch(frozen, digest)
        self.assertEqual((result["passed"], result["failed"]), (1, 1))
        self.assertFalse((self.output / "final").exists())
        proof = m.util.strict_json(next((self.output / "failures").glob("*/failure.json")).read_bytes())
        self.assertEqual(proof["failure_stage"], "final_hash_recheck")
    def test_orphan_reservation_fails_closed(self):
        destination = self.root / "orphan"
        destination.mkdir()
        (destination / "audio.wav").write_bytes(b"synthetic")
        with self.assertRaises(FileNotFoundError):
            m.verify_publication(destination)
        with self.assertRaisesRegex(ValueError, "conflict"):
            m.publish_directory(destination, {"audio.wav": b"different"})
    def test_hardlink_commit_no_replace_and_corruption(self):
        destination = self.root / "published"
        m.publish_directory(destination, {"proof.json": b"{}"})
        original = (destination / "COMMIT.json").read_bytes()
        with self.assertRaises(FileExistsError):
            m.commit_directory(destination, {"proof.json"})
        self.assertEqual((destination / "COMMIT.json").read_bytes(), original)
        (destination / "proof.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "payload hash"):
            m.verify_publication(destination)
    def test_unsupported_hardlink_preserves_uncommitted_reservation(self):
        destination = self.root / "uncommitted"
        with mock.patch.object(m.os, "link", side_effect=OSError("unsupported")), self.assertRaises(OSError):
            m.publish_directory(destination, {"payload": b"synthetic"})
        self.assertTrue(destination.exists())
        self.assertFalse((destination / "COMMIT.json").exists())
    def test_synthetic_cannot_bypass_production_pin(self):
        with mock.patch.object(m, "get_backend", return_value=self.backend), self.assertRaisesRegex(ValueError, "Input hash mismatch"):
            m.build_contract(self.root, self.output)
    def test_symlink_and_traversal(self):
        link = self.root / "alias"
        link.symlink_to(self.root / "synthetic_selection")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            m.path_ok(link)
        with self.assertRaisesRegex(ValueError, "traversal"):
            m.path_ok(self.root / "x/../alias")
    def test_source_or_crop_proof_mismatch(self):
        frozen, digest = self.freeze_measurement()
        original = self.backend.verify_center_interval
        def wrong(*args):
            native, standard, proof = original(*args)
            proof["crop_start_frame"] += 1
            return native, standard, proof
        with mock.patch.object(self.backend, "verify_center_interval", side_effect=wrong):
            self.assertEqual(self.launch(frozen, digest)["failed"], 2)


try:
    import saraga_interval_audio_v1 as real_helper
except ImportError:
    real_helper = None


@unittest.skipIf(real_helper is None, "Existing NumPy/SciPy/SoundFile runtime unavailable")
class RealSyntheticWavTests(unittest.TestCase):
    def test_complete_real_synthetic_batch_both_rates(self):
        import test_saraga_interval_audio_v1 as fixtures
        helper = real_helper
        with tempfile.TemporaryDirectory(prefix="saraga-real-batch-synthetic-") as tmp:
            root = Path(tmp).resolve()
            synthetic = fixture(root)
            selection_dir = root / "synthetic_selection"
            rows = m.util.csv_rows((selection_dir / "metadata.csv").read_bytes())
            inventory = {}
            for row in rows:
                rate, frames = int(row["native_sample_rate_hz"]), int(row["actual_eof_frames"])
                path = Path(row["source_audio_path"])
                helper.sf.write(path, fixtures._wave(0, frames), rate, format="WAV", subtype="DOUBLE")
                record = fixtures._physical(path.read_bytes(), rate, frames, fmt="WAV", subtype="DOUBLE")
                record["item"]["mbid"] = row["mbid"]
                row["soundfile_header_frames"] = str(frames)
                row["acquisition_raw_sha256"] = record["measurement"]["raw_hashes_before_decode"]["sha256"]
                relative = f"{m.util.PHYSICAL}/receipts/{row['mbid']}.json"
                (root / relative).write_bytes(m.util.canonical({"record": record, "record_sha256": m.util.seal(record)}))
                inventory[relative] = {"sha256": m.file_sha(root / relative)}
            (selection_dir / "metadata.csv").write_bytes(m.csv_bytes(rows))
            (selection_dir / "input_inventory.json").write_bytes(m.util.canonical(inventory))
            selection = m.util.strict_json((root / m.SELECTION).read_bytes())
            selection["selection_files_sha256"] = {name: m.file_sha(selection_dir / name) for name in selection["selection_files_sha256"]}
            (root / m.SELECTION).write_bytes(m.util.canonical(selection))
            synthetic["selection_sha256"] = m.file_sha(root / m.SELECTION)
            output = root / "real_synthetic_output"
            args = {"_synthetic": synthetic, "_backend": helper}
            draft = m.create_draft(root, output, root / "draft", **args)
            evidence = root / "synthetic_review.json"
            evidence.write_bytes(b'{"synthetic_test_only":true}')
            frozen = {"status": "frozen", "authorized_stage": m.STAGE, "contract": draft["contract"],
                      "contract_sha256": draft["contract_sha256"], "independent_review": {"approved": True, "reviewer": "root",
                      "evidence": evidence.name, "evidence_sha256": m.file_sha(evidence)}}
            frozen_path = root / "synthetic_frozen.json"
            frozen_path.write_bytes(m.util.canonical(frozen))
            result = m.run(root, frozen_path, m.file_sha(frozen_path), **args)
            self.assertEqual((result["passed"], result["failed"]), (2, 0))
            for row in rows:
                saved = m.util.strict_json((output / "items" / row["item_id"] / "proof.json").read_bytes())
                self.assertGreater(saved["interval_proof"]["standardized_peak_absolute"], 1)
                self.assertEqual(saved["interval_proof"]["standardized_frames"], 2646000)
            self.assertEqual(m.run(root, frozen_path, m.file_sha(frozen_path), resume=True, **args), result)

    def test_float_wav_exact_readback_preserves_above_one(self):
        helper = real_helper
        with tempfile.TemporaryDirectory(prefix="saraga-float-synthetic-") as tmp:
            path = Path(tmp).resolve() / "synthetic.wav"
            data = helper.np.full((2646000, 2), 1.25, dtype="<f4")
            helper.sf.write(path, data, 44100, format="WAV", subtype="FLOAT")
            result = m.verify_written(path, helper, helper._sample_sha(data, "<f4"))
            self.assertTrue(result["readback_verified"])
            with self.assertRaisesRegex(ValueError, "byte identity"):
                m.verify_written(path, helper, "0" * 64)


if __name__ == "__main__":
    unittest.main()
