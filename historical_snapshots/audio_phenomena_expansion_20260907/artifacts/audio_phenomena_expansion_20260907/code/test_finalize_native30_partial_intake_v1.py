"""Synthetic mutation and publication tests; no corpus/native DSP execution."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import finalize_native30_partial_intake_v1 as f
from test_audit_native30_new1695_inventory_v1 import record_fixtures


def write(path, value):
    path.write_bytes(f.a.canonical(value))
    return f.a.sha(path)


def fixture(base):
    """Nine synthetic stereo attempts, eight synthetic WAVs, one short record."""
    root, pilot = base / "original", base / "pilot"
    root.mkdir(); pilot.mkdir(); (pilot / "audio").mkdir()
    (root / "writer.lock").write_bytes(b"")
    for name in ("items", "audio", "failures", "runs"):
        (root / name).mkdir()
    rows, receipts = [], []
    for index in range(9):
        receipt, row, audit, contract = record_fixtures()
        ident = "id" + str(index)
        source = base / (ident + ".source")
        source.write_bytes(b"synthetic source " + str(index).encode())
        row.update(id=ident, execution_native_path=str(source))
        row["native_evidence"]["sha256"] = f.a.sha(source)
        audit.update(source_path=str(source), source_sha256_before=f.a.sha(source), source_sha256_after=f.a.sha(source),
                     source_stat_signature=f.a.signature(source)[:5], output_frames=f.a.FRAMES, sample_count=f.a.FRAMES * 2)
        receipt.update(row=row, row_sha256=f.a.vh(row), **{k: row[k] for k in
                       ("id", "source_group", "group_id", "component_id", "label", "role")})
        receipt["audit"] = audit
        rows.append(row); receipts.append(receipt)
    mono = [{"id": "mono", "source_group": "S", "label": "1", "reason": "native_mono", "native_evidence": {"channels": 1}}]
    contract.update(status="frozen_before_any_audio_processing", scope="new1695_native_stereo_DSP_only", expected_count=9,
                    output_root=str(root), rows=rows, excluded_native_mono=mono, **f.SCOPE)
    contract_sha = write(root / "contract.json", contract)
    pilot_products = {}
    for index, receipt in enumerate(receipts[:8]):
        ident = rows[index]["id"]
        wav = root / "audio" / (ident + ".wav")
        values = np.arange(f.a.FRAMES * 2, dtype=np.float32).reshape(-1, 2) / 100
        sf.write(wav, values, f.a.RATE, format="WAV", subtype="FLOAT")
        pcm = hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
        receipt["audit"].update(output_waveform_float32_sha256=pcm, peak_absolute=float(np.max(np.abs(values))),
                                samples_abs_above_one=int(np.count_nonzero(np.abs(values) > 1)))
        receipt.update(contract_sha256=contract_sha, standardized_path=str(wav), file_sha256=f.a.sha(wav),
                       file_bytes=wav.stat().st_size, waveform_float32_sha256=pcm)
        write(root / "items" / (ident + ".json"), {"payload": receipt, "receipt_sha256": f.a.vh(receipt)})
        (pilot / "audio" / wav.name).write_bytes(wav.read_bytes())
        pilot_products["audio/" + wav.name] = {"sha256": f.a.sha(wav), "bytes": wav.stat().st_size}
    pc_sha = write(pilot / "COMMIT.json", {"status": "committed_DSP_pilot_only", "products": pilot_products})
    pa = base / "pilot_audit.json"
    pa_sha = write(pa, {"status": "passed_independent_pilot_DSP_replay", "producer_imported": False,
                       "cohort_admitted": False, "classifier_fits": 0, "bindings": {"commit": pc_sha},
                       "records": [{"id": r["id"], "float32_bit_exact": True, "output_frames": f.a.FRAMES} for r in rows[:8]]})
    row = rows[-1]
    failure = {"id": row["id"], "contract_sha256": contract_sha, "row_sha256": f.a.vh(row), "run_id": f.RUN,
               "exception": "ValueError", "message": "native region shorter than30s: padding forbidden"}
    failure_path = root / "failures" / (row["id"] + "." + f.RUN + ".json")
    failure_sha = write(failure_path, failure)
    run = {"status": "partial_no_COMMIT", "run_id": f.RUN, "contract_sha256": contract_sha, "expected": 9,
           "completed": 8, "failed": 1, "failures": [failure], "classifier_fits": 0, "cohort_admitted": False}
    run_path = root / "runs" / (f.RUN + ".json")
    run_sha = write(run_path, run)
    source = Path(row["execution_native_path"])
    excluded = {k: row[k] for k in ("id", "source_group", "label", "group_id", "component_id", "role")}
    excluded.update(native_path=str(source), native_sha256=f.a.sha(source), native_rate_hz=48000, native_channels=2,
                    actual_frames=1439471, primary_second_pass_frames=1439471, primary_pcm_sha256="a" * 64,
                    primary_second_pcm_sha256="b" * 64, different_block_pcm_bit_exact=False, header_frames=1440000,
                    required_frames=1440000, deficit_frames=529, deficit_seconds=529 / 48000, empty_eof_observed=True,
                    diagnosis_method="two_fresh_libsndfile_sequential_passes_different_blocks_to_empty_EOF",
                    producer_failure={"path": str(failure_path), "sha256": failure_sha},
                    source_binding={"path": str(source), "sha256": f.a.sha(source)}, source_byte_count=source.stat().st_size,
                    source_stat_signature=f.a.signature(source)[:5], ffmpeg_diagnostic={"status": "unavailable"})
    diagnosis = {"status": "confirmed_39_native_short_inputs_source_blind_policy", "contract_sha256": contract_sha,
                 "run_summary_sha256": run_sha, "policy": f.POLICY, "runtime": {k: "v" for k in ("soundfile", "libsndfile", "numpy")},
                 "decoder_scope": "pinned_libsndfile_sequential_EOF_not_universal_audio_duration", "records": [excluded],
                 "input_bindings": {str(root / "contract.json"): contract_sha, str(run_path): run_sha}, **f.SCOPE}
    dp = base / "diagnosis.json"
    ds = write(dp, diagnosis)
    return {"contract": contract, "run": run, "diagnosis": diagnosis, "contract_sha": contract_sha,
            "pc_sha": pc_sha, "pa_sha": pa_sha, "args": (root, dp, ds, run_path, run_sha, pilot, pa),
            "row": row, "failure": failure, "exclusion": excluded}


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        for target, name, value in ((f, "TOTAL", 9), (f, "ELIGIBLE", 8), (f, "EXCLUDED", 1), (f, "MONO", 1), (f.a, "FRAMES", 8)):
            p = patch.object(target, name, value); p.start(); self.addCleanup(p.stop)
        self.data = fixture(self.base)
        for target, name, value in ((f, "CONTRACT", self.data["contract_sha"]),
                                    (f.a, "PILOT_COMMIT", self.data["pc_sha"]), (f.a, "PILOT_AUDIT", self.data["pa_sha"])):
            p = patch.object(target, name, value); p.start(); self.addCleanup(p.stop)

    def documents(self, data=None):
        d = data or self.data
        return f.validate_documents(d["contract"], d["run"], d["diagnosis"], d["args"][4])

    def test_complete_disjoint_accountability(self):
        rows, fails, excluded = self.documents()
        self.assertEqual((len(rows), len(fails), set(fails) == set(excluded)), (9, 1, True))
        for section, key, value in (("run", "completed", 9), ("run", "status", "complete_DSP_only"),
                                     ("diagnosis", "policy", {**f.POLICY, "padding": True}),
                                     ("diagnosis", "run_summary_sha256", "c" * 64)):
            d = copy.deepcopy(self.data); d[section][key] = value
            with self.assertRaises(ValueError): self.documents(d)

    def test_duplicate_missing_and_mono_overlap_rejected(self):
        for change in (lambda d: d["contract"]["rows"].append(d["row"]),
                       lambda d: d["run"]["failures"].append(d["failure"]),
                       lambda d: d["diagnosis"]["records"].clear(),
                       lambda d: d["contract"]["excluded_native_mono"][0].update(id="id0")):
            d = copy.deepcopy(self.data); change(d)
            with self.assertRaises(ValueError): self.documents(d)

    def test_different_block_pcm_hashes_are_valid_duration_evidence(self):
        d = self.data; bindings = {}
        f.validate_failure(d["exclusion"], d["failure"], d["row"], d["args"][0], bindings)
        self.assertEqual(len(bindings), 2)

    def test_failure_mutations_rejected(self):
        mutations = {"role": "test", "group_id": "other", "component_id": "other", "native_path": "/elsewhere",
                     "native_sha256": "d" * 64, "actual_frames": 1440000, "primary_second_pass_frames": 1439472,
                     "deficit_frames": 0, "deficit_seconds": -1, "empty_eof_observed": False,
                     "different_block_pcm_bit_exact": True, "source_stat_signature": [0] * 5}
        d = self.data
        for key, value in mutations.items():
            record = copy.deepcopy(d["exclusion"]); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.validate_failure(record, d["failure"], d["row"], d["args"][0], {})

    def test_original_inventory_rejects_commit_or_unreceipted_output(self):
        root = self.data["args"][0]; good = {"id" + str(i) for i in range(8)}
        f.inventory(root, good, {"id8"})
        for path in (root / "COMMIT.json", root / "audio" / "id8.wav", root / "audio" / ".id0.wav.tmp"):
            path.write_bytes(b"x")
            with self.assertRaises(ValueError): f.inventory(root, good, {"id8"})
            path.unlink()

    def test_wav_pcm_and_format_mutations_rejected(self):
        root = self.data["args"][0]; wav = root / "audio" / "id0.wav"
        receipt = f.a.load(root / "items" / "id0.json")["payload"]
        f.validate_output(wav, receipt)
        changed = copy.deepcopy(receipt); changed["waveform_float32_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "PCM hash"): f.validate_output(wav, changed)
        sf.write(wav, np.zeros((8, 2)), f.a.RATE, format="WAV", subtype="PCM_16")
        changed.update(file_sha256=f.a.sha(wav), file_bytes=wav.stat().st_size)
        with self.assertRaisesRegex(ValueError, "WAV format"): f.validate_output(wav, changed)

    def test_full_synthetic_closure_preserves_original_and_binds_products(self):
        d = self.data; root = d["args"][0]; out = self.base / "closure"
        original = {str(p): f.a.sha(p) for p in root.rglob("*") if p.is_file()}
        with patch.object(f, "validate_runtime", return_value={"synthetic": True}):
            result = f.finalize(*d["args"], out)
        self.assertEqual(result["status"], f.STATUS)
        self.assertEqual(original, {str(p): f.a.sha(p) for p in root.rglob("*") if p.is_file()})
        manifest, summary, commit = (f.a.load(out / name) for name in ("manifest.json", "summary.json", "COMMIT.json"))
        self.assertEqual(manifest["excluded_short_records"], [d["exclusion"]])
        self.assertEqual(manifest["excluded_native_mono"], d["contract"]["excluded_native_mono"])
        self.assertEqual(summary["attrition"]["source_group"]["S"]["short_attrition_denominator"], 9)
        self.assertEqual(len(summary["pilot_bit_exact_matches"]), 8)
        self.assertFalse(summary["full1656_native_DSP_replay_performed"])
        self.assertFalse((root / "COMMIT.json").exists())
        self.assertFalse((root / "manifest.json").exists())
        self.assertEqual(len(list(out.rglob("*.wav"))), 0)
        for name, binding in commit["products"].items():
            self.assertEqual(f.a.sha(out / name), binding["sha256"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            f.finalize(*d["args"], out)

    def test_no_publication_with_wrong_pin_or_damaged_success(self):
        d = self.data; args = list(d["args"]); args[2] = "0" * 64
        with self.assertRaises(ValueError): f.finalize(*args, self.base / "badpin")
        self.assertFalse((self.base / "badpin").exists())
        wav = args[0] / "audio" / "id0.wav"; wav.write_bytes(b"broken")
        with patch.object(f, "validate_runtime", return_value={}), self.assertRaises(ValueError):
            f.finalize(*d["args"], self.base / "damaged")
        self.assertFalse((self.base / "damaged").exists())

    def test_end_rehash_rejects_source_mutation(self):
        d = self.data; original = f.a.recheck; calls = 0
        def mutate_once(bindings):
            nonlocal calls
            calls += 1
            if calls == 1:
                Path(d["row"]["execution_native_path"]).write_bytes(b"changed")
            original(bindings)
        with patch.object(f, "validate_runtime", return_value={}), patch.object(f.a, "recheck", side_effect=mutate_once):
            with self.assertRaises(ValueError): f.finalize(*d["args"], self.base / "changed")
        self.assertFalse((self.base / "changed").exists())


if __name__ == "__main__":
    unittest.main()
