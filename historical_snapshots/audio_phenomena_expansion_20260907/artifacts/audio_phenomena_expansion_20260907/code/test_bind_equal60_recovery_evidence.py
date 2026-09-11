#!/usr/bin/env python3
"""Small synthetic offline fixtures; no models, GPU, audio or real run writes."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bind_equal60_recovery_evidence as B


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_features(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    code = Path(B.__file__).parent
    protocol = code.parent / "EXACT60_BEAT_SERIALIZATION_RECOVERY_EN.md"
    inference, audit_dir = root / "prepared/inference", root / "audit"
    (audit_dir / "features").mkdir(parents=True)
    for name in ("receipts", "logs", "beats", "recovery_empty_beats"):
        (inference / name).mkdir(parents=True, exist_ok=True)
    items, features = [], []
    for index in range(6):
        item_id = f"synthetic_v4_binder_{index}"
        audio = dict(path=str(root / "prepared/audio" / (item_id + ".wav")), sha256=h(item_id + "audio"),
                     frames=2646000, sample_rate=44100, channels=2, finite=True, subtype="FLOAT")
        beat_sha = B.EMPTY_SHA if index in (0, 2) else h(item_id + "beats")
        item = dict(item_id=item_id, input=audio,
                    stems={s: {**audio, "sha256": h(item_id + s)} for s in ("bass", "drums", "other", "vocals")},
                    beats=dict(sha256=beat_sha, status="empty_unavailable" if index in (0, 2) else "nonempty", beat_count=0 if index in (0, 2) else 120),
                    structure=dict(sha256=h(item_id + "structure"), input_path=audio["path"]),
                    spectrogram=dict(sha256=h(item_id + "spec"), shape=[4, 6000, 81]))
        items.append(item)
        hashes = {"source_audio_sha256": audio["sha256"], **{s + "_sha256": a["sha256"] for s, a in item["stems"].items()},
                  "beats_sha256": beat_sha, "structure_sha256": item["structure"]["sha256"]}
        feature = dict(item_id=item_id, status="complete", duration_sec="60", input_hashes=json.dumps(hashes),
                       r_eligible="0" if index in (0, 2) else "1", beat_output_status="empty" if index in (0, 2) else "nonempty",
                       n_beats="0" if index in (0, 2) else "120", n_downbeats="0" if index in (0, 2) else "30")
        feature.update({c: "nan" if index in (0, 2) else "0.4" for c in B.R_COLUMNS + B.BAR_COLUMNS})
        feature.update({c: "1.25" for c in B.DURATION_COLUMNS})
        features.append(feature)
    runtime_checks = {"checkpoint": B.CHECKPOINT, "bias": B.P.BIAS_SHA}
    contract = dict(rows=6, manifest_sha256=h("manifest"), output_root=str(inference), runtime_root="/synthetic/runtime",
                    shards=[dict(index=i, rows=2, sha256=h("".join(r["input"]["path"] + "\n" for r in items[i * 2:i * 2 + 2]))) for i in range(3)],
                    runtime_code_checkpoint_bias_checks=runtime_checks, code_sha256=B.sha(code / "run_equal60_inference_batches.py"),
                    dependencies={name: B.sha(code / name) for name in ("verify_extract_equal60.py", "materialize_equal60_inputs_v2.py")})
    dump(inference / "run_contract.json", contract)
    contract_sha = B.sha(inference / "run_contract.json")
    dump(inference / "completion.json", dict(status="passed", rows=6, completed_stage_shards=6, run_contract_sha256=contract_sha))
    for shard in range(3):
        selected = items[shard * 2:shard * 2 + 2]
        if shard == 2:
            for stage in ("beats", "allinone"):
                products = {p: {"sha256": value, "bytes": 20} for item in selected for p, value in B.product_hashes(inference, item, stage).items()}
                dump(inference / "receipts" / f"{stage}_{shard:02d}.json", dict(status="passed", stage=stage, shard_index=shard,
                     run_contract_sha256=contract_sha, item_ids=[r["item_id"] for r in selected], gpu=5, outputs=products, log_sha256=h("ordinary successful log")))
            continue
        recovered = selected[0]
        item_id = recovered["item_id"]
        log = inference / "logs" / f"beats_{shard:02d}.log"
        log.write_text(f'Could not process "{recovered["input"]["path"]}". Rerun with this file alone for details.\n')
        code_name = "recover_equal60_empty_beats.py" if shard == 0 else "recover_equal60_empty_beats_v2.py"
        invocation = ["/synthetic/runtime/venv/bin/python", str(code / "recover_equal60_empty_beats_v2.py"),
                      "--prepared-dir", str(inference.parent), "--output-root", str(inference), "--gpu", "5"]
        sidecar = dict(status="verified_unavailable_no_beat_events", item_id=item_id, input_path=recovered["input"]["path"],
                       input_sha256=recovered["input"]["sha256"], run_contract_sha256=contract_sha,
                       recovery_code_sha256=B.RECOVERY_CODES[code_name], checkpoint_sha256=B.CHECKPOINT,
                       dependency_sha256={"/synthetic/" + k: v for k, v in B.UPSTREAM_CODES.items()}, gpu=5,
                       gpu_name="NVIDIA GeForce RTX 5090", float16=True, dbn=False, beats=[], downbeats=[0.0],
                       error="Not all downbeats are beats.", original_log_sha256=B.sha(log))
        if shard:
            sidecar.update(recovery_invocation=invocation, recovery_api=B.REPLAY_API)
        side_path = inference / "recovery_empty_beats" / (item_id + ".json")
        dump(side_path, sidecar)
        (inference / "beats" / (item_id + ".beats")).write_bytes(b"")
        for stage in ("beats", "allinone"):
            products = {p: {"sha256": value, "bytes": 0 if value == B.EMPTY_SHA else 20}
                        for item in selected for p, value in B.product_hashes(inference, item, stage).items()}
            receipt = dict(status="passed", stage=stage, shard_index=shard, run_contract_sha256=contract_sha,
                           item_ids=[r["item_id"] for r in selected], gpu=5, outputs=products, log_sha256=B.sha(log))
            if stage == "beats":
                products[str(side_path)] = dict(sha256=B.sha(side_path), bytes=side_path.stat().st_size)
                receipt.update(recovery_status="same_model_replay_verified_empty_unavailable", recovery_code_sha256=B.RECOVERY_CODES[code_name],
                               recovery_sidecars=[str(side_path)], preserved_existing_sha256=B.product_hashes(inference, selected[1], "beats"))
                if shard:
                    receipt.update(original_shard_command=B.original_command(contract, selected), recovery_invocation=invocation)
                else:
                    receipt["command"] = B.original_command(contract, selected)
            dump(inference / "receipts" / f"{stage}_{shard:02d}.json", receipt)
    audit = dict(status="passed", rows=6, items=items, manifest_sha256=h("manifest"), materialization_contract_sha256=h("materialized"),
                 code_sha256=B.sha(code / "verify_extract_equal60.py"), runtime_code_checkpoint_bias_checks=runtime_checks)
    dump(audit_dir / "strict_inference_audit.json", audit)
    write_features(audit_dir / "features/expanded_features_60s.csv", features)
    dump(audit_dir / "extraction_receipt.json", dict(status="passed", rows=6, classifier_fitted=False, no_short_padding_possible=True,
         strict_inference_audit_sha256=B.sha(audit_dir / "strict_inference_audit.json"), features_sha256=B.sha(audit_dir / "features/expanded_features_60s.csv")))
    return audit_dir, inference, protocol


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.audit, self.inference, self.protocol = fixture(self.root)
        self.output = self.root / "bound"

    def run_binding(self):
        return B.bind(self.audit, self.inference, self.output, self.protocol, synthetic=True)

    def json_mutate(self, path, mutate):
        value = B.parse_json(path.read_text())
        mutate(value)
        dump(path, value)

    def sidecar_mutate(self, mutate):
        path = self.inference / "recovery_empty_beats/synthetic_v4_binder_0.json"
        self.json_mutate(path, mutate)
        self.json_mutate(self.inference / "receipts/beats_00.json", lambda r: r["outputs"].__setitem__(str(path), dict(sha256=B.sha(path), bytes=path.stat().st_size)))

    def test_complete_self_contained_binding_preserves_original_and_raw_arrays(self):
        before = {str(p): B.sha(p) for p in self.root.rglob("*") if p.is_file()}
        report = self.run_binding()
        self.assertEqual(report["status"], "bound_completed_recovery_evidence")
        derivative = B.parse_json((self.output / "extraction_receipt.json").read_text())
        binding = derivative.pop("recovery_evidence_binding")
        self.assertEqual(derivative, B.parse_json((self.audit / "extraction_receipt.json").read_text()))
        self.assertEqual(binding["original_extraction_receipt"]["original_utf8"], (self.audit / "extraction_receipt.json").read_text())
        payload_hash = binding.pop("binding_payload_sha256")
        self.assertEqual(payload_hash, B.digest(binding))
        self.assertEqual(len(binding["all_stage_receipts_sha256"]), 6)
        self.assertEqual(len(binding["beat_stage_receipts"]), 3)
        self.assertEqual(len(binding["v1_provenance_corrections"]), 1)
        for side in binding["recovery_sidecars"].values():
            self.assertEqual(side["content"]["beats"], [])
            self.assertEqual(side["content"]["downbeats"], [0.0])
        for row in binding["recovered_feature_rows"].values():
            self.assertTrue(all(row[c] == "nan" for c in B.R_COLUMNS + B.BAR_COLUMNS))
            self.assertTrue(all(row[c] == "1.25" for c in B.DURATION_COLUMNS))
        self.assertFalse(binding["audio_stems_spectrograms_reopened"])
        self.assertEqual(before, {p: B.sha(p) for p in before})
        # No synthetic audio, stems or spectrogram files exist: success proves no reopening.
        self.assertFalse((self.root / "prepared/audio").exists())
        with self.assertRaisesRegex(ValueError, "existing output"):
            self.run_binding()

    def test_no_completed_result_no_output(self):
        (self.inference / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing evidence"):
            self.run_binding()
        self.assertFalse(self.output.exists())

    def test_real_incomplete_result_no_output(self):
        (self.inference / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing evidence"):
            B.bind(self.audit, self.inference, self.output, self.protocol)
        self.assertFalse(self.output.exists())

    def test_missing_sidecar(self):
        (self.inference / "recovery_empty_beats/synthetic_v4_binder_0.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing evidence"):
            self.run_binding()

    def test_changed_features(self):
        with (self.audit / "features/expanded_features_60s.csv").open("a") as handle:
            handle.write(" ")
        with self.assertRaises(ValueError):
            self.run_binding()

    def test_false_zero_R_after_receipt_rehash(self):
        path = self.audit / "features/expanded_features_60s.csv"
        rows = B.rows_from_bytes(path.read_bytes())
        rows[0][B.R_COLUMNS[0]] = "0"
        write_features(path, rows)
        self.json_mutate(self.audit / "extraction_receipt.json", lambda r: r.__setitem__("features_sha256", B.sha(path)))
        with self.assertRaisesRegex(ValueError, "must be NaN"):
            self.run_binding()

    def test_false_zero_P_bars_after_receipt_rehash(self):
        path = self.audit / "features/expanded_features_60s.csv"
        rows = B.rows_from_bytes(path.read_bytes())
        rows[0][B.BAR_COLUMNS[0]] = "0"
        write_features(path, rows)
        self.json_mutate(self.audit / "extraction_receipt.json", lambda r: r.__setitem__("features_sha256", B.sha(path)))
        with self.assertRaisesRegex(ValueError, "must be NaN"):
            self.run_binding()

    def test_invalid_arrays_after_rehash(self):
        self.sidecar_mutate(lambda s: s.__setitem__("downbeats", [0, 61]))
        with self.assertRaisesRegex(ValueError, "Invalid orphan"):
            self.run_binding()

    def test_sidecar_model_parameter_mismatch(self):
        self.sidecar_mutate(lambda s: s.__setitem__("float16", False))
        with self.assertRaisesRegex(ValueError, "precision"):
            self.run_binding()

    def test_count_and_contract_mismatch(self):
        self.json_mutate(self.inference / "completion.json", lambda r: r.__setitem__("completed_stage_shards", 134))
        with self.assertRaisesRegex(ValueError, "all134"):
            self.run_binding()
        self.json_mutate(self.inference / "completion.json", lambda r: r.update(completed_stage_shards=6, run_contract_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "completion/manifest"):
            self.run_binding()

    def test_missing_stage_receipt(self):
        (self.inference / "receipts/allinone_01.json").unlink()
        with self.assertRaisesRegex(ValueError, "stage receipts"):
            self.run_binding()

    def test_extra_unreferenced_sidecar(self):
        dump(self.inference / "recovery_empty_beats/extra.json", {"item_id": "extra"})
        with self.assertRaisesRegex(ValueError, "sidecar inventory"):
            self.run_binding()

    def test_duplicate_sidecar_reference(self):
        self.json_mutate(self.inference / "receipts/beats_00.json", lambda r: r["recovery_sidecars"].append(r["recovery_sidecars"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate recovery"):
            self.run_binding()

    def test_product_hash_differs_from_strict_audit(self):
        self.json_mutate(self.inference / "receipts/allinone_00.json", lambda r: next(iter(r["outputs"].values())).__setitem__("sha256", "0" * 64))
        with self.assertRaisesRegex(ValueError, "Product receipt"):
            self.run_binding()

    def test_changed_evidence_before_publication(self):
        original = B.Snapshot.check
        checks = []
        def check(snapshot):
            checks.append(True)
            if len(checks) == 2:
                with (self.audit / "extraction_receipt.json").open("a") as f:
                    f.write(" ")
            original(snapshot)
        with patch.object(B.Snapshot, "check", check), self.assertRaisesRegex(ValueError, "changed during"):
            self.run_binding()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob("bound.tmp.*")))

    def test_synthetic_fixture_refused_in_real_mode(self):
        with self.assertRaisesRegex(ValueError, "full1604"):
            B.bind(self.audit, self.inference, self.output, self.protocol)


if __name__ == "__main__":
    unittest.main()
