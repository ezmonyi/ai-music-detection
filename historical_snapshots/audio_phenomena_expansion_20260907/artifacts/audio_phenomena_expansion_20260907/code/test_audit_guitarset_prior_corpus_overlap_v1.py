#!/usr/bin/env python3
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_guitarset_prior_corpus_overlap_v1.py")
SPEC = importlib.util.spec_from_file_location("overlap_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_fp(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                          encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Fixture:
    def __init__(self, root, *, common_pcm=None, incompatible_same_digest=False,
                 identifier_match=False, encoded_match=False):
        self.root = Path(root)
        self.common_pcm = common_pcm or digest("guitar-pcm")
        self.guitarset = self.root / "guitarset"
        self.guitarset.mkdir()
        self._make_guitarset()
        self.inputs = self._make_inputs(incompatible_same_digest, identifier_match, encoded_match)
        self.counts = {key: 1 for key in audit.EXPECTED_COUNTS}

    def _make_guitarset(self):
        rows = []
        for performance in ("comp", "solo"):
            item_id = f"00_BN1-129-Eb_{performance}"
            audio_rel = f"audio/{item_id}_mic.wav"
            annotation_rel = f"annotations/{item_id}.jams"
            audio_path = self.guitarset / audio_rel
            annotation_path = self.guitarset / annotation_rel
            audio_path.parent.mkdir(exist_ok=True)
            annotation_path.parent.mkdir(exist_ok=True)
            audio_path.write_bytes(("audio-" + performance).encode())
            annotation_path.write_bytes(("jams-" + performance).encode())
            audio_fp, annotation_fp = file_fp(audio_path), file_fp(annotation_path)
            rows.append({
                "item_id": item_id, "player_id": "00", "score_id": "BN1-129-Eb",
                "performance": performance, "role": "external_measurement_control_only",
                "class_label": None,
                "audio": {
                    "archive_member": f"audio/{item_id}_mic.wav",
                    "archive_member_sha256": audio_fp["sha256"],
                    "materialized_path": audio_rel, "materialized": audio_fp,
                    "decoded": {
                        "decoded_pcm_sha256": self.common_pcm if performance == "comp"
                        else digest("solo-pcm"),
                        "decoded_pcm_canonical_encoding":
                            "IEEE754 float64 little-endian; C order [frame, channel]; no header",
                        "sample_rate_hz": 16000, "channels": 1, "decoded_frames": 4,
                    },
                },
                "annotation": {"materialized_path": annotation_rel,
                               "archive_member_sha256": annotation_fp["sha256"],
                               "materialized": annotation_fp},
            })
        write_jsonl(self.guitarset / "source_manifest.jsonl", rows)
        (self.guitarset / "validation_summary.json").write_text(
            json.dumps({"status": "passed_archive_crc_materialization_and_physical_decode_not_measurement_admission",
                        "counts": {"exact_audio_annotation_pairs": 2}}),
            encoding="utf-8")
        products = {}
        for path in sorted(self.guitarset.rglob("*")):
            if path.is_file():
                products[path.relative_to(self.guitarset).as_posix()] = file_fp(path)
        commit = {"status": "committed",
                  "kind": "guitarset_archive_validation_materialization_v1",
                  "products": products}
        (self.guitarset / "COMMIT.json").write_text(
            json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        self.commit_sha = file_fp(self.guitarset / "COMMIT.json")["sha256"]
        self.comp_file_sha = products["audio/00_BN1-129-Eb_comp_mic.wav"]["sha256"]

    def _make_inputs(self, incompatible_same_digest, identifier_match, encoded_match):
        base = self.root / "inputs"
        base.mkdir()
        paths = {}
        values = {key: digest(key) for key in ("source", "view10", "view60", "stdfile",
                                               "f32", "crop64", "crop32", "nsynth-file")}
        master_id = "00_BN1-129-Eb_comp" if identifier_match else "unrelated_master_item"
        master_hash = self.comp_file_sha if encoded_match else values["source"]
        for adapter in ("master_10s", "master_30s"):
            path = base / f"{adapter}.csv"
            row = {key: "unrelated" for key in audit.CSV_REQUIRED[adapter]}
            row.update(id=master_id, track="unrelated track",
                       source_id="GuitarSet" if identifier_match else "corpus",
                       source_group="corpus", group_id="group", artist_or_creator="artist",
                       raw_sha256=master_hash, source_audio_path="/source/unrelated.wav",
                       source_locator="catalog:unrelated")
            write_csv(path, audit.CSV_REQUIRED[adapter], [row]); paths[adapter] = path

        path = base / "materialization.jsonl"
        row = {key: "unrelated" for key in audit.JSONL_REQUIRED["materialization_2641"]}
        row.update(item_id="materialized_unrelated", source_id="corpus", group_id="group",
                   native_path="/source/unrelated.wav", native_sha256=values["source"],
                   view_10s_sha256=values["view10"], view_max60s_sha256=values["view60"])
        write_jsonl(path, [row]); paths["materialization_2641"] = path

        path = base / "v6.csv"
        write_csv(path, audit.CSV_REQUIRED["v6_metadata"], [{"id": "v6_unrelated",
                  "source_group": "corpus", "group_id": "group", "native_sample_rate_hz": "16000"}])
        paths["v6_metadata"] = path

        path = base / "equal60.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["equal60_lineage"]}
        row.update(item_id="equal_unrelated", source_id="corpus", group_id="group",
                   standardized_path="/std.wav", standardized_sr="16000",
                   standardized_channels="1", standardized_frames="4",
                   standardized_file_sha256=values["stdfile"], source_sample_rate="16000",
                   source_channels="1", source_total_frames="4",
                   source_audio_path="/source/unrelated.wav", source_audio_sha256=values["source"],
                   standardized_waveform_sha256=(self.common_pcm if incompatible_same_digest
                                                 else values["f32"]))
        write_csv(path, audit.CSV_REQUIRED["equal60_lineage"], [row]); paths["equal60_lineage"] = path

        path = base / "mureka.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["mureka"]}
        row.update(item_id="mureka_unrelated", source_id="Mureka_v9", source_group="Mureka_v9",
                   group_id="group", source_audio_path="/source/mureka.mp3",
                   source_audio_sha256=digest("mureka-source"), source_sample_rate="16000",
                   source_channels="1", sf_actual_read_frames="4",
                   sf_sequential_float64_sha256=self.common_pcm,
                   sf_sequential_float32_sha256=digest("mureka-full32"), crop_start_frame="0",
                   crop_frames="4", native_crop_float64_sha256=digest("mureka-crop64"),
                   native_crop_float32_sha256=digest("mureka-crop32"), standardized_sr="16000",
                   standardized_channels="1", standardized_frames="4",
                   standardized_waveform_sha256=digest("mureka-std32"),
                   standardized_file_sha256=digest("mureka-std-file"))
        write_csv(path, audit.CSV_REQUIRED["mureka"], [row]); paths["mureka"] = path

        path = base / "saraga.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["saraga"]}
        saraga_source, saraga_crop = digest("saraga-source"), digest("saraga-crop")
        row.update(id="saraga_unrelated", source_id="saraga", source_group="saraga",
                   group_id="group", audio_path="/source/saraga.mp3",
                   registered_raw_sha256=saraga_source, source_audio_sha256=saraga_source,
                   physical_frames="8", physical_sample_rate_hz="16000", physical_channels="1",
                   crop_start_frame="2", crop_frames="4",
                   native_crop_float64_sha256=saraga_crop,
                   native_crop_float32_sha256=digest("saraga-crop32"))
        write_csv(path, audit.CSV_REQUIRED["saraga"], [row]); paths["saraga"] = path
        proofs = base / "saraga_proofs"
        proof_dir = proofs / "items" / "saraga_unrelated"; proof_dir.mkdir(parents=True)
        proof = {"item_id": "saraga_unrelated", "interval_proof": {
            "whole_pcm_sha256": digest("saraga-whole"),
            "whole_pcm_encoding": "IEEE754 float64 little-endian; C order [frame, channel]; no header",
            "observed_actual_frames": 8, "native_sample_rate_hz": 16000,
            "native_channels": 1, "native_float64_sha256": saraga_crop}}
        (proof_dir / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
        paths["saraga_proofs"] = proofs

        path = base / "nsynth.jsonl"
        row = {"id": "nsynth_unrelated", "path": "/source/nsynth.wav",
               "file_sha256": values["nsynth-file"],
               "pcm_sha256": self.common_pcm if incompatible_same_digest else digest("pcm16"),
               "native_sample_rate": 16000, "native_channels": 1, "frames": 4}
        write_jsonl(path, [row]); paths["nsynth"] = path

        path = base / "vocal.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["vocalset_pairs"]}
        row.update(pair_id="pair", singer="singer", content_id="content",
                   straight_filename="straight.wav", straight_audio_path="/straight.wav",
                   straight_audio_sha256=digest("straight"), straight_archive_member="straight.wav",
                   straight_sample_rate_hz="44100", straight_frames="10",
                   vibrato_filename="vibrato.wav", vibrato_audio_path="/vibrato.wav",
                   vibrato_audio_sha256=digest("vibrato"), vibrato_archive_member="vibrato.wav",
                   vibrato_sample_rate_hz="44100", vibrato_frames="10")
        write_csv(path, audit.CSV_REQUIRED["vocalset_pairs"], [row]); paths["vocalset_pairs"] = path

        path = base / "mir.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["mir1k"]}
        row.update(stem="mir_unrelated", singer="singer", song_id="song", clip_id="clip",
                   wav_remote_path="remote.wav", wav_local_path="/mir.wav",
                   wav_sha256=digest("mir"), sample_rate_hz="16000", channels="2", audio_frames="10")
        write_csv(path, audit.CSV_REQUIRED["mir1k"], [row]); paths["mir1k"] = path

        path = base / "musdb.jsonl"
        write_jsonl(path, [{"track_id": "musdb_unrelated", "track_name": "track",
                            "dataset": "MUSDB18", "source_container": "track.stem.mp4",
                            "mixture": "/mixture.wav", "mixture_sha256": digest("musdb"),
                            "sample_rate": 44100, "channels": 2}])
        paths["musdb"] = path

        path = base / "timbre_pairs.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["timbre_pairs"]}
        row.update(pair_id="tpair", corpus="MUSDB", instrument_group="guitar",
                   anchor_source_id="anchor", peer_source_id="peer",
                   anchor_source_path="/anchor.wav", peer_source_path="/peer.wav",
                   anchor_source_sha256=digest("anchor"), peer_source_sha256=digest("peer"))
        write_csv(path, audit.CSV_REQUIRED["timbre_pairs"], [row]); paths["timbre_pairs"] = path

        path = base / "timbre_chunks.csv"
        row = {key: "unrelated" for key in audit.CSV_REQUIRED["timbre_chunks"]}
        row.update(corpus="MUSDB", source_id="anchor", source_path="/anchor.wav",
                   track_name="track", instrument_group="guitar", source_sha256=digest("anchor"),
                   native_sr_hz="44100", native_duration_sec="60")
        write_csv(path, audit.CSV_REQUIRED["timbre_chunks"], [row]); paths["timbre_chunks"] = path
        return paths

    def run(self, output):
        return audit.run(self.guitarset, self.commit_sha, self.inputs, output,
                         expected_counts=self.counts, _expected_guitarset_shape=(1, 1))


class OverlapAuditTests(unittest.TestCase):
    def test_scoped_json_commit_and_exact_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, identifier_match=True, encoded_match=True)
            output = Path(tmp) / "out"
            report = fixture.run(output)
            self.assertTrue((output / "COMMIT.json").is_file())
            self.assertEqual(json.loads((output / "overlap_report.json").read_text()), report)
            self.assertGreaterEqual(report["candidate_counts"]["identifier_candidates"], 2)
            self.assertTrue(any(row.get("candidate_scope") ==
                                "declared_dataset_or_upstream_source"
                                for row in report["comparisons"]["identifier_candidates"]))
            self.assertGreaterEqual(report["candidate_counts"]["exact_encoded_file_matches"], 2)
            matches = report["comparisons"]["exact_compatible_float64_pcm_matches"]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["prior"]["adapter"], "mureka")
            self.assertFalse(report["claims"]["global_non_overlap_established"])
            self.assertFalse(report["claims"]["independent_reviewer_acceptance"])

    def test_pcm16_and_float32_same_digest_are_not_float64_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, incompatible_same_digest=True)
            # Remove the one intentionally compatible Mureka f64 equality.
            rows, _ = audit.read_csv(fixture.inputs["mureka"], "mureka")
            rows[0]["sf_sequential_float64_sha256"] = digest("different-f64")
            write_csv(fixture.inputs["mureka"], audit.CSV_REQUIRED["mureka"], rows)
            report = fixture.run(Path(tmp) / "out")
            self.assertEqual(report["candidate_counts"]["exact_compatible_float64_pcm_matches"], 0)
            incompatible = report["comparisons"]["incompatible_pcm_digest_coincidences"]
            self.assertEqual({row["prior"]["representation"] for row in incompatible},
                             {audit.F32_REPRESENTATION, audit.PCM16_REPRESENTATION})

    def test_duplicate_source_hash_multiplicity_is_retained_but_accounting_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, encoded_match=True)
            report = fixture.run(Path(tmp) / "out")
            matched_adapters = [row["prior"]["adapter"] for row in
                                report["comparisons"]["exact_encoded_file_matches"]]
            self.assertIn("master_10s", matched_adapters)
            self.assertIn("master_30s", matched_adapters)
            accounting = report["accounting"]
            self.assertLess(accounting["deduplicated_connected_source_entities"],
                            accounting["logical_audio_evidence_entries"])
            self.assertTrue(accounting["multiplicity_preserved_in_match_lists"])

    def test_unmatched_identifiers_never_become_global_nonoverlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            report = fixture.run(Path(tmp) / "out")
            self.assertEqual(report["candidate_counts"]["identifier_candidates"], 0)
            self.assertFalse(report["claims"]["global_non_overlap_established"])
            self.assertEqual(report["scope"]["transformed_overlap_coverage"], 0)
            self.assertTrue(report["claims"]["parent_adjudication_required"])

    def test_missing_manifest_path_fails_visibly_without_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            fixture.inputs["mir1k"].unlink()
            output = Path(tmp) / "out"
            with self.assertRaises(audit.AuditError):
                fixture.run(output)
            self.assertTrue((output / "EXECUTION_FAILURE.json").is_file())
            self.assertFalse((output / "COMMIT.json").exists())

    def test_missing_required_field_fails_visibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            write_csv(fixture.inputs["mir1k"], ["stem"], [{"stem": "broken"}])
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(audit.AuditError, "missing required fields"):
                fixture.run(output)
            self.assertTrue((output / "EXECUTION_FAILURE.json").exists())
            self.assertFalse((output / "COMMIT.json").exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            output = Path(tmp) / "out"; output.mkdir()
            sentinel = output / "sentinel"; sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditError, "new exclusive"):
                fixture.run(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_guitarset_product_inventory_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            (fixture.guitarset / "uncommitted.txt").write_text("unexpected", encoding="utf-8")
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(audit.AuditError, "product inventory mismatch"):
                fixture.run(output)
            self.assertFalse((output / "COMMIT.json").exists())

    def test_extra_saraga_proof_fails_exact_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            extra = fixture.inputs["saraga_proofs"] / "items" / "extra"
            extra.mkdir(); (extra / "proof.json").write_text("{}", encoding="utf-8")
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(audit.AuditError, "proof inventory mismatch"):
                fixture.run(output)
            self.assertFalse((output / "COMMIT.json").exists())


if __name__ == "__main__":
    unittest.main()
