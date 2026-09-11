#!/usr/bin/env python3
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("origin_plan_v2", HERE / "build_native30_origin_plan_v2.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
H = "a" * 64
F = "b" * 64


def common_fixtures():
    screen = {"id": "x", "label": "1", "source_group": "S", "group_id": "g", "component_id": "c", "role": "development",
              "duration_exposure_candidate": True, "exclusion_reasons": [], "input_occurrences": ["master30"], "selected_metadata_input": "master30"}
    freeze = {"id": "x", "label": "1", "source_group": "S", "group_id": "g", "role": "development"}
    evidence = {"id": "x", "label": "1", "source_group": "S", "component_id": "c", "native_stereo_metadata_eligible": True,
                "native_evidence": {"sha256": H, "duration_s": 30, "channels": 2},
                "join_fields": {"canonical_id": "x", "raw_sha256": H, "native_duration_s": 30}}
    return screen, freeze, evidence


def mureka_fixtures():
    freeze = {"group_id": "g", "native_sample_rate_hz": "44100",
              "freeze_evidence": {"standardized_path": "/out.wav", "standardized_file_sha256": F,
                                  "standardized_frames": 2646000, "standardized_sr": 44100, "native_channels": 2}}
    receipt = {"source_audio_sha256": H, "source_audio_path": "/raw.mp3", "standardized_path": "/out.wav", "standardized_file_sha256": F,
               "crop_start_frame": "100", "crop_frames": "2646000", "crop_end_frame_exclusive": "2646100",
               "native_crop_float64_sha256": F, "source_sample_rate": "44100", "source_channels": "2", "group_id": "g",
               "decoder_admission_claim": "acquisition_center_interval_verified_not_full_stream_decoder_agreement", "center_seek_equals_sequential_float64": "True",
               "standardized_frames": "2646000", "standardized_sr": "44100"}
    native = {"native_sample_rate_hz": "44100", "physical_channels": "2", "crop_start_frame": "100", "crop_frames": "2646000",
              "crop_end_frame_exclusive": "2646100", "native_crop_float64_sha256": F, "source_audio_sha256": H, "audio_path": "/raw.mp3",
              "decoder_admission_claim": "acquisition_center_interval_verified_not_full_stream_decoder_agreement", "center_seek_equals_sequential_float64": "True"}
    item = {"id": "123", "sha256": H, "sample_rate": 44100, "channels": 2, "contract_sha256": MODULE.EXPECTED["mureka_acquisition_contract"], "reference_group_id": "g"}
    contract = {"acquisition_contract_sha256": MODULE.EXPECTED["mureka_acquisition_contract"],
                "decoder_admission_claim": "acquisition_center_interval_verified_not_full_stream_decoder_agreement",
                "configuration": {"sample_rate_hz": 44100, "channels": 2, "crop_frames": 2646000,
                                  "full_stream_decoder_equality_required": False, "seek_agreement": "exact_float64_array_equal"}}
    acquisition = {"status": "frozen_for_acquisition_only", "classifier_authorized": False}
    return receipt, native, item, freeze, contract, acquisition


def saraga_fixtures():
    freeze = {"group_id": "g", "native_sample_rate_hz": "44100",
              "freeze_evidence": {"native_channels": 2, "standardized_path": "/out.wav", "standardized_file_sha256": F,
                                  "standardized_frames": 2646000}}
    receipt = {"original_source_audio_path": "/raw.mp3", "original_source_audio_sha256": H, "native_sample_rate_hz": "44100",
               "standardized_file_sha256": F, "standardized_path": "/out.wav", "audio_path": "/out.wav", "standardized_frames": "2646000", "group_id": "g"}
    native = {"native_sample_rate_hz": "44100", "physical_channels": "2", "crop_start_frame": "100", "crop_frames": "2646000",
              "crop_end_frame_exclusive": "2646100", "native_crop_float64_sha256": F, "source_audio_sha256": H,
              "registered_raw_sha256": H, "audio_path": "/raw.mp3", "group_id": "g"}
    interval = {"source_audio_path": "/raw.mp3", "raw_hashes_after": {"sha256": H}, "crop_start_frame": 100, "crop_frames": 2646000,
                "crop_end_frame_exclusive": 2646100, "retained_crop_frames": 2646000, "native_float64_sha256": F, "native_sample_rate_hz": 44100,
                "native_channels": 2, "observed_header": {"sample_rate": 44100, "channels": 2, "decoder_format": "MP3", "decoder_subtype": "MPEG_LAYER_III"},
                "seek_proof": {"requested_start_frame": 100, "returned_start_frame": 100, "requested_frames": 2646000,
                               "observed_frames": 2646000, "float64_sha256": F, "exact_array_equal": True, "exact_bytes_equal": True}}
    proof = {"interval_proof": interval, "wav": {"file_sha256": F}}
    physical = {"record": {"measurement": {"raw_hashes_after_decode": {"sha256": H}, "sample_rate": 44100, "channels": 2,
                                                  "decoder_format": "MP3", "decoder_subtype": "MPEG_LAYER_III"}}}
    return receipt, native, proof, physical, freeze


class OriginPlanV2Test(unittest.TestCase):
    def test_strict_json_and_csv(self):
        with self.assertRaises(ValueError): MODULE.strict_json_bytes(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError): MODULE.strict_json_bytes(b'{"a":NaN}')
        with self.assertRaises(ValueError): MODULE.strict_csv_bytes(b'id,id\na,b\n', "id")
        with self.assertRaises(ValueError): MODULE.strict_csv_bytes(b'id,x\na,1\na,2\n', "id")

    def test_strict_crop_types_and_bounds(self):
        for bad in (True, -1, 1.5, "1.5"):
            with self.assertRaises(ValueError): MODULE.approved_region_center30(bad, 60 * 44100, 44100, None, "receipt", False)
        for bad_rate in (False, 0, -44100, 44100.5):
            with self.assertRaises(ValueError): MODULE.full_native_center30(30, bad_rate)
        with self.assertRaises(ValueError): MODULE.full_native_center30(float("nan"), 44100)
        with self.assertRaises(ValueError): MODULE.full_native_center30(60.0001, 44100)

    def test_header_is_never_actual_eof(self):
        plan = MODULE.full_native_center30(30.08, 48000, 1443840)
        self.assertIsNone(plan["start_frame"])
        self.assertIsNone(plan["actual_decoded_frames"])
        self.assertTrue(plan["physical_frame_validation_required"])
        self.assertEqual(plan["header_suggestion_not_execution_coordinate"]["header_suggested_start_frame"], 1920)
        self.assertEqual(plan["header_suggestion_not_execution_coordinate"]["authority"], "container_header_only_not_actual_decoded_eof")

    def test_region_schema_is_unambiguous_and_hash_policy(self):
        region, plan = MODULE.approved_region_center30(100, 60 * 48000, 48000, H, "verified_native_interval_hash", True)
        self.assertEqual(region, {"start_frame": 100, "frames": 2880000, "float64_sha256": H, "evidence_strength": "verified_native_interval_hash"})
        self.assertEqual(plan["start_frame"], 720100); self.assertEqual(plan["frames"], 1440000)
        region, _ = MODULE.approved_region_center30(0, 60 * 44100, 44100, None, "frozen_native_frame_bounds_from_receipt", False)
        self.assertIsNone(region["float64_sha256"])
        with self.assertRaises(ValueError): MODULE.approved_region_center30(0, 60 * 44100, 44100, None, "verified", True)

    def test_common_join_mutations(self):
        screen, freeze, evidence = common_fixtures()
        MODULE.validate_common(screen, "prior60", freeze=freeze); MODULE.validate_common(screen, "new", evidence=evidence)
        for obj, key, bad, mode in ((screen, "duration_exposure_candidate", False, "screen"), (screen, "exclusion_reasons", ["x"], "screen"),
                                    (screen, "role", "evaluation", "screen"), (freeze, "group_id", "other", "freeze"),
                                    (evidence, "source_group", "other", "evidence"), (evidence, "label", "0", "evidence"),
                                    (evidence, "component_id", "other", "evidence")):
            s, f, e = copy.deepcopy((screen, freeze, evidence)); {"screen": s, "freeze": f, "evidence": e}[mode][key] = bad
            with self.assertRaises(ValueError): MODULE.validate_common(s, "x", freeze=f if mode != "evidence" else None, evidence=e if mode == "evidence" else None)

    def test_mureka_join_accepts_complete_match(self):
        r, n, i, f, c, a = mureka_fixtures()
        joined = MODULE.validate_mureka_join("music8k_mureka_v9_123", r, n, i, f, c, a)
        self.assertEqual((joined["start"], joined["frames"], joined["float64_sha256"]), (100, 2646000, F))

    def test_mureka_join_rejects_mutations(self):
        for target, key, bad in (("receipt", "standardized_path", "/wrong.wav"), ("receipt", "standardized_file_sha256", H),
                                 ("native", "crop_start_frame", "101"), ("receipt", "native_crop_float64_sha256", H),
                                 ("item", "sample_rate", 48000), ("item", "channels", 1)):
            r, n, i, f, c, a = copy.deepcopy(mureka_fixtures()); {"receipt": r, "native": n, "item": i}[target][key] = bad
            with self.assertRaises(ValueError): MODULE.validate_mureka_join("music8k_mureka_v9_123", r, n, i, f, c, a)

    def test_saraga_join_accepts_complete_match(self):
        r, n, p, ph, f = saraga_fixtures(); joined = MODULE.validate_saraga_join("saraga_hindustani_x", r, n, p, ph, f)
        self.assertEqual((joined["start"], joined["frames"], joined["decoder_format"]), (100, 2646000, "MP3"))

    def test_saraga_join_rejects_mutations(self):
        for target, path, bad in (("native", ("crop_start_frame",), "101"), ("proof", ("interval_proof", "native_float64_sha256"), H),
                                  ("physical", ("record", "measurement", "decoder_format"), "WAV"), ("receipt", ("standardized_file_sha256",), H)):
            r, n, p, ph, f = copy.deepcopy(saraga_fixtures()); obj = {"receipt": r, "native": n, "proof": p, "physical": ph}[target]
            for key in path[:-1]: obj = obj[key]
            obj[path[-1]] = bad
            with self.assertRaises(ValueError): MODULE.validate_saraga_join("saraga_hindustani_x", r, n, p, ph, f)

    def test_atomic_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"; first = MODULE.atomic_exclusive_json(target, {"status": "draft"})
            self.assertEqual(first, MODULE.digest(target.read_bytes()))
            with self.assertRaises(FileExistsError): MODULE.atomic_exclusive_json(target, {"status": "changed"})
            self.assertEqual(json.loads(target.read_text()), {"status": "draft"})


if __name__ == "__main__": unittest.main()
