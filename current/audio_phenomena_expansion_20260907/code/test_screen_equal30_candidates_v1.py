"""Synthetic tests for the metadata-only equal-30 feasibility screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import screen_equal30_candidates_v1 as S


FIELDS = [
    "id", "label", "source_group", "role", "original_role", "group_id", "condition_id",
    "native_duration_s", "raw_sha256", "available", "evaluation_allowed", "native_channels",
    "standardized_channels", "source_audio_path", "duration_view", "native_sample_rate_hz", "duration_sec",
]
ADD_FIELDS = ["id", "label", "source_group", "group_id", "role", "duration_view", "native_sample_rate_hz"]


def row(identity: str, source: str, **updates: str) -> dict[str, str]:
    value = dict.fromkeys(FIELDS, "")
    value.update(id=identity, label="1", source_group=source, role="development",
                 group_id=f"group:{identity}", native_duration_s="30", available="1")
    value.update(updates)
    return value


class ScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_csv(self, name: str, rows: list[dict[str, str]], fields: list[str] = FIELDS) -> Path:
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        return path

    def write_freeze(self, additional: list[dict[str, str]], channels: dict[str, int] | None = None) -> Path:
        channels = channels or {}
        selected = []
        for metadata in additional:
            selected.append({
                "metadata": {key: metadata.get(key, "") for key in ADD_FIELDS},
                "native_channels": channels.get(metadata["id"], 2),
                "standardized_channels": 2, "standardized_frames": 2646000, "standardized_sr": 44100,
                "standardized_path": f"/synthetic/{metadata['id']}.wav",
                "standardized_file_sha256": "a" * 64,
            })
        path = self.root / "freeze.json"
        path.write_text(json.dumps({"status": "frozen_for_measurement_only", "selected": selected}), encoding="utf-8")
        return path

    def run_screen(self, master10, master30, additional=None, channels=None):
        additional = additional or []
        paths = [self.write_csv("m10.csv", master10), self.write_csv("m30.csv", master30),
                 self.write_csv("a60.csv", additional, ADD_FIELDS)]
        freeze = self.write_freeze(additional, channels)
        return S.screen(*paths, freeze, S.sha256_file(freeze), self.root / "out.json")

    def by_id(self, result, identity):
        return next(item for item in result["rows"] if item["id"] == identity)

    def test_transitive_cross_source_protection_and_same_prompt(self):
        a = row("a", "AI-A", condition_id="prompt-x")
        b = row("b", "AI-B", group_id="prompt-x", condition_id="prompt-y")
        protected = row("locked", "Human", label="0", role="locked", group_id="prompt-y")
        result = self.run_screen([a, b, protected], [a, b])
        for identity in ("a", "b"):
            item = self.by_id(result, identity)
            self.assertIn("component_touches_non_development", item["exclusion_reasons"])
            self.assertFalse(item["duration_exposure_candidate"])
        self.assertEqual(self.by_id(result, "a")["component_id"], self.by_id(result, "b")["component_id"])

    def test_raw_hash_links_protected_component(self):
        digest = "1" * 64
        candidate = row("candidate", "AI", raw_sha256=digest)
        locked = row("held", "Human", label="0", role="locked", raw_sha256=digest)
        result = self.run_screen([candidate, locked], [candidate])
        self.assertIn("component_touches_non_development", self.by_id(result, "candidate")["exclusion_reasons"])

    def test_missing_and_zero_duration_are_excluded(self):
        missing = row("missing", "AI", native_duration_s="", duration_view="")
        zero = row("zero", "AI", native_duration_s="0")
        result = self.run_screen([missing, zero], [missing, zero])
        self.assertIn("missing_required_metadata:duration", self.by_id(result, "missing")["exclusion_reasons"])
        self.assertIn("native_or_accepted_context_duration_lt_30", self.by_id(result, "zero")["exclusion_reasons"])

    def test_nan_duration_is_rejected(self):
        bad = row("bad", "AI", native_duration_s="NaN")
        m10 = self.write_csv("m10.csv", [bad]); m30 = self.write_csv("m30.csv", [bad])
        add = self.write_csv("a60.csv", [], ADD_FIELDS); freeze = self.write_freeze([])
        with self.assertRaisesRegex(ValueError, "non-finite native_duration_s"):
            S.screen(m10, m30, add, freeze, S.sha256_file(freeze), self.root / "out.json")

    def test_shared_id_conflict_is_rejected(self):
        first = row("same", "AI-A")
        second = row("same", "AI-B")
        with self.assertRaisesRegex(ValueError, "shared id conflict"):
            self.run_screen([first], [second])

    def test_duplicate_id_within_one_input_is_rejected(self):
        duplicate = row("same", "AI")
        master10 = self.write_csv("m10.csv", [duplicate, duplicate])
        master30 = self.write_csv("m30.csv", [])
        additional = self.write_csv("a60.csv", [], ADD_FIELDS)
        freeze = self.write_freeze([])
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate id"):
            S.screen(master10, master30, additional, freeze, S.sha256_file(freeze), self.root / "out.json")

    def test_malformed_raw_hash_is_rejected(self):
        malformed = row("bad-hash", "AI", raw_sha256="ABC")
        master10 = self.write_csv("m10.csv", [malformed])
        master30 = self.write_csv("m30.csv", [])
        additional = self.write_csv("a60.csv", [], ADD_FIELDS)
        freeze = self.write_freeze([])
        with self.assertRaisesRegex(ValueError, "malformed raw_sha256"):
            S.screen(master10, master30, additional, freeze, S.sha256_file(freeze), self.root / "out.json")

    def test_v6_duplicate_conflict_is_rejected(self):
        first = row("same", "AI-A")
        additional = [{key: first.get(key, "") for key in ADD_FIELDS}]
        additional[0]["source_group"] = "AI-B"
        additional[0]["duration_view"] = "60s"
        with self.assertRaisesRegex(ValueError, "shared id conflict"):
            self.run_screen([first], [first], additional)

    def test_mono_is_flagged_and_never_admitted_as_stereo(self):
        mono = row("mono", "AI", native_channels="1", standardized_channels="2")
        result = self.run_screen([mono], [mono])
        item = self.by_id(result, "mono")
        self.assertTrue(item["duration_exposure_candidate"])
        self.assertEqual(item["recorded_native_channel_status"], "mono")
        self.assertFalse(item["native_stereo_admitted"])

    def test_additional60_uses_bound_freeze_but_not_full_native_claim(self):
        additional = [{
            "id": "new", "label": "1", "source_group": "Mureka", "group_id": "m:1",
            "role": "development", "duration_view": "60s", "native_sample_rate_hz": "44100",
        }]
        result = self.run_screen([], [], additional)
        item = self.by_id(result, "new")
        self.assertTrue(item["duration_exposure_candidate"])
        self.assertEqual(item["duration_evidence"], "accepted_prior60_view")
        self.assertFalse(item["full_native_duration_known"])
        self.assertEqual(item["recorded_native_channel_status"], "stereo")
        self.assertFalse(result["feature_extraction_authorized"])
        self.assertEqual(result["classifier_fits"], 0)

    def test_available_and_evaluation_reasons_are_preserved(self):
        denied = row("denied", "AI", available="0", evaluation_allowed="no_until_verified",
                     original_role="provisional_development")
        result = self.run_screen([denied], [denied])
        reasons = self.by_id(result, "denied")["exclusion_reasons"]
        self.assertIn("available_not_1", reasons)
        self.assertIn("evaluation_not_allowed:master10:no_until_verified", reasons)
        self.assertIn("evaluation_not_allowed:master30:no_until_verified", reasons)
        self.assertIn("provisional_role:master10", reasons)
        self.assertIn("provisional_role:master30", reasons)

    def test_adverse_master10_occurrence_survives_master30_precedence(self):
        adverse = row("shared", "AI", evaluation_allowed="no_until_verified",
                      original_role="provisional_development")
        preferred = row("shared", "AI")
        result = self.run_screen([adverse], [preferred])
        reasons = self.by_id(result, "shared")["exclusion_reasons"]
        self.assertIn("evaluation_not_allowed:master10:no_until_verified", reasons)
        self.assertIn("provisional_role:master10", reasons)

    def test_untrusted_declared_duration_and_missing_available_are_excluded(self):
        declared = row("declared", "AI", native_duration_s="", duration_sec="60", available="")
        result = self.run_screen([declared], [declared])
        reasons = self.by_id(result, "declared")["exclusion_reasons"]
        self.assertEqual(self.by_id(result, "declared")["duration_context_s"], 60.0)
        self.assertEqual(self.by_id(result, "declared")["duration_evidence"], "declared_duration_sec_not_native")
        self.assertIn("unverified_duration_evidence", reasons)
        self.assertIn("missing_required_metadata:available", reasons)

    def test_missing_label_is_explicit_and_invalid_label_rejected(self):
        missing = row("missing-label", "AI", label="")
        result = self.run_screen([missing], [missing])
        self.assertIn("missing_required_metadata:label", self.by_id(result, "missing-label")["exclusion_reasons"])
        bad = row("bad-label", "AI", label="AI")
        m10 = self.write_csv("bad_m10.csv", [bad]); m30 = self.write_csv("bad_m30.csv", [])
        add = self.write_csv("bad_a60.csv", [], ADD_FIELDS); freeze = self.write_freeze([])
        with self.assertRaisesRegex(ValueError, "label must be exactly 0 or 1"):
            S.screen(m10, m30, add, freeze, S.sha256_file(freeze), self.root / "bad_out.json")

    def test_duplicate_csv_header_is_rejected(self):
        bad = self.root / "duplicate_header.csv"
        bad.write_text("id,label,label\nx,1,1\n", encoding="utf-8")
        m30 = self.write_csv("m30.csv", []); add = self.write_csv("a60.csv", [], ADD_FIELDS)
        freeze = self.write_freeze([])
        with self.assertRaisesRegex(ValueError, "duplicate CSV header"):
            S.screen(bad, m30, add, freeze, S.sha256_file(freeze), self.root / "out.json")

    def test_freeze_hash_status_exact_duration_and_strict_json(self):
        m10 = self.write_csv("m10.csv", []); m30 = self.write_csv("m30.csv", [])
        add = self.write_csv("a60.csv", [], ADD_FIELDS); freeze = self.write_freeze([])
        with self.assertRaisesRegex(ValueError, "does not match"):
            S.screen(m10, m30, add, freeze, "0" * 64, self.root / "wrong_hash.json")
        freeze.write_text('{"status":"draft","selected":[]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "status"):
            S.screen(m10, m30, add, freeze, S.sha256_file(freeze), self.root / "draft.json")
        freeze.write_text('{"status":"frozen_for_measurement_only","selected":[],"selected":[]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            S.screen(m10, m30, add, freeze, S.sha256_file(freeze), self.root / "duplicate.json")
        freeze.write_text('{"status":"frozen_for_measurement_only","selected":[],"x":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            S.screen(m10, m30, add, freeze, S.sha256_file(freeze), self.root / "nonfinite.json")

    def test_non_exact60_freeze_record_is_rejected(self):
        additional = [{
            "id": "new", "label": "1", "source_group": "Mureka", "group_id": "m:1",
            "role": "development", "duration_view": "60s", "native_sample_rate_hz": "44100",
        }]
        m10 = self.write_csv("m10.csv", []); m30 = self.write_csv("m30.csv", [])
        add = self.write_csv("a60.csv", additional, ADD_FIELDS); freeze = self.write_freeze(additional)
        payload = json.loads(freeze.read_text())
        payload["selected"][0]["standardized_frames"] -= 1
        freeze.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "framing differs"):
            S.screen(m10, m30, add, freeze, S.sha256_file(freeze), self.root / "out.json")

    def test_changed_bound_input_writes_no_output(self):
        candidate = row("candidate", "AI")
        m10 = self.write_csv("m10.csv", [candidate]); m30 = self.write_csv("m30.csv", [candidate])
        add = self.write_csv("a60.csv", [], ADD_FIELDS); freeze = self.write_freeze([])
        output = self.root / "out.json"
        original = S.load_freeze

        def mutate_after_parse(path, rows):
            result = original(path, rows)
            with m10.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            return result

        with mock.patch.object(S, "load_freeze", side_effect=mutate_after_parse):
            with self.assertRaisesRegex(ValueError, "changed during screen"):
                S.screen(m10, m30, add, freeze, S.sha256_file(freeze), output)
        self.assertFalse(output.exists())

    def test_refuses_overwrite_and_path_traversal(self):
        candidate = row("ok", "AI")
        result = self.run_screen([candidate], [candidate])
        self.assertEqual(result["counts"]["duration_exposure_candidates"], 1)
        with self.assertRaisesRegex(ValueError, "traversal-free"):
            S.safe_path(Path("../escape.csv"))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            S.safe_path(self.root / "out.json", output=True)

    def test_symlink_ancestor_is_rejected(self):
        real = self.root / "real"; real.mkdir()
        source = real / "data.csv"; source.write_text("id\n", encoding="utf-8")
        link = self.root / "link"; link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path component"):
            S.safe_path(link / "data.csv")


if __name__ == "__main__":
    unittest.main()
