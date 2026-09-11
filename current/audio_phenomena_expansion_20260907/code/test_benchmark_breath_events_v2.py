import json
from pathlib import Path
import tempfile
import unittest

from benchmark_breath_events_v2 import select_primary_cohort


def write_label(root, name, labeler, events):
    path = root / "labels" / "vocalset" / (Path(name).stem + ".breath.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "audio_file": name,
        "labeler": labeler,
        "review_time_sec": 10,
        "breath_events": [
            {"start_sec": start, "end_sec": end, "confidence": confidence}
            for start, end, confidence in events
        ],
    }))


class BreathBenchmarkV2CohortTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.labels = Path(self.temporary.name)
        self.manifest = [
            {"filename": "f1_caro_straight.wav", "singer": "f1", "archive_member": "FULL/female1/excerpts/straight/f1_caro_straight.wav"},
            {"filename": "m9_caro_vibrato.wav", "singer": "m9", "archive_member": "FULL/male8/excerpts/vibrato/m9_caro_vibrato.wav"},
            {"filename": "m1_scales_straight_a.wav", "singer": "m1", "archive_member": "FULL/male1/scales/straight/m1_scales_straight_a.wav"},
        ]
        write_label(self.labels, "f1_caro_straight.wav", "festus", [(1., 2., "high")])
        write_label(self.labels, "m9_caro_vibrato.wav", "festus", [(1., 2., "high"), (1.5, 2.5, "medium")])
        write_label(self.labels, "m1_scales_straight_a.wav", "", [])
        self.expected_counts = {
            "pre_identity_primary_clips": 2,
            "pre_identity_raw_high_medium_events": 3,
            "pre_identity_merged_high_medium_events": 2,
            "pre_identity_zero_event_clips": 0,
            "eligible_primary_clips": 1,
            "eligible_raw_high_medium_events": 1,
            "eligible_merged_high_medium_events": 1,
            "eligible_zero_event_clips": 0,
            "eligible_singers": 1,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_conflicting_primary_is_excluded_before_fit(self):
        selected, excluded, audit = select_primary_cohort(
            self.manifest,
            self.labels,
            expected_conflicts={"m9_caro_vibrato.wav"},
            expected_counts=self.expected_counts,
        )
        self.assertEqual([row["filename"] for row in selected], ["f1_caro_straight.wav"])
        self.assertTrue(audit["conflict_excluded_before_any_feature_fit"])
        self.assertEqual(audit["counts"]["eligible_primary_clips"], 1)
        self.assertEqual(len(excluded), 2)

    def test_unexpected_additional_conflict_aborts(self):
        self.manifest[0]["archive_member"] = "FULL/female2/excerpts/straight/f1_caro_straight.wav"
        with self.assertRaisesRegex(ValueError, "conflict set changed"):
            select_primary_cohort(
                self.manifest,
                self.labels,
                expected_conflicts={"m9_caro_vibrato.wav"},
                expected_counts=self.expected_counts,
            )

    def test_touching_events_are_not_merged(self):
        write_label(
            self.labels,
            "f1_caro_straight.wav",
            "festus",
            [(1., 2., "high"), (2., 3., "medium")],
        )
        changed = dict(self.expected_counts)
        changed["pre_identity_raw_high_medium_events"] = 4
        changed["pre_identity_merged_high_medium_events"] = 3
        changed["eligible_raw_high_medium_events"] = 2
        changed["eligible_merged_high_medium_events"] = 2
        _, _, audit = select_primary_cohort(
            self.manifest,
            self.labels,
            expected_conflicts={"m9_caro_vibrato.wav"},
            expected_counts=changed,
        )
        self.assertEqual(audit["counts"]["eligible_merged_high_medium_events"], 2)


if __name__ == "__main__":
    unittest.main()
