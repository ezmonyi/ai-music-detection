import unittest

from acquire_vocalset_breath_v2 import choose_candidate


class VocalSetV2ResolutionTest(unittest.TestCase):
    def test_identical_hashes_are_canonicalized_deterministically(self):
        records = [
            {"archive_member": "z/file.wav", "audio_sha256": "same", "duration_sec": 1.0},
            {"archive_member": "a/file.wav", "audio_sha256": "same", "duration_sec": 2.0},
        ]
        selected, method = choose_candidate(records, 1.5)
        self.assertEqual(selected["archive_member"], "a/file.wav")
        self.assertEqual(method, "byte_identical_full_sha256")

    def test_unique_duration_match_selects_one_different_member(self):
        records = [
            {"archive_member": "male8/file.wav", "audio_sha256": "a", "duration_sec": 9.0},
            {"archive_member": "male9/file.wav", "audio_sha256": "b", "duration_sec": 10.001},
        ]
        selected, method = choose_candidate(records, 10.0)
        self.assertEqual(selected["archive_member"], "male9/file.wav")
        self.assertEqual(method, "unique_annotation_duration_match")

    def test_different_members_without_unique_duration_are_unresolved(self):
        records = [
            {"archive_member": "a/file.wav", "audio_sha256": "a", "duration_sec": 10.0},
            {"archive_member": "b/file.wav", "audio_sha256": "b", "duration_sec": 10.005},
        ]
        with self.assertRaises(ValueError):
            choose_candidate(records, 10.0)


if __name__ == "__main__":
    unittest.main()
