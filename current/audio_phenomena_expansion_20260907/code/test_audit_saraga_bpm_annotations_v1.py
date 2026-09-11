import unittest
from audit_saraga_bpm_annotations_v1 import parse_bpm


class AnnotationTests(unittest.TestCase):
    def test_unknown_tempo_is_not_zero(self):
        r = parse_bpm(b'-,0,70\n54,70,400\n194,400,500\n')
        self.assertIsNone(r['segments'][0]['bpm'])
        self.assertEqual(len(r['contiguous_known_bpm_changes']), 1)
        self.assertEqual(r['contiguous_known_bpm_changes'][0]['time_s'], 400)

    def test_overlap_invalidates_change_labels(self):
        r = parse_bpm(b'60,0,100\n120,90,200\n')
        self.assertEqual(r['status'], 'invalid_annotation')
        self.assertEqual(r['contiguous_known_bpm_changes'], [])

    def test_gap_is_not_a_known_boundary(self):
        self.assertEqual(parse_bpm(b'60,0,100\n120,120,200')['contiguous_known_bpm_changes'], [])

    def test_double_tempo_remains_flagged_not_deleted(self):
        t = parse_bpm(b'60,0,100\n120,100,200')['contiguous_known_bpm_changes'][0]
        self.assertTrue(t['near_half_or_double_within5pct'])
        self.assertTrue(t['context_30s_both_sides_within_adjacent_annotations'])

    def test_nonfinite_or_extra_column_rejected(self):
        for text in (b'nan,0,10', b'60,0,inf', b'60,0,10,20'):
            self.assertEqual(parse_bpm(text)['status'], 'invalid_annotation')


if __name__ == '__main__':
    unittest.main()
