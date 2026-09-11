import unittest
from audit_saraga_phrase_annotations_v1 import parse_phrases


class PhraseTests(unittest.TestCase):
    def test_flagzero_and_mixed_flags_are_preserved(self):
        r=parse_phrases(b'0\t0\t5\tDnP\n16\t1\t5\tDnP')
        self.assertEqual(r['flag_counts'],{'0':1,'1':1})
        self.assertEqual(r['note_labels_with_mixed_flags'],['DnP'])
        self.assertFalse(r['same_label_pairs_lag8to56s'][0]['both_flags_1_or_2'])

    def test_note_case_is_not_collapsed(self):
        r=parse_phrases(b'0 1 5 DnP\n16 1 5 dnp')
        self.assertEqual(r['same_label_pairs_lag8to56s'],[])

    def test_duration_and_lag_screens_not_groundtruth(self):
        r=parse_phrases(b'0 1 5 DnP\n16 2 3 DnP\n80 1 5 DnP')
        self.assertEqual(len(r['same_label_pairs_lag8to56s']),1)
        self.assertFalse(r['same_label_pairs_lag8to56s'][0]['both_duration_at_least4s'])

    def test_invalid_and_unknown_flags_are_visible(self):
        r=parse_phrases(b'nan 1 3 DnP\n0 3 5 DnP')
        self.assertEqual(r['status'],'invalid_annotation')
        self.assertEqual(r['unknown_flag_values'],[3])


if __name__=='__main__': unittest.main()
