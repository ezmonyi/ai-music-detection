import unittest
from audit_saraga_tempo_concordance_v1 import compare, parse


class ConcordanceTests(unittest.TestCase):
    def test_gap_not_exact_boundary(self):
        r=compare(b'60,0,60\n120,60,120\n', b'60,1,4,4,0,55\n120,.5,2,4,60,118\n')
        self.assertEqual(r['status'],'matched_tempo_and_starts')
        self.assertEqual(r['changes'][0]['detailed_unannotated_gap_s'],5)
        self.assertFalse(r['changes'][0]['detailed_adjacent'])
        self.assertTrue(r['changes'][0]['thirty_second_windows_inside_both_detailed_intervals'])
        self.assertEqual(r['changes'][0]['before_window_s'],[25,55])

    def test_exact(self):
        r=compare(b'60,0,60\n120,60,120\n', b'60,1,4,4,0,60\n120,.5,2,4,60,120\n')
        self.assertTrue(r['changes'][0]['detailed_adjacent'])

    def test_unknown_not_zero(self):
        r=compare(b'-,0,60\n120,60,120\n', b'-1,-1,-1,-1,0,60\n120,.5,2,4,60,120\n')
        self.assertEqual(r['changes'],[])
        self.assertIsNone(r['pairs'][0]['coarse']['tempo'])

    def test_mismatch_no_claim(self):
        for b in [b'61,1,4,4,0,60\n', b'60,1,4,4,1,60\n', b'60,1,4,4,0,60\n60,1,4,4,60,90\n']:
            r=compare(b'60,0,60\n',b)
            self.assertEqual(r['status'],'requires_manual_review')
            self.assertEqual(r['changes'],[])

    def test_invalid(self):
        for b in [b'60,0,nan',b'60,2,1',b'60,0,60\n120,30,70',b'60,1']:
            self.assertTrue(parse(b,3)[1])

    def test_short_context_is_unavailable(self):
        r=compare(b'60,0,20\n120,20,80\n',b'60,1,4,4,0,20\n120,.5,2,4,20,80\n')
        self.assertFalse(r['changes'][0]['thirty_second_windows_inside_both_detailed_intervals'])


if __name__=='__main__':
    unittest.main()
