import unittest
from audit_saraga_tempo_bounds_v1 import join


class BoundsTests(unittest.TestCase):
    def fixture(self):
        s={'records':[{'item':{'mbid':'a','reconciled_metadata':{
            'speech_title_review_flag':False,'performer_credits':[]}},
            'measurement':{'actual_duration_seconds':120}}]}
        t={'rows':[{'mbid':'a','changes':[{'before_window_s':[0,30],'after_window_s':[90,120]}]}]}
        return s,t

    def test_bounds_and_missing_not_admission(self):
        r=join(*self.fixture())
        self.assertEqual(r['physical_bounds_pass'],1)
        self.assertEqual(r['missing_performer_pairs'],1)
        self.assertFalse(r['cohort_selected'])

    def test_out_of_bounds_retained(self):
        s,t=self.fixture(); t['rows'][0]['changes'][0]['after_window_s']=[91,121]
        r=join(s,t)
        self.assertEqual(r['candidate_pairs'],1)
        self.assertEqual(r['physical_bounds_pass'],0)

    def test_duplicate_physical_fatal(self):
        s,t=self.fixture(); s['records']*=2
        with self.assertRaises(ValueError):join(s,t)

    def test_missing_identity_fatal(self):
        s,t=self.fixture(); t['rows'][0]['mbid']='b'
        with self.assertRaises(KeyError):join(s,t)


if __name__=='__main__':unittest.main()
