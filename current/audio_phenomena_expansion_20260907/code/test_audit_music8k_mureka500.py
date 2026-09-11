import unittest
from audit_music8k_mureka500 import check_decode,rank,text_hash

class AcquisitionAuditTests(unittest.TestCase):
    def base(self):
        return dict(status='passed',all_samples_finite=True,sample_rate=44100,channels=2,
            decoded_frames=44100*60,decoded_duration_s=60.,eligible_native60=True,
            decoded_native_float32_sha256='a'*64)
    def test_exact60(self):self.assertTrue(check_decode(self.base()))
    def test_short(self):
        r=self.base();r.update(decoded_frames=44100*59,decoded_duration_s=59.,eligible_native60=False)
        self.assertFalse(check_decode(r))
    def test_inconsistent_receipt(self):
        for key,value in [('decoded_duration_s',61.),('all_samples_finite',False),('decoded_frames',2646000.),('eligible_native60',False),('decoded_native_float32_sha256','x'*64)]:
            r=self.base();r[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):check_decode(r)
    def test_ranking_order_independence(self):
        self.assertEqual(rank(['1','2','3'],'seed'),rank(['3','2','1'],'seed'))
    def test_normalization(self):self.assertEqual(text_hash(' ＡＢＣ  hi '),text_hash('abc hi'))

if __name__=='__main__':unittest.main()
