import unittest
from pathlib import Path
import tempfile
import hashlib
import materialize_music8k_mureka500 as m


class AcquisitionTests(unittest.TestCase):
    def test_seeded_selection_excludes_probes(self):
        candidates=[{'id':str(i)} for i in range(662)]
        excluded={str(i) for i in range(12)}
        a=m.formal_rank(candidates,excluded)
        b=m.formal_rank(list(reversed(candidates)),excluded)
        self.assertEqual(a,b)
        self.assertEqual(len(a[:500]),500)
        self.assertFalse({r['id'] for r in a}&excluded)

    def test_missing_probe_or_duplicate_rejected(self):
        candidates=[{'id':str(i)} for i in range(662)]
        with self.assertRaises(ValueError):
            m.formal_rank(candidates,{str(i) for i in range(11)})
        candidates[-1]={'id':'100'}
        with self.assertRaises(ValueError):
            m.formal_rank(candidates,{str(i) for i in range(12)})

    def test_normalization_hash(self):
        self.assertEqual(m.text_hash(' Ａ  B\n'),m.text_hash('a b'))
        self.assertNotEqual(m.text_hash('a b'),m.text_hash('a c'))

    def test_raw_hash_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.mp3'
            path.write_bytes(b'abc')
            row={'id':'1','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
            m.verify_raw(path,row)
            path.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                m.verify_raw(path,row)


if __name__=='__main__':
    unittest.main()
