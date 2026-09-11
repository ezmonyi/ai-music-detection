import unittest
from audit_music8k_metadata_overlap import norm, meaningful, digest, containment, screen


class OverlapTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(norm(' ＡＢＣ\n  Test '), 'abc test')
        self.assertEqual(digest(' ＡＢＣ\n  Test '), digest('abc test'))

    def test_missing_not_matches(self):
        self.assertFalse(meaningful(None))
        self.assertEqual(screen([('a','x','unknown')],[('b','x','unknown')]), [])

    def test_containment_length_and_direction(self):
        self.assertFalse(containment('short phrase','a short phrase'))
        self.assertTrue(containment('x'*32,'before '+'x'*32+' after'))
        self.assertTrue(containment('before '+'x'*32+' after','x'*32))

    def test_no_semantic_claim(self):
        self.assertEqual(screen([('a','x','a happy song')],[('b','x','a joyful song')]), [])

    def test_output_no_text_and_retains_ids(self):
        result=screen([('a','lyrics','A'*40)],[('b','lyrics','prefix '+'a'*40)],True)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['existing_id'],'b')
        self.assertEqual(result[0]['match'],'literal_containment_min32')
        self.assertNotIn('A'*40,str(result))


if __name__ == '__main__':
    unittest.main()
