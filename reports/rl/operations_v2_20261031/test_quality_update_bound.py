import unittest
from quality_upper_bound_watch import upper_bound


class FeasibilityProofTests(unittest.TestCase):
    def test_overlapping_invalid_proofs_cannot_double_count_groups(self):
        self.assertEqual(upper_bound(range(50),range(1,51)),999)
        self.assertEqual(upper_bound(range(50),range(50)),1000)

    def test_threshold_is_conservative_even_if_every_other_group_passes(self):
        self.assertEqual(upper_bound(range(51),()),999)
        self.assertEqual(upper_bound((),()),1050)


if __name__=='__main__':unittest.main()
