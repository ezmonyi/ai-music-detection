"""Test exact URL screening; not a legal opinion about underlying rights."""
import unittest
from plan_fma_test_view_rights_v1 import is_derivative_candidate_license as allowed


class LicenseTests(unittest.TestCase):
    def test_nd_is_excluded(self):
        for kind in ['by-nd', 'by-nc-nd']:
            for version in ['1.0','2.0','2.5','3.0','4.0']:
                for port in ['', 'us/']:
                    self.assertFalse(allowed(f'https://creativecommons.org/licenses/{kind}/{version}/{port}'))

    def test_recorded_ported_and_unported_terms(self):
        self.assertTrue(allowed('https://creativecommons.org/licenses/by-nc-sa/3.0/us/'))
        self.assertTrue(allowed('https://creativecommons.org/licenses/by/4.0/'))

    def test_cc0(self):
        self.assertTrue(allowed('https://creativecommons.org/publicdomain/zero/1.0/'))

    def test_unknown_and_lookalike_terms(self):
        for url in ['', 'MIT', 'https://creativecommons.org.evil/licenses/by/4.0/',
                    'https://creativecommons.org/licenses/by/9.0/',
                    'https://creativecommons.org/licenses/by/4.0/?unknown=1']:
            self.assertFalse(allowed(url))


if __name__ == '__main__': unittest.main()
