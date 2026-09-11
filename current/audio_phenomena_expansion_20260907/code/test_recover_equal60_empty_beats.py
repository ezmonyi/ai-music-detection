import unittest
from recover_equal60_empty_beats import validate_failure_arrays

class EmptyBeatRecoveryTests(unittest.TestCase):
    def test_exact_reviewed_case(self):
        validate_failure_arrays([], [0.0])

    def test_no_accept_nonempty_beats(self):
        with self.assertRaises(ValueError): validate_failure_arrays([1.0], [0.0])

    def test_no_accept_empty_both(self):
        with self.assertRaises(ValueError): validate_failure_arrays([], [])

    def test_invalid_downbeats(self):
        for values in ([float('nan')],[-1],[61],[1,1],[2,1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_failure_arrays([],values)

if __name__ == '__main__': unittest.main()
