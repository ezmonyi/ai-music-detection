import unittest

from audit_breath_results import maximum_event_matches


class BreathResultAuditTest(unittest.TestCase):
    def test_independent_matcher_is_one_to_one(self):
        self.assertEqual(
            maximum_event_matches([(1., 1.4)], [(1., 1.2), (1.05, 1.3)]),
            (1, 1, 0),
        )

    def test_far_onset_and_overlong_predictions_do_not_match(self):
        self.assertEqual(maximum_event_matches([(1., 2.)], [(1.5, 1.8)]), (0, 1, 1))
        self.assertEqual(maximum_event_matches([(1., 1.4)], [(1., 3.1)]), (0, 1, 1))

    def test_bipartite_augmenting_path_finds_maximum(self):
        reference = [(1.0, 1.5), (1.19, 1.7)]
        prediction = [(1.18, 1.4), (0.81, 1.2)]
        self.assertEqual(maximum_event_matches(reference, prediction), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
