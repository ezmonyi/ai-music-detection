import copy
import unittest
from native60_score_kernel_v1 import score, positive_summary


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.model = dict(columns=['a', 'b'], medians=[2., 4.],
                          mean=[1., 2., 0.25, 0.5], scale=[2., 4., 0.5, 1.],
                          coefficients_with_intercept=[0.1, 0.2, -0.3, 0.4, 0.6],
                          threshold=0.5, feature_mode='values_plus_missing')

    def test_observed_and_input_unchanged(self):
        original = copy.deepcopy(self.model)
        value, label = score({'a':3., 'b':6.}, self.model)
        self.assertAlmostEqual(value, -0.5)
        self.assertEqual(label, 0)
        self.assertEqual(self.model, original)

    def test_missing_indicators_not_just_imputation(self):
        self.assertAlmostEqual(score({'a':None, 'b':float('nan')}, self.model)[0], .95)

    def test_threshold_equality_and_no_clipping(self):
        self.model['coefficients_with_intercept'] = [.5, 0., 0., 0., 0.]
        self.assertEqual(score({'a':0., 'b':0.}, self.model), (.5, 1))
        self.model['coefficients_with_intercept'][0] = 2.
        self.assertEqual(score({'a':0., 'b':0.}, self.model), (2., 1))

    def test_absent_column_and_inf_rejected(self):
        with self.assertRaises(KeyError): score({'a':0.}, self.model)
        with self.assertRaises(ValueError): score({'a':float('inf'), 'b':0.}, self.model)

    def test_invalid_parameters(self):
        for field, value in [('scale', [0., 1., 1., 1.]), ('threshold', 0.),
                             ('columns', ['a', 'a']), ('medians', [0.])]:
            model = copy.deepcopy(self.model)
            model[field] = value
            with self.assertRaises(ValueError): score({'a':0., 'b':0.}, model)

    def test_single_class_scope(self):
        result = positive_summary([1, 0, 1, 1])
        self.assertEqual(result['ai_sensitivity'], .75)
        self.assertIsNone(result['balanced_accuracy'])
        self.assertIsNone(result['human_specificity'])
        self.assertIsNone(result['roc_auc'])
        with self.assertRaises(ValueError): positive_summary([])


if __name__ == '__main__':
    unittest.main()
