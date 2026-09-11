import unittest
import numpy as np
from source_resampling_sensitivity import reweight


class SourceResamplingTests(unittest.TestCase):
    def test_uniform_equals_macro(self):
        matrix = np.arange(2*6*4*3*2, dtype=float).reshape(2,6,4,3,2)
        np.testing.assert_allclose(reweight(matrix,np.ones(6),np.ones(4)),matrix.mean(axis=(0,1,2)))

    def test_one_source_pair_uses_both_directions(self):
        matrix = np.arange(2*6*4*3*2, dtype=float).reshape(2,6,4,3,2)
        np.testing.assert_allclose(reweight(matrix,np.array([0,6,0,0,0,0]),np.array([0,0,4,0])),matrix[:,1,2].mean(axis=0))

    def test_invalid(self):
        matrix = np.zeros((2,6,4,3,2))
        with self.assertRaises(ValueError):
            reweight(matrix,np.zeros(6),np.ones(4))
        with self.assertRaises(ValueError):
            reweight(matrix,-np.ones(6),np.ones(4))


if __name__ == '__main__':
    unittest.main()
