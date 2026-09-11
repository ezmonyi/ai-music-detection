"""Synthetic-only weighting compatibility; no corpus data or classifier fits."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).with_name('frozen_evaluate_expanded_20260905.py')
PIN = 'd2ed30d9833fbe122f023de1223e44a40c1b4f95f99f63f0ba27647c87cc7232'
if hashlib.sha256(PATH.read_bytes()).hexdigest() != PIN:
    raise ValueError('Frozen weighting implementation changed')
SPEC = importlib.util.spec_from_file_location('native30_weighting_reference', PATH)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def fixture():
    # The shared ID is paired across both labels and across two AI sources.
    return pd.DataFrame([
        ('h1', 0, 'human_a', 'shared'), ('h2', 0, 'human_a', 'shared'),
        ('h3', 0, 'human_a', 'other'), ('h4', 0, 'human_b', 'solo_h'),
        ('a1', 1, 'ai_a', 'shared'), ('a2', 1, 'ai_a', 'solo_a'),
        ('a3', 1, 'ai_b', 'shared'), ('a4', 1, 'ai_b', 'shared'),
        ('a5', 1, 'ai_b', 'other_ai')],
        columns=['__id', '__label', '__source', '__group'])


class SharedGroupWeights(unittest.TestCase):
    def test_hierarchical_mass_with_crosslabel_crosssource_groups(self):
        table = fixture()
        weights = BASE.sample_weights(table)
        self.assertTrue(np.isfinite(weights).all() and (weights > 0).all())
        self.assertAlmostEqual(weights.sum(), len(table))
        for label in (0, 1):
            indices = np.flatnonzero(table['__label'].to_numpy() == label)
            self.assertAlmostEqual(weights[indices].sum(), len(table) / 2)
            for source in table.iloc[indices]['__source'].unique():
                chosen = table['__source'].to_numpy() == source
                self.assertAlmostEqual(weights[chosen].sum(), len(table) / 4)
                groups = table.loc[chosen, '__group'].unique()
                for group in groups:
                    mask = chosen & (table['__group'].to_numpy() == group)
                    self.assertAlmostEqual(weights[mask].sum(), len(table) / 4 / len(groups))

    def test_permutation_equivariance(self):
        table = fixture()
        expected = dict(zip(table['__id'], BASE.sample_weights(table)))
        permuted = table.iloc[[8, 2, 0, 5, 4, 1, 7, 3, 6]].copy()
        actual = dict(zip(permuted['__id'], BASE.sample_weights(permuted)))
        for ident in expected:
            self.assertAlmostEqual(expected[ident], actual[ident])

    def test_shared_name_does_not_merge_source_conditioned_weight_cells(self):
        table = fixture()
        renamed = table.copy()
        renamed['__group'] = renamed['__source'] + ':' + renamed['__group']
        np.testing.assert_allclose(BASE.sample_weights(table), BASE.sample_weights(renamed), rtol=0, atol=0)
        # This is only a weighting identity. Renaming must NOT be used for splits.

    def test_class_absence_is_not_silently_weighted(self):
        table = fixture()
        with self.assertRaisesRegex(ValueError, 'both classes'):
            BASE.sample_weights(table[table['__label'] == 1])


if __name__ == '__main__':
    unittest.main()
