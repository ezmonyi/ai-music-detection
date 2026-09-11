import unittest
from plot_bc_nsynth_bifrequency_v1 import GRID,balanced,panels

class AggregationTests(unittest.TestCase):
    def test_missing_is_not_zero_and_groups_are_equal(self):
        r=balanced([1.,3.,None,8.,None],['a','a','b','b','c'])
        self.assertEqual(r,{'value':5.,'covered_notes':3,'covered_instruments':2,
                            'note_denominator':5,'instrument_denominator':3})
    def test_missing_and_zero_distinct(self):
        self.assertIsNone(balanced([None],['a'])['value'])
        self.assertEqual(balanced([0.],['a'])['value'],0.)
    def test_grid_target_bounds_and_uniqueness(self):
        self.assertEqual(len(GRID),228)
        self.assertEqual(len(set(GRID)),228)
        self.assertIn((32,48,80),GRID)
        self.assertTrue(all(a<=b and a+b==c<=256 for a,b,c in GRID))
    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError): balanced([1.],[])
        with self.assertRaises(ValueError): balanced([float('nan')],['a'])
    def test_paired_missing_is_excluded_before_grouping(self):
        notes=[]
        for index in range(54):
            conditions={}
            for name in ('baseline','closed_minus6db','independent_minus6db','closed_0db','independent_0db'):
                value=.5 if name=='baseline' else .8 if name.startswith('closed') else .1
                if (index==0 and name.startswith('closed')) or (index==1 and name.startswith('independent')):
                    value=None
                conditions[name]={'grid_squared_bicoherence':[value]*228,
                                  'grid_eligibility':[value is not None]*228}
            notes.append({'instrument':str(index//2),'conditions':conditions})
        summary={'per_note':notes,'levels':{k:{'equal_instrument_mean_difference':.7,'covered_notes':52}
                                           for k in ('minus6db','0db')}}
        result=panels(summary)
        self.assertEqual(result['baseline'][0]['covered_notes'],54)
        self.assertEqual(result['minus6db'][0]['covered_notes'],52)
        self.assertEqual(result['minus6db'][0]['covered_instruments'],26)
        self.assertAlmostEqual(result['minus6db'][0]['value'],.7)
        summary['per_note'][0]['conditions']['baseline']['grid_eligibility'][0]=False
        with self.assertRaises(ValueError): panels(summary)

if __name__=='__main__':
    unittest.main()
