import copy
import unittest

from plot_bc_guitarset_bifrequency_v1 import CONDITIONS,GRID,cell_summary,panels


def fixture():
    records=[]
    for player in range(3):
        for score in range(15):
            for performance in ('comp','solo'):
                conditions={}
                for name in CONDITIONS:
                    value=.2 if name in ('baseline','common_gain','polarity') else .8 if name.startswith('closed') else .1
                    conditions[name]={'pools':[{'pool_index':pool,'grid_squared_bicoherence':[value]*228,
                                                'grid_eligibility':[True]*228} for pool in range(2)]}
                records.append({'item_id':f'{player}_{score}_{performance}','score_id':str(score),
                                'player_id':str(player),'split_role':'development','conditions':conditions})
    return {'per_recording':records,'baseline':{'target_covered_pools':180,'grid_eligible_cells':41040},
            'levels':{level:{'equal_score':{'equal_group_mean_difference':.7},
                             'covered_recordings':90,'paired_covered_pools':180} for level in ('minus6db','0db')}}


class PlotTests(unittest.TestCase):
    def test_pool_record_score_weighting_and_missingness(self):
        result=cell_summary([[1.,3.],[None,None],[8.,None],[0.,None]],['a','a','b','b'])
        self.assertEqual(result,{'value':3.,'covered_pools':4,'pool_denominator':8,
                                'covered_recordings':3,'recording_denominator':4,
                                'covered_scores':2,'score_denominator':2})

    def test_zero_not_missing_and_empty_group_remains_denominator(self):
        result=cell_summary([[None,None],[0.,None]],['a','b'])
        self.assertEqual(result['value'],0.);self.assertEqual(result['covered_scores'],1)
        self.assertEqual(result['score_denominator'],2)
        self.assertIsNone(cell_summary([[None,None]],['a'])['value'])

    def test_grid_and_all_panels(self):
        self.assertEqual(len(GRID),228);self.assertEqual(len(set(GRID)),228)
        result=panels(fixture())
        for key,cells in result.items():
            self.assertEqual(len(cells),228)
            self.assertAlmostEqual(cells[0]['value'],.2 if key=='baseline' else .7)

    def test_pair_pools_before_aggregating(self):
        data=fixture(); record=data['per_recording'][0]
        for level in ('minus6db','0db'):
            for kind,pool in [('closed',0),('independent',1)]:
                p=record['conditions'][kind+'_'+level]['pools'][pool]
                p['grid_squared_bicoherence']=[None]*228;p['grid_eligibility']=[False]*228
            data['levels'][level]['covered_recordings']=89
            data['levels'][level]['paired_covered_pools']=178
        result=panels(data)
        self.assertEqual(result['minus6db'][0]['covered_recordings'],89)
        self.assertEqual(result['minus6db'][0]['covered_scores'],15)
        self.assertAlmostEqual(result['minus6db'][0]['value'],.7)

    def test_corruption_and_reserved_rows_fail(self):
        data=fixture();data['per_recording'][0]['split_role']='reserved'
        with self.assertRaises(ValueError):panels(data)
        data=fixture();data['per_recording'][0]['conditions']['baseline']['pools'][0]['grid_eligibility'][0]=False
        with self.assertRaises(ValueError):panels(data)
        data=fixture();data['levels']['0db']['equal_score']['equal_group_mean_difference']=.6
        with self.assertRaises(ValueError):panels(data)

    def test_bad_aggregate_arguments(self):
        for values,scores in [([[1.,2.]],[]),([[1.]],['a']),([[float('nan'),None]],['a']),([[True,None]],['a'])]:
            with self.assertRaises(ValueError):cell_summary(values,scores)


if __name__=='__main__':
    unittest.main()
