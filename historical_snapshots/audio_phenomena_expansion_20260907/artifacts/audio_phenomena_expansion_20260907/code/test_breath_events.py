import unittest
import numpy as np
from benchmark_breath_events import targets, predicted_events, match_events, event_metrics


class BreathEventsTest(unittest.TestCase):
    def test_silence_and_low_confidence_are_not_positives(self):
        times = np.arange(101)*.01
        label = {"breath_events":[{"start_sec":.1,"end_sec":.3,"confidence":"high"},
                                   {"start_sec":.4,"end_sec":.5,"confidence":"low"}],
                 "silent_breaths":[{"start_sec":.6,"end_sec":.8}]}
        y,valid,events = targets(times,label)
        self.assertEqual(events,[(.1,.3)])
        self.assertTrue((y[(times>=.6)&(times<=.8)]==0).all())
        self.assertFalse(valid[(times>=.4)&(times<=.5)].any())
        self.assertFalse(valid[(times>=.6)&(times<=.8)].any())

    def test_matching_is_one_to_one(self):
        self.assertEqual(match_events([(1.,1.4)],[(1.,1.2),(1.05,1.3)]),(1,1,0))
        self.assertEqual(match_events([(1.,1.2)],[(1.2,1.4)]),(0,1,1))

    def test_far_onsets_do_not_match_even_with_overlap(self):
        self.assertEqual(match_events([(1.,2.)],[(1.5,1.8)]),(0,1,1))

    def test_merges_short_gaps_but_not_ignored_regions(self):
        times = np.arange(40)*.01
        scores = np.zeros(40); scores[5:12]=1; scores[14:21]=1
        valid = np.ones(40,dtype=bool)
        self.assertEqual(len(predicted_events(scores,valid,times)),1)
        valid[12:14]=False
        self.assertEqual(len(predicted_events(scores,valid,times)),2)

    def test_no_event_metrics_and_duration_gate(self):
        self.assertIsNone(event_metrics(0,0,0)["recall"])
        self.assertEqual(event_metrics(0,1,0)["f1"],0)
        times=np.arange(300)*.01; valid=np.ones(300,dtype=bool)
        long = predicted_events(np.ones(300),valid,times)
        self.assertEqual(len(long),1)
        self.assertEqual(match_events([(.1,.5)],long),(0,1,1))
        scores=np.zeros(300); scores[10:13]=1
        self.assertEqual(predicted_events(scores,valid,times),[])

    def test_overlapping_reference_events_are_one_inhale(self):
        times=np.arange(200)*.01
        label={"breath_events":[{"start_sec":.3,"end_sec":.8,"confidence":"high"},
                                {"start_sec":.5,"end_sec":.7,"confidence":"medium"}]}
        y,valid,events=targets(times,label)
        self.assertEqual(events,[(.3,.8)])
        self.assertTrue(valid[50])


if __name__ == "__main__":
    unittest.main()
