import copy
import json
from pathlib import Path
import unittest
from music_detector.rl.reviewed_receiver import check_options


class ReviewedReceiverOptionsTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((Path(__file__).parents[1]/'configs/rl/offline_1000.json').read_text())
        self.options={**self.base,'replay_tolerance':.0003,'replay_steps':31}

    def test_original_defaults_stay_strict(self):
        check_options(self.options,self.base)
        self.assertEqual(self.base['replay_tolerance'],1e-6)

    def test_other_guards_cannot_be_relaxed(self):
        for key,value in (('max_epochs',2),('max_behavior_kl',1.0),('max_consecutive_skips',1000),('train_timesteps',1),('replay_tolerance',.001)):
            bad=copy.deepcopy(self.options);bad[key]=value
            with self.assertRaises(ValueError):check_options(bad,self.base)

    def test_original_configuration_cannot_be_relabelled_as_reviewed(self):
        with self.assertRaises(ValueError):check_options(self.base,self.base)

    def test_modified_original_file_cannot_authorize_other_changes(self):
        original=copy.deepcopy(self.base);original['max_epochs']=2
        options={**original,'replay_tolerance':.0003,'replay_steps':31}
        with self.assertRaises(ValueError):check_options(options,original)


if __name__=='__main__':unittest.main()
