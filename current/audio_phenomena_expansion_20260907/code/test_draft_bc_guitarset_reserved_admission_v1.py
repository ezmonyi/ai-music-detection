import importlib.util
import json
import os
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
ARTIFACT = HERE.parent
SPEC = importlib.util.spec_from_file_location('reserved_draft_under_test',
    HERE / 'draft_bc_guitarset_reserved_admission_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ReservedAdmissionDraftTests(unittest.TestCase):
    def test_frozen_source_and_accepted_development_pins(self):
        self.assertEqual(m.SOURCE_DRAFT_SHA,
            '7632e91ddb3b2c25e0ebebf9b10357b7c1ddfaa96d4e79c603b28f5ef4ee97eb')
        self.assertEqual(m.SOURCE_COMMIT_SHA,
            'aa33f0dfc55e3bbc6f48e618d1203c29e258608c309412d333f06c6e671b73dc')
        self.assertEqual(m.DEVELOPMENT_COMMIT_SHA,
            'cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c')
        self.assertEqual(m.CODEC_DEVELOPMENT_COMMIT_SHA,
            '9b38fb43457dbd3da03e46f12ff3e24e68f58e03e02eea6d6bfdd8d74fdc378b')
        self.assertEqual(m.CODEC_DEVELOPMENT_AUDIT_SHA,
            'aca0a828ed85f093158f532e5fb150217fa800f97e9bfe2e21cfb96ed85e77b3')

    def test_crop_plan_is_exact_centered_8s_no_tail_or_padding(self):
        plan = m.crop_plan(863725, 44100)
        self.assertEqual((plan['up'], plan['down']), (160, 441))
        self.assertEqual(plan['expected_resampled_frames'], (863725 * 160 + 440) // 441)
        self.assertEqual(plan['crop_stop_sample_exclusive'] - plan['crop_start_sample'], 128000)
        self.assertEqual(plan['complete_pool_count'], 2)
        self.assertEqual(plan['measurement_tail_discarded_samples'], 0)
        self.assertEqual(plan['padding_samples'], 0)
        self.assertLessEqual(abs(plan['leading_context_excluded_samples']
                                 - plan['trailing_context_excluded_samples']), 1)

    def test_exact_condition_and_measurement_denominators(self):
        self.assertEqual(len(m.CONDITIONS), 7)
        self.assertEqual(m.CODEC_CONDITIONS,
                         ('baseline', 'closed_minus6db', 'independent_minus6db'))
        self.assertEqual(m.expected_denominators(), {
            'reserved_recordings': 90, 'float64_conditions': 630,
            'float64_condition_pools': 1260, 'selected_float32_precision_controls': 270,
            'precision_control_pools': 540, 'codec_encodes': 540, 'codec_decodes': 540,
            'codec_decoded_BC_measurements': 540, 'codec_decoded_pools': 1080,
            'total_BC_measurements': 1440, 'total_BC_pools': 2880,
            'precision_scalar_comparisons': 270, 'codec_scalar_comparisons': 540,
            'baseline_target_pool_denominator': 180,
            'paired_injection_pool_denominator_per_level': 180,
            'recording_contrast_denominator_per_level': 90,
            'codec_baseline_mask_comparisons': 360,
            'codec_minus6_recording_contrasts': 180,
            'gain_grid_cell_comparisons': 41040, 'polarity_grid_cell_comparisons': 41040})

    def test_margins_are_not_frozen_and_each_codec_gets_full_observability_gate(self):
        margins = m.margins()
        self.assertEqual(margins['provenance'],
                         'development-informed engineering margins; not externally validated')
        self.assertTrue(margins['lock_required_before_reserved_measurement'])
        self.assertFalse(margins['currently_frozen'])
        codec = margins['each_codec']
        self.assertEqual(codec['baseline_target_mask_agreement_fraction_min'], .95)
        self.assertEqual(codec['median_absolute_scalar_error_max_inclusive'], .02)
        self.assertEqual(codec['p95_absolute_scalar_error_max_inclusive'], .10)
        self.assertEqual(codec['each_decoded_codec_baseline_observability'], {
            'eligible_target_pools_min': 144, 'target_pool_denominator': 180,
            'available_recording_scalars_min': 81, 'recording_denominator': 90,
            'each_player_by_performance_eligible_pool_fraction_min': .80,
            'strata_denominator': 6})

    def test_actual_frozen_metadata_yields_only_exact_reserved_cross(self):
        source = json.loads((ARTIFACT /
            'preregistration/bicoherence_guitarset_pilot_v1/draft.json').read_text())
        rows = m.reserved_roster(source)
        self.assertEqual(len(rows), 90)
        self.assertEqual([row['item_id'] for row in rows], sorted(row['item_id'] for row in rows))
        self.assertEqual({row['split_role'] for row in rows}, {'reserved'})
        self.assertEqual({row['player_id'] for row in rows}, {'00', '02', '05'})
        self.assertEqual(len({row['score_id'] for row in rows}), 15)
        self.assertEqual(sum(row['performance'] == 'comp' for row in rows), 45)
        self.assertTrue(all(row['historical_native_decode']['sample_rate_hz'] == 44100 for row in rows))
        self.assertTrue(all(not row['preprocessing_executed_in_this_draft'] for row in rows))

    def test_non_reserved_or_short_roster_is_rejected(self):
        source = json.loads((ARTIFACT /
            'preregistration/bicoherence_guitarset_pilot_v1/draft.json').read_text())
        source['split']['rows'] = [row for row in source['split']['rows']
                                   if row['split_role'] != 'reserved']
        with self.assertRaisesRegex(ValueError, 'roster'):
            m.reserved_roster(source)
        with self.assertRaisesRegex(ValueError, 'too short'):
            m.crop_plan(100, 44100)

    def test_binding_rejects_symlink_before_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / 'target.json'
            target.write_text('{}')
            link = root / 'link.json'
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'non-symlink'):
                m.binding(link.absolute())

    def test_actual_metadata_build_binds_codec_results_through_commit(self):
        document = m.build(ARTIFACT, '/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/'
                           'bicoherence_guitarset_reserved_admission_v1', {})
        self.assertEqual(len(document['roster']), 90)
        self.assertEqual(document['authorities']['synthetic_codec_COMMIT']['sha256'],
                         m.CODEC_SOURCE_COMMIT_SHA)
        self.assertEqual(document['authorities']['synthetic_codec_results']['sha256'],
                         m.CODEC_RESULTS_SHA)
        self.assertFalse(document['reserved_audio_read'])
        self.assertFalse(document['thresholds_tuned_on_reserved'])
        self.assertTrue(document['parent_freeze_required'])
        self.assertIn('reserved producer and synthetic tests do not yet exist',
                      document['implementation_gaps'])

    def test_publish_is_new_nonauthorizing_commit(self):
        document = m.build(ARTIFACT, '/future/exclusive/reserved-output', {})
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory) / 'draft'
            result = m.publish(document, destination)
            commit = json.loads((destination / 'COMMIT.json').read_text())
            self.assertEqual(commit['status'], 'committed_nonauthorizing_reserved_admission_draft')
            self.assertFalse(commit['reserved_audio_accessed'])
            self.assertFalse(commit['numerical_gate_evaluated'])
            self.assertEqual(result['draft']['sha256'], commit['draft']['sha256'])
            verified = m.verify_draft(destination, result['commit']['sha256'])
            self.assertEqual(verified['status'],
                             'verified_nonauthorizing_reserved_admission_draft')
            self.assertEqual(verified['reserved_rows'], 90)
            with self.assertRaisesRegex(ValueError, 'new draft'):
                m.publish(document, destination)

    def test_verify_draft_rejects_resealed_roster_mutation(self):
        document = m.build(ARTIFACT, '/future/exclusive/reserved-output', {})
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory) / 'draft'
            result = m.publish(document, destination)
            draft_path = destination / 'draft.json'
            changed = json.loads(draft_path.read_text())
            changed['roster'][0]['preprocessing_executed_in_this_draft'] = True
            changed['reserved_roster_sha256'] = m.value_hash(changed['roster'])
            draft_path.unlink()
            m.write_json_new(draft_path, changed)
            commit_path = destination / 'COMMIT.json'
            commit = json.loads(commit_path.read_text())
            commit['draft'] = m.binding(draft_path.absolute())
            commit_path.unlink()
            m.write_json_new(commit_path, commit)
            with self.assertRaisesRegex(ValueError, 'preprocessing'):
                m.verify_draft(destination, m.digest(commit_path))


if __name__ == '__main__':
    unittest.main()
