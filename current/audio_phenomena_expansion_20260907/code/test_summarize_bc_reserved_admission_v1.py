import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('bc_reserved_report_under_test',
    HERE / 'summarize_bc_reserved_admission_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def coverage(successful=90, nulls=0):
    return {'recording_denominator': 90, 'successful_measurements': successful,
        'processing_failures_or_skips': 90 - successful, 'pool_denominator': 180,
        'eligible_pools': 180 - 2 * nulls,
        'scientifically_missing_pools': 2 * nulls,
        'unmeasured_pools_due_to_failure_or_skip': 2 * (90 - successful),
        'covered_recording_scalars': successful - nulls,
        'scientifically_null_recording_scalars': nulls}


def contrast(value=.3):
    per = [{'item_id': f'id{i:03d}', 'player_id': str(i % 3),
            'score_id': f's{i % 15}', 'performance': 'comp' if i % 2 == 0 else 'solo',
            'status': 'compared_not_thresholded',
            'operational_median_difference': value,
            'paired_pool_mean_difference': value,
            'paired_pool_count': 2, 'pool_denominator': 2}
           for i in range(90)]
    metrics = {}
    for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
        metrics[metric] = {'covered_recordings': 90, 'recording_mean': value,
            'equal_score': {'equal_group_mean': value, 'rows': []},
            'equal_player_secondary': {'rows': [
                {'player_id': str(i), 'mean': value} for i in range(3)]},
            'performance_strata': {'comp': {'mean': value}, 'solo': {'mean': value}}}
    return {'recording_denominator': 90, 'processing_failure_recordings': 0,
            'metrics': metrics, 'paired_pool_denominator': 180,
            'paired_covered_pools': 180, 'per_recording': per}


def delta(nulls=0):
    return {'comparison_denominator': 90, 'successful_comparisons': 90,
            'processing_failure_comparisons': 0, 'scalar_pair_covered': 90 - nulls,
            'scientific_null_pairs': nulls, 'signed_mean_delta': 0.001,
            'median_absolute_delta': 0.001, 'p95_absolute_delta': 0.002,
            'p95_quantile_method': 'linear interpolation at h=(n-1)*0.95'}


def transitions():
    return {'eligible_to_eligible': 180, 'missing_to_missing': 0,
            'missing_to_eligible': 0, 'eligible_to_missing': 0,
            'comparison_denominator': 90, 'processing_failures': 0}


class Fixture:
    def __init__(self, *, gate=True, failures=False, scientific_nulls=0):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.result = self.base / 'result'; self.result.mkdir()
        self.audits = self.base / 'audits'; self.audits.mkdir()
        self.reports = self.base / 'reports'; self.reports.mkdir()
        counts = {
            'float64': {'expected': 630, 'attempted': 630, 'successful': 630,
                        'failed_after_attempt': 0, 'skipped_before_attempt': 0},
            'float32_precision_control': {'expected': 270, 'attempted': 270,
                        'successful': 270, 'failed_after_attempt': 0, 'skipped_before_attempt': 0},
            'codec_decoded': {'expected': 540, 'attempted': 540, 'successful': 540,
                        'failed_after_attempt': 0, 'skipped_before_attempt': 0},
            'all': {'expected': 1440, 'attempted': 1440, 'successful': 1440,
                    'failed_after_attempt': 0, 'skipped_before_attempt': 0}}
        if failures:
            counts['float64'].update(attempted=630, successful=629, failed_after_attempt=1)
            counts['all'].update(attempted=1440, successful=1439, failed_after_attempt=1)
        checks = [{'name': name, 'observed': index, 'relation': '>=',
                   'threshold': 0, 'passed': gate}
                  for index, name in enumerate(m.expected_check_names())]
        check_block = {'status': 'producer_checks_only_not_authoritative_not_admitted',
            'margins_source_sha256': 'a' * 64, 'check_count': 72,
            'passed_checks': 72 if gate else 0,
            'all_producer_checks_passed': gate,
            'independent_replay_required': True, 'BC_admitted': False, 'checks': checks}
        representations = {name: coverage(nulls=scientific_nulls)
                           for name in m.REPRESENTATIONS}
        delta_by = {condition: delta(scientific_nulls) for condition in m.SELECTED}
        codec_deltas = {codec: {'all_conditions': delta(scientific_nulls),
            'by_condition': dict(delta_by)} for codec in m.CODECS}
        codec_contrasts = {name: contrast() for name in m.REPRESENTATIONS}
        self.summary = {'version': m.PRODUCER_VERSION,
            'status': 'reserved_measurements_complete_pending_independent_replay_not_admitted',
            'measurement_counts': counts, 'expected_denominators': {'total': 1440},
            'original_float64': {'baseline_coverage': coverage(nulls=scientific_nulls),
                'levels': {'minus6db': contrast(), '0db': contrast()},
                'full_grid_summary_available': True,
                'full_grid_historical_summary': {'nuisance': {
                    'common_gain': {'pool_denominator': 180,
                        'maximum_finite_b2_difference': 0.0},
                    'polarity': {'pool_denominator': 180,
                        'maximum_finite_b2_difference': 0.0}}}},
            'precision_and_codec': {'baseline': {
                'representation_coverage': representations,
                'precision_pool_transition_counts': transitions(),
                'codec_pool_transition_counts': {codec: transitions() for codec in m.CODECS}},
                'scalar_delta_distributions': {
                    'precision_float32_minus_float64': {
                        'all_conditions': delta(scientific_nulls), 'by_condition': dict(delta_by)},
                    'codec_decoded_minus_float32': codec_deltas},
                'condition_contrasts': codec_contrasts},
            'margins': {'development_informed': True},
            'producer_margin_checks': check_block,
            'per_recording': [{'item_id': f'id{i:03d}',
                'construction': {'status': 'processing_failure' if failures and i == 0 else 'success'}}
                for i in range(90)]}
        if failures:
            self.summary['status'] = 'reserved_measurements_with_retained_failures_pending_replay_not_admitted'
        summary_path = self.result / 'summary.json'; m.write_new(
            summary_path, m.canonical(self.summary) + b'\n')
        summary_entry = m.binding(summary_path)
        products = {'summary.json': {key: summary_entry[key] for key in ('bytes', 'sha256')}}
        self.commit = {'version': m.PRODUCER_VERSION,
            'status': ('committed_reserved_measurements_complete_pending_independent_replay_not_admitted'
                       if not failures else
                       'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted'),
            **m.PRODUCER_SCOPE, 'expected_measurements': m.EXPECTED,
            'measurement_counts': counts, 'summary_sha256': m.value_hash(self.summary),
            'products': products, 'products_sha256': m.value_hash(products),
            'source_graph_sha256': 'b' * 64}
        commit_path = self.result / 'COMMIT.json'; m.write_new(
            commit_path, m.canonical(self.commit) + b'\n')
        self.commit_entry = m.binding(commit_path)
        complete = not failures
        scientific_gate = complete and gate
        self.audit = {'version': m.AUDITOR_VERSION, 'status': m.AUDIT_STATUS,
            'passed': True, 'BC_admitted': False, 'classifier_fits': 0,
            'model_scoring': False, 'thresholds_changed': False,
            'final_admission_decision': None, 'separate_decision_required': True,
            'result_COMMIT': self.commit_entry, 'summary': summary_entry,
            'measurement_counts': counts,
            'expected_denominators': self.summary['expected_denominators'],
            'result_products_sha256': self.commit['products_sha256'],
            'source_graph_sha256': self.commit['source_graph_sha256'],
            'planned_fixed_target_measurements': 1440, 'planned_fixed_target_pools': 2880,
            'successful_fixed_target_pools_recomputed': counts['all']['successful'] * 2,
            'full_float64_grid_pools_recomputed': counts['float64']['successful'] * 2,
            'independently_reconstructed_margin_checks': check_block,
            'all_scientific_requirements_satisfied_after_replay': scientific_gate}
        audit_path = self.audits / 'audit.json'; m.write_new(audit_path, m.canonical(self.audit) + b'\n')
        self.audit_entry = m.binding(audit_path)

    def close(self):
        self.tmp.cleanup()

    def sources(self):
        return m.load_sources(self.commit_entry['path'], self.commit_entry['sha256'],
                              self.audit_entry['path'], self.audit_entry['sha256'])


class ReservedReportTests(unittest.TestCase):
    def test_latex_escape_is_single_pass_for_every_special_character(self):
        self.assertEqual(m.latex_escape(r'\&%$#_{}~^'),
            r'\textbackslash{}\&\%\$\#\_\{\}\textasciitilde{}\textasciicircum{}')
        self.assertEqual(m.latex_wrap('a_b.c,d:e/f'),
            r'a\_\allowbreak{}b.\allowbreak{}c,\allowbreak{}d:\allowbreak{}e/\allowbreak{}f')

    def test_latex_wide_tables_use_bounded_landscape_columns_and_compact_values(self):
        f = Fixture(); self.addCleanup(f.close)
        sources = f.sources(); tables = m.machine_tables(sources)
        tables['contrasts'][0]['recording_mean'] = 0.123456789012345
        tex = m.render_latex(sources, tables)
        self.assertIn(r'\usepackage{pdflscape}', tex)
        self.assertIn(r'\begin{landscape}', tex)
        self.assertIn(r'\begin{longtable}{L{3.0cm}L{3.2cm}L{2.2cm}', tex)
        self.assertIn(r'\begin{longtable}{L{3.0cm}L{3.2cm}L{2.0cm}', tex)
        self.assertIn(r'\begin{longtable}{L{3.5cm}L{3.0cm}L{3.0cm}', tex)
        self.assertNotIn('{llllrrrr}', tex)
        self.assertNotIn('{llllp{8cm}}', tex)
        self.assertEqual(tex.count(r'\endfirsthead'), 4)
        self.assertEqual(tex.count(r'\endhead'), 4)
        self.assertLess(tex.index(r'\subsection*{Interpretation}'),
                        tex.index(r'\begin{landscape}'))
        self.assertIn('0.123457', tex)
        self.assertEqual(m.report_number(None), 'NA')
        self.assertEqual(m.report_number(123), '123')

    def test_machine_tables_retain_full_float_value_while_report_display_is_compact(self):
        f = Fixture(); self.addCleanup(f.close)
        precise = 0.123456789012345
        f.summary['precision_and_codec']['scalar_delta_distributions'][
            'precision_float32_minus_float64']['all_conditions']['signed_mean_delta'] = precise
        tables = m.machine_tables({'summary': f.summary})
        row = next(value for value in tables['drift'] if value['kind'] == 'scalar_delta'
                   and value['representation'] == 'float32_control'
                   and value['condition'] == 'all_conditions')
        self.assertEqual(row['signed_mean_delta'], precise)
        self.assertIn('0.123456789012345', m.csv_text([row]))
        self.assertEqual(m.report_number(precise), '0.123457')

    def test_exact_scientific_check_registry_has_72_unique_names(self):
        names = m.expected_check_names()
        self.assertEqual(len(names), 72)
        self.assertEqual(len(set(names)), 72)
        self.assertEqual(names[-1], 'processing.all_measurements_successful')

    def test_valid_audit_and_satisfied_gate_are_distinct_nonauthorizing_fields(self):
        f = Fixture(); self.addCleanup(f.close)
        sources = f.sources()
        self.assertTrue(sources['scientific_gate_satisfied'])
        package = m.publish(sources, f.reports / 'report', Path(__file__).resolve())
        self.assertTrue(package['independent_audit_valid'])
        self.assertTrue(package['scientific_gate_satisfied'])
        self.assertFalse(package['BC_admitted'])

    def test_valid_accountability_audit_can_report_scientific_gate_failure(self):
        f = Fixture(gate=False); self.addCleanup(f.close)
        sources = f.sources()
        self.assertFalse(sources['scientific_gate_satisfied'])
        package = m.publish(sources, f.reports / 'report', Path(__file__).resolve())
        self.assertIn('gate_not_satisfied', package['status'])
        self.assertTrue(package['independent_audit_valid'])
        self.assertFalse(package['BC_admitted'])

    def test_retained_processing_failure_and_scientific_nulls_remain_visible(self):
        f = Fixture(gate=False, failures=True, scientific_nulls=2); self.addCleanup(f.close)
        sources = f.sources(); tables = m.machine_tables(sources)
        self.assertEqual(sources['summary']['measurement_counts']['all']['failed_after_attempt'], 1)
        row = next(x for x in tables['coverage'] if x['representation'] == 'float32_control')
        self.assertEqual(row['scientifically_null_recording_scalars'], 2)
        package = m.publish(sources, f.reports / 'report', Path(__file__).resolve())
        self.assertFalse(package['scientific_gate_satisfied'])

    def test_missing_failed_or_unpinned_audit_is_rejected(self):
        f = Fixture(); self.addCleanup(f.close)
        with self.assertRaisesRegex(ValueError, 'caller must pin'):
            m.load_sources(f.commit_entry['path'], f.commit_entry['sha256'],
                           f.audit_entry['path'], '')
        f.audit['passed'] = False
        path = Path(f.audit_entry['path']); path.unlink(); m.write_new(path, m.canonical(f.audit) + b'\n')
        with self.assertRaisesRegex(ValueError, 'passed independent audit'):
            m.load_sources(f.commit_entry['path'], f.commit_entry['sha256'], path, m.digest(path))

    def test_mismatched_commit_summary_or_audited_checks_are_rejected(self):
        f = Fixture(); self.addCleanup(f.close)
        f.audit['independently_reconstructed_margin_checks']['checks'][0]['passed'] = False
        path = Path(f.audit_entry['path']); path.unlink(); m.write_new(path, m.canonical(f.audit) + b'\n')
        with self.assertRaisesRegex(ValueError, 'audited numerical'):
            m.load_sources(f.commit_entry['path'], f.commit_entry['sha256'], path, m.digest(path))

    def test_fake_success_counts_and_short_check_list_are_rejected(self):
        f = Fixture(); self.addCleanup(f.close)
        f.summary['measurement_counts']['all']['successful'] = 1439
        path = f.result / 'summary.json'; path.unlink(); m.write_new(path, m.canonical(f.summary) + b'\n')
        with self.assertRaisesRegex(ValueError, 'hash and count join'):
            f.sources()
        g = Fixture(); self.addCleanup(g.close)
        g.summary['producer_margin_checks']['checks'].pop()
        path = g.result / 'summary.json'; path.unlink(); m.write_new(path, m.canonical(g.summary) + b'\n')
        with self.assertRaisesRegex(ValueError, 'hash and count join'):
            g.sources()

    def test_machine_tables_cover_checks_coverage_contrasts_and_drift(self):
        f = Fixture(); self.addCleanup(f.close)
        tables = m.machine_tables(f.sources())
        self.assertEqual(len(tables['scientific_checks']), 72)
        self.assertEqual(len(tables['coverage']), 5)
        self.assertEqual(len(tables['contrasts']), 12)
        self.assertEqual(len(tables['drift']), 17)

    def test_current_independent_auditor_summary_fixture_matches_table_extractors(self):
        import audit_bc_guitarset_reserved_admission_v1 as auditor
        spec = importlib.util.spec_from_file_location('_current_reserved_auditor_tests_fixture',
            HERE / 'test_audit_bc_guitarset_reserved_admission_v1.py')
        fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
        items, document = fixtures.summary_fixture()
        summary = auditor.reconstruct_summary(items, document)
        rows = m.validate_checks(summary['producer_margin_checks'])
        tables = m.machine_tables({'summary': summary})
        self.assertEqual(len(rows), 72)
        self.assertEqual(len(tables['coverage']), 5)
        self.assertEqual(len(tables['contrasts']), 12)
        self.assertEqual(len(tables['drift']), 17)
        self.assertEqual(tables['contrasts'][0]['recording_denominator'], 90)
        self.assertEqual(tables['coverage'][0]['pool_denominator'], 180)
        self.assertEqual(tables['drift'][0]['grid_denominator'], 41040)

    def test_report_product_mutation_is_rejected_by_reconstruction(self):
        f = Fixture(); self.addCleanup(f.close)
        root = f.reports / 'report'
        package = m.publish(f.sources(), root, Path(__file__).resolve())
        report = root / 'REPORT_EN.md'; report.write_text(report.read_text() + 'changed\n')
        with self.assertRaisesRegex(ValueError, 'product inventory'):
            m.verify_package(root, package['commit']['sha256'])

    def test_output_must_be_new_and_outside_immutable_result_and_audit(self):
        f = Fixture(); self.addCleanup(f.close)
        with self.assertRaisesRegex(ValueError, 'external report'):
            m.publish(f.sources(), f.result / 'nested', Path(__file__).resolve())
        root = f.reports / 'report'; root.mkdir()
        with self.assertRaisesRegex(ValueError, 'external report'):
            m.publish(f.sources(), root, Path(__file__).resolve())


if __name__ == '__main__':
    unittest.main()
