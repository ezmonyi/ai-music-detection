import copy
import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


m = load('bc_reserved_report_v2_under_test', 'summarize_bc_reserved_admission_v2.py')
t1 = load('bc_reserved_report_v1_tests_for_fixture', 'test_summarize_bc_reserved_admission_v1.py')


class CorrectionFixture:
    def __init__(self, *, gate=True, failures=False, scientific_nulls=0):
        self.old = t1.Fixture(gate=gate, failures=failures, scientific_nulls=scientific_nulls)
        self.base = self.old.base; self.result = self.old.result
        self.audits = self.old.audits; self.reports = self.old.reports
        authority_dir = self.base / 'correction'; authority_dir.mkdir()
        freeze = authority_dir / 'parent_freeze.json'; freeze.write_text('{}\n', encoding='utf-8')
        failed_log = authority_dir / 'failed.log'; failed_log.write_text('synthetic failed audit\n', encoding='utf-8')
        diagnosis = {}
        for index, name in enumerate(sorted(m.DIAGNOSIS_NAMES)):
            path = authority_dir / 'diagnosis' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic diagnosis %d\n' % index).encode())
            diagnosis[name] = m.binding(path)
        original = m.binding(HERE / 'audit_bc_guitarset_reserved_admission_v1.py',
                             m.ORIGINAL_AUDITOR_SHA)
        corrected = m.binding(HERE / 'audit_bc_guitarset_reserved_admission_v2.py',
                              m.CORRECTED_AUDITOR_SHA)
        corrected_tests = m.binding(HERE / 'test_audit_bc_guitarset_reserved_admission_v2.py',
                                    m.CORRECTED_AUDITOR_TESTS_SHA)
        self.authority = {'version': m.CORRECTION_VERSION, 'status': m.CORRECTION_STATUS,
            'original_parent_freeze': m.binding(freeze),
            'original_result_COMMIT': self.old.commit_entry,
            'original_auditor': original, 'original_failed_audit_log': m.binding(failed_log),
            'corrected_auditor': corrected, 'corrected_tests': corrected_tests,
            'diagnosis': diagnosis, 'scope': copy.deepcopy(m.CORRECTION_SCOPE),
            'correction': m.CORRECTION_DESCRIPTION}
        self.authority_path = authority_dir / 'authority.json'
        m.write_new(self.authority_path, m.canonical(self.authority) + b'\n')
        self.authority_entry = m.binding(self.authority_path)
        self.audit = copy.deepcopy(self.old.audit)
        self.audit.update(version=m.AUDITOR_VERSION,
            correction_authority=self.authority_entry,
            original_auditor=original,
            original_failed_audit_log=self.authority['original_failed_audit_log'],
            audit_code=corrected, audit_tests=corrected_tests,
            parent_freeze=self.authority['original_parent_freeze'],
            correction_diagnosis=diagnosis, correction=m.CORRECTION_DESCRIPTION,
            reserved_audio_read=True, development_audio_read=False, unused_audio_read=False,
            processing_failures_do_not_reduce_denominators=True,
            producer_extractor_primitive_scalar_construction_reducer_imported=False,
            construction_and_float32_cast_tolerance='bit_exact_dtype_shape_and_bytes_no_tolerance',
            BC_numerical_comparison_tolerance={
                'absolute': 2e-12, 'relative': 2e-11, 'masks_and_nulls': 'exact'},
            scope=m.AUDIT_SCOPE)
        self.audit_path = self.audits / 'corrected.json'
        m.write_new(self.audit_path, m.canonical(self.audit) + b'\n')
        self.audit_entry = m.binding(self.audit_path)

    def close(self):
        self.old.close()

    def rewrite_audit(self):
        self.audit_path.unlink()
        m.write_new(self.audit_path, m.canonical(self.audit) + b'\n')
        self.audit_entry = m.binding(self.audit_path)

    def rewrite_authority_and_join(self):
        self.authority_path.unlink()
        m.write_new(self.authority_path, m.canonical(self.authority) + b'\n')
        self.authority_entry = m.binding(self.authority_path)
        self.audit['correction_authority'] = self.authority_entry
        self.rewrite_audit()

    def set_observed_float_delta(self, producer_value, audit_delta):
        """Reseal the synthetic producer/auditor graph around one float leaf."""
        self.old.summary['producer_margin_checks']['checks'][0]['observed'] = producer_value
        summary_path = self.result / 'summary.json'; summary_path.unlink()
        m.write_new(summary_path, m.canonical(self.old.summary) + b'\n')
        summary_entry = m.binding(summary_path)
        self.old.commit['products']['summary.json'] = {
            key: summary_entry[key] for key in ('bytes', 'sha256')}
        self.old.commit['products_sha256'] = m.value_hash(self.old.commit['products'])
        self.old.commit['summary_sha256'] = m.value_hash(self.old.summary)
        commit_path = self.result / 'COMMIT.json'; commit_path.unlink()
        m.write_new(commit_path, m.canonical(self.old.commit) + b'\n')
        self.old.commit_entry = m.binding(commit_path)
        self.authority['original_result_COMMIT'] = self.old.commit_entry
        self.audit['result_COMMIT'] = self.old.commit_entry
        self.audit['summary'] = summary_entry
        self.audit['result_products_sha256'] = self.old.commit['products_sha256']
        self.audit['independently_reconstructed_margin_checks']['checks'][0][
            'observed'] = producer_value + audit_delta
        self.rewrite_authority_and_join()

    def sources(self):
        return m.load_sources(self.old.commit_entry['path'], self.old.commit_entry['sha256'],
                              self.audit_entry['path'], self.audit_entry['sha256'])


class ReservedReportV2Tests(unittest.TestCase):
    def test_v1_only_receipt_is_rejected(self):
        f = t1.Fixture(); self.addCleanup(f.close)
        with self.assertRaisesRegex(ValueError, 'corrected independent audit'):
            m.load_sources(f.commit_entry['path'], f.commit_entry['sha256'],
                           f.audit_entry['path'], f.audit_entry['sha256'])

    def test_valid_correction_lineage_preserves_every_numerical_table(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        sources = f.sources()
        self.assertEqual(m.machine_tables(sources), m.b.machine_tables(sources))
        self.assertEqual(len(m.machine_tables(sources)['scientific_checks']), 72)
        self.assertTrue(sources['scientific_gate_satisfied'])
        self.assertEqual(sources['correction_lineage']['correction'], m.CORRECTION_DESCRIPTION)

    def test_load_sources_accepts_only_observed_float_within_frozen_tolerance(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        f.set_observed_float_delta(0.5, 5e-12)
        self.assertTrue(f.sources()['scientific_gate_satisfied'])
        g = CorrectionFixture(); self.addCleanup(g.close)
        g.set_observed_float_delta(0.5, 2e-11)
        with self.assertRaisesRegex(ValueError, 'floating tolerance'):
            g.sources()

    def test_actual_auditor_summary_fixture_observed_delta_uses_same_tolerance(self):
        auditor = load('_corrected_reserved_auditor_for_report_test',
                       'audit_bc_guitarset_reserved_admission_v2.py')
        fixtures = load('_corrected_reserved_auditor_fixture_for_report_test',
                        'test_audit_bc_guitarset_reserved_admission_v2.py')
        items, document = fixtures.summary_fixture()
        producer = auditor.reconstruct_summary(items, document)['producer_margin_checks']
        audited = copy.deepcopy(producer)
        target = next(row for row in audited['checks'] if isinstance(row['observed'], float))
        target['observed'] += 2e-12 + 1e-11 * abs(target['observed'])
        m.compare_margin_checks(audited, producer)
        target['observed'] += 1e-6
        with self.assertRaisesRegex(ValueError, 'floating tolerance'):
            m.compare_margin_checks(audited, producer)

    def test_check_decisions_thresholds_and_integer_observations_remain_exact(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        producer = f.old.summary['producer_margin_checks']
        for key, value in (('passed', False), ('relation', '>'), ('threshold', 1)):
            audited = copy.deepcopy(producer); audited['checks'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'exact ' + key):
                m.compare_margin_checks(audited, producer)
        audited = copy.deepcopy(producer); audited['checks'][0]['observed'] = 0.0
        with self.assertRaisesRegex(ValueError, 'exact value/type'):
            m.compare_margin_checks(audited, producer)

    def test_missing_or_mismatched_correction_lineage_is_rejected(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        del f.audit['correction_authority']; f.rewrite_audit()
        with self.assertRaisesRegex(ValueError, 'correction authority'):
            f.sources()
        g = CorrectionFixture(); self.addCleanup(g.close)
        g.audit['audit_code'] = g.authority['original_auditor']; g.rewrite_audit()
        with self.assertRaisesRegex(ValueError, 'lineage join'):
            g.sources()

    def test_resealed_scope_or_correction_change_is_rejected(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        f.authority['scope']['BC_admission_authorized'] = True
        f.rewrite_authority_and_join()
        with self.assertRaisesRegex(ValueError, 'identity/scope'):
            f.sources()
        g = CorrectionFixture(); self.addCleanup(g.close)
        g.authority['correction'] = 'different'; g.rewrite_authority_and_join()
        with self.assertRaisesRegex(ValueError, 'identity/scope'):
            g.sources()

    def test_corrected_receipt_scope_or_numerical_tolerance_change_is_rejected(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        f.audit['scope'] = 'admission authorized'; f.rewrite_audit()
        with self.assertRaisesRegex(ValueError, 'corrected independent audit'):
            f.sources()
        g = CorrectionFixture(); self.addCleanup(g.close)
        g.audit['BC_numerical_comparison_tolerance']['absolute'] = 1e-6; g.rewrite_audit()
        with self.assertRaisesRegex(ValueError, 'corrected independent audit'):
            g.sources()

    def test_original_result_or_diagnosis_inventory_mismatch_is_rejected(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        f.authority['original_result_COMMIT'] = f.authority['original_parent_freeze']
        f.rewrite_authority_and_join()
        with self.assertRaisesRegex(ValueError, 'lineage join'):
            f.sources()
        g = CorrectionFixture(); self.addCleanup(g.close)
        g.authority['diagnosis'].pop(next(iter(g.authority['diagnosis'])))
        g.audit['correction_diagnosis'] = g.authority['diagnosis']
        g.rewrite_authority_and_join()
        with self.assertRaisesRegex(ValueError, 'diagnosis inventory'):
            g.sources()

    def test_bound_correction_file_mutation_is_rejected(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        sources = f.sources()
        path = Path(next(iter(sources['correction_lineage']['diagnosis'].values()))['path'])
        path.write_bytes(path.read_bytes() + b'mutation')
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            f.sources()

    def test_report_adds_provenance_without_changing_gate_semantics(self):
        f = CorrectionFixture(gate=False); self.addCleanup(f.close)
        sources = f.sources(); tables = m.machine_tables(sources)
        md = m.render_markdown(sources, tables); tex = m.render_latex(sources, tables)
        self.assertIn('Corrected-audit provenance', md)
        self.assertIn(m.CORRECTION_DESCRIPTION, md)
        self.assertIn(r'\subsection*{Corrected-audit provenance}', tex)
        self.assertEqual(tex.count(r'\endfirsthead'), 4)
        self.assertEqual(tex.count(r'\endhead'), 4)
        self.assertFalse(sources['scientific_gate_satisfied'])
        self.assertIn('was not fully satisfied', md)

    def test_latex_hash_preserves_text_and_adds_seven_breakpoints(self):
        value = '0123456789abcdef' * 4
        wrapped = m.latex_hash(value)
        self.assertEqual(wrapped.replace(r'\allowbreak{}', ''), value)
        self.assertEqual(wrapped.count(r'\allowbreak{}'), 7)
        f = CorrectionFixture(); self.addCleanup(f.close)
        tex = m.render_latex(f.sources(), m.machine_tables(f.sources()))
        self.assertIn(m.latex_hash(f.authority_entry['sha256']), tex)
        self.assertNotIn(r'\texttt{' + f.authority_entry['sha256'] + '}', tex)

    def test_publish_commit_binds_full_correction_lineage(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        result = m.publish(f.sources(), f.reports / 'report-v2', Path(__file__).resolve())
        self.assertTrue(result['independent_corrected_audit_valid'])
        commit = m.read_json(f.reports / 'report-v2' / 'COMMIT.json')[0]
        self.assertEqual(commit['correction_lineage']['correction_authority'], f.authority_entry)
        self.assertEqual(commit['correction_lineage']['corrected_auditor']['sha256'],
                         m.CORRECTED_AUDITOR_SHA)
        self.assertFalse(commit['BC_admitted'])

    def test_verify_package_rejects_postpublication_correction_mutation(self):
        f = CorrectionFixture(); self.addCleanup(f.close)
        result = m.publish(f.sources(), f.reports / 'report-v2', Path(__file__).resolve())
        path = Path(f.authority['original_failed_audit_log']['path'])
        path.write_bytes(path.read_bytes() + b'mutation')
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            m.verify_package(f.reports / 'report-v2', result['commit']['sha256'])

    def test_retained_failure_and_scientific_null_remain_visible(self):
        f = CorrectionFixture(gate=False, failures=True, scientific_nulls=2)
        self.addCleanup(f.close)
        sources = f.sources(); tables = m.machine_tables(sources)
        self.assertEqual(sources['summary']['measurement_counts']['all']['failed_after_attempt'], 1)
        row = next(x for x in tables['coverage'] if x['representation'] == 'float32_control')
        self.assertEqual(row['scientifically_null_recording_scalars'], 2)
        self.assertFalse(sources['scientific_gate_satisfied'])


if __name__ == '__main__':
    unittest.main()
