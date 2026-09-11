#!/usr/bin/env python3
"""Report corrected reserved-BC audit v2 without changing scientific semantics.

The terminal producer COMMIT and corrected independent-audit receipt are both
caller-pinned.  Correction provenance is revalidated and rehashed separately.
No reserved audio, measurement array, threshold selection, or admission logic is
used by this reporter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile


VERSION = 'summarize_bc_reserved_admission_v2'
BASE_SHA = 'a1f736a3517335a2e6b07aa53d8ad7bd8ba272ab5cecca356ebaf8c9349b6280'
AUDITOR_VERSION = 'audit_bc_guitarset_reserved_admission_v2'
AUDIT_STATUS = 'passed_independent_reserved_numerical_and_accountability_replay_not_admitted'
AUDIT_SCOPE = 'independent numerical and retained-failure audit only; admission requires separately bound decision'
CORRECTION_VERSION = 'bc_reserved_independent_audit_correction_authority_v1'
CORRECTION_STATUS = 'authorized_independent_auditor_v2_replay_only_not_admission'
CORRECTION_DESCRIPTION = 'independent_normalized_columns_explicit_F_order_and_contiguous_assertion_only'
ORIGINAL_AUDITOR_SHA = 'cba9e6821e96f5280b21a724559a5b78c43e5b8a1995b8a72e69f3089f6893b7'
CORRECTED_AUDITOR_SHA = 'c5f5ca2f64224a5114063d51dbb674d921ea4efe5ced9bc70488ad413daa5c41'
CORRECTED_AUDITOR_TESTS_SHA = 'c660fd001e5c795c9405c8e251405d8228ea6d66a03cb1d46972f9e5fedab09a'
CORRECTION_SCOPE = {
    'independent_reserved_reaudit_authorized': True,
    'source_or_product_modification_authorized': False,
    'producer_rerun_authorized': False,
    'tolerance_changes_authorized': False,
    'threshold_changes_authorized': False,
    'BC_admission_authorized': False,
    'classifier_fits_authorized': False,
    'model_scoring_authorized': False,
}
DIAGNOSIS_NAMES = {
    'audit/bc_reserved_raw_sum_diagnosis_v1/DIAGNOSIS_EN.md',
    'audit/bc_reserved_raw_sum_diagnosis_v1/diagnose.py',
    'audit/bc_reserved_raw_sum_diagnosis_v1/result.json',
    'audit/bc_reserved_raw_sum_diagnosis_v1/capture_exception_v2.py',
    'audit/bc_reserved_raw_sum_diagnosis_v1/capture_process01/report.json',
    'audit/bc_reserved_raw_sum_diagnosis_v1/capture_process01/captured_arrays.npz',
    'audit/bc_reserved_raw_sum_diagnosis_v1/probe_alignment_v3.py',
    'audit/bc_reserved_raw_sum_diagnosis_v1/probe_neighbor_v4.py',
    'audit/bc_reserved_raw_sum_diagnosis_v1/neighbor_process01.json',
    'audit/bc_reserved_raw_sum_diagnosis_v1/parent_decimal_check.py',
    'audit/bc_reserved_raw_sum_diagnosis_v1/parent_decimal_result.json',
}


def _load_base():
    path = Path(__file__).resolve().with_name('summarize_bc_reserved_admission_v1.py')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('accepted reporter v1 pin')
    spec = importlib.util.spec_from_file_location('_accepted_bc_reserved_report_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b = _load_base()
require, canonical, value_hash = b.require, b.canonical, b.value_hash
binding, read_json = b.binding, b.read_json


def exact_binding(value, label):
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'},
            label + ' binding schema')
    require(value == binding(value['path']), label + ' binding changed')
    return value


def validate_correction_lineage(audit, commit_entry):
    authority_entry = exact_binding(audit.get('correction_authority'), 'correction authority')
    authority, reread = read_json(authority_entry['path'], authority_entry['sha256'])
    require(reread == authority_entry, 'correction authority receipt binding')
    expected_keys = {'version', 'status', 'original_parent_freeze', 'original_result_COMMIT',
                     'original_auditor', 'original_failed_audit_log', 'corrected_auditor',
                     'corrected_tests', 'diagnosis', 'scope', 'correction'}
    require(isinstance(authority, dict) and set(authority) == expected_keys
            and authority.get('version') == CORRECTION_VERSION
            and authority.get('status') == CORRECTION_STATUS
            and authority.get('scope') == CORRECTION_SCOPE
            and authority.get('correction') == CORRECTION_DESCRIPTION,
            'exact correction authority identity/scope')
    for key in ('original_parent_freeze', 'original_result_COMMIT', 'original_auditor',
                'original_failed_audit_log', 'corrected_auditor', 'corrected_tests'):
        exact_binding(authority[key], 'correction ' + key)
    diagnosis = authority.get('diagnosis')
    require(isinstance(diagnosis, dict) and set(diagnosis) == DIAGNOSIS_NAMES,
            'exact correction diagnosis inventory')
    for name, entry in diagnosis.items():
        exact_binding(entry, 'correction diagnosis ' + name)
    require(authority['original_result_COMMIT'] == commit_entry
            and audit.get('result_COMMIT') == commit_entry
            and audit.get('parent_freeze') == authority['original_parent_freeze']
            and audit.get('original_auditor') == authority['original_auditor']
            and audit.get('original_failed_audit_log') == authority['original_failed_audit_log']
            and audit.get('audit_code') == authority['corrected_auditor']
            and audit.get('audit_tests') == authority['corrected_tests']
            and audit.get('correction_diagnosis') == diagnosis
            and audit.get('correction') == CORRECTION_DESCRIPTION,
            'corrected receipt/authority lineage join')
    require(authority['original_auditor']['sha256'] == ORIGINAL_AUDITOR_SHA,
            'accepted original auditor pin')
    require(authority['corrected_auditor']['sha256'] == CORRECTED_AUDITOR_SHA
            and authority['corrected_tests']['sha256'] == CORRECTED_AUDITOR_TESTS_SHA,
            'accepted corrected auditor/test pins')
    return {'correction_authority': authority_entry,
            'original_parent_freeze': authority['original_parent_freeze'],
            'original_result_COMMIT': authority['original_result_COMMIT'],
            'original_auditor': authority['original_auditor'],
            'original_failed_audit_log': authority['original_failed_audit_log'],
            'corrected_auditor': authority['corrected_auditor'],
            'corrected_tests': authority['corrected_tests'],
            'diagnosis': diagnosis, 'scope': CORRECTION_SCOPE,
            'correction': CORRECTION_DESCRIPTION}


def load_sources(producer_commit_path, producer_commit_sha, audit_path, audit_sha):
    require(b.hash_string(producer_commit_sha) and b.hash_string(audit_sha),
            'caller must pin exact producer COMMIT and corrected audit receipt SHAs')
    commit, commit_entry = read_json(producer_commit_path, producer_commit_sha)
    root = Path(commit_entry['path']).parent
    require(Path(commit_entry['path']) == root / 'COMMIT.json'
            and commit.get('version') == b.PRODUCER_VERSION
            and commit.get('status') in (
                'committed_reserved_measurements_complete_pending_independent_replay_not_admitted',
                'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted')
            and all(commit.get(key) == value for key, value in b.PRODUCER_SCOPE.items())
            and commit.get('expected_measurements') == b.EXPECTED,
            'terminal nonadmitting producer COMMIT')
    audit, audit_entry = read_json(audit_path, audit_sha)
    require(not Path(audit_entry['path']).is_relative_to(root)
            and audit.get('version') == AUDITOR_VERSION and audit.get('status') == AUDIT_STATUS
            and audit.get('passed') is True and audit.get('BC_admitted') is False
            and audit.get('classifier_fits') == 0 and audit.get('model_scoring') is False
            and audit.get('thresholds_changed') is False
            and audit.get('final_admission_decision') is None
            and audit.get('separate_decision_required') is True
            and audit.get('reserved_audio_read') is True
            and audit.get('development_audio_read') is False
            and audit.get('unused_audio_read') is False
            and audit.get('processing_failures_do_not_reduce_denominators') is True
            and audit.get('producer_extractor_primitive_scalar_construction_reducer_imported') is False
            and audit.get('construction_and_float32_cast_tolerance') ==
                'bit_exact_dtype_shape_and_bytes_no_tolerance'
            and audit.get('BC_numerical_comparison_tolerance') ==
                {'absolute': 2e-12, 'relative': 2e-11, 'masks_and_nulls': 'exact'}
            and audit.get('scope') == AUDIT_SCOPE,
            'passed corrected independent audit authority required')
    correction = validate_correction_lineage(audit, commit_entry)
    summary, summary_entry = read_json(root / 'summary.json')
    product = commit.get('products', {}).get('summary.json')
    require(product == {key: summary_entry[key] for key in ('bytes', 'sha256')}
            and audit.get('summary') == summary_entry
            and commit.get('summary_sha256') == value_hash(summary)
            and summary.get('version') == b.PRODUCER_VERSION
            and summary.get('measurement_counts') == commit.get('measurement_counts')
            and audit.get('measurement_counts') == commit.get('measurement_counts')
            and audit.get('expected_denominators') == summary.get('expected_denominators')
            and audit.get('result_products_sha256') == commit.get('products_sha256')
            and audit.get('source_graph_sha256') == commit.get('source_graph_sha256'),
            'producer/auditor/summary hash and count join')
    counts = summary['measurement_counts']; b.validate_counts(counts)
    checks = b.validate_checks(summary.get('producer_margin_checks'))
    compare_margin_checks(audit.get('independently_reconstructed_margin_checks'),
                          summary['producer_margin_checks'])
    complete = (counts['all']['successful'] == b.EXPECTED['measurements']
                and all(row.get('construction', {}).get('status') == 'success'
                        for row in summary.get('per_recording', []))
                and len(summary.get('per_recording', [])) == b.EXPECTED['recordings'])
    gate = complete and all(row['passed'] for row in checks)
    require(audit.get('successful_fixed_target_pools_recomputed') == counts['all']['successful'] * 2
            and audit.get('full_float64_grid_pools_recomputed') == counts['float64']['successful'] * 2
            and audit.get('planned_fixed_target_measurements') == b.EXPECTED['measurements']
            and audit.get('planned_fixed_target_pools') == b.EXPECTED['pools']
            and audit.get('all_scientific_requirements_satisfied_after_replay') is gate,
            'audit pool counts/scientific-gate classification')
    expected_status = ('committed_reserved_measurements_complete_pending_independent_replay_not_admitted'
                       if complete else
                       'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted')
    require(commit['status'] == expected_status, 'producer qualified terminal status')
    return {'commit': commit, 'commit_binding': commit_entry, 'audit': audit,
            'audit_binding': audit_entry, 'summary': summary,
            'summary_binding': summary_entry, 'result_root': str(root),
            'complete_measurements': complete, 'scientific_gate_satisfied': gate,
            'correction_lineage': correction}


def machine_tables(sources):
    return b.machine_tables(sources)


def compare_observed(actual, expected, message='observed'):
    """Compare only floating observed leaves with the auditor's frozen tolerance.

    Container shape, nulls, booleans, strings, and integer counts remain exact,
    including Python/JSON scalar type.  This intentionally does not apply a
    tolerance to thresholds, relations, or pass/fail decisions.
    """
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), message + ' keys')
        for key in expected:
            compare_observed(actual[key], expected[key], message + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), message + ' length')
        for index, value in enumerate(expected):
            compare_observed(actual[index], value, f'{message}[{index}]')
    elif isinstance(expected, float):
        require(isinstance(actual, float) and math.isfinite(actual) and math.isfinite(expected)
                and abs(actual - expected) <= 2e-12 + 2e-11 * abs(expected),
                message + ' floating tolerance')
    else:
        require(type(actual) is type(expected) and actual == expected, message + ' exact value/type')


def compare_margin_checks(audited, producer):
    """Join independently reconstructed checks without weakening decisions."""
    require(isinstance(audited, dict) and isinstance(producer, dict)
            and set(audited) == set(producer), 'audited numerical check block keys')
    require({key: value for key, value in audited.items() if key != 'checks'} ==
            {key: value for key, value in producer.items() if key != 'checks'},
            'audited numerical check envelope differs')
    left, right = audited.get('checks'), producer.get('checks')
    require(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 72,
            'audited numerical check row count')
    for index, (actual, expected) in enumerate(zip(left, right)):
        require(isinstance(actual, dict) and set(actual) == set(expected) ==
                {'name', 'observed', 'relation', 'threshold', 'passed'},
                f'audited numerical check row schema {index}')
        for key in ('name', 'relation', 'threshold', 'passed'):
            require(type(actual[key]) is type(expected[key]) and actual[key] == expected[key],
                    f'audited numerical check {index} exact {key}')
        compare_observed(actual['observed'], expected['observed'],
                         f'audited numerical check {index}.observed')


def correction_markdown(sources):
    line = sources['correction_lineage']
    return '\n'.join(['## Corrected-audit provenance', '',
        f"Correction authority: `{line['correction_authority']['sha256']}`  ",
        f"Original auditor: `{line['original_auditor']['sha256']}`  ",
        f"Original failed audit log: `{line['original_failed_audit_log']['sha256']}`  ",
        f"Corrected auditor: `{line['corrected_auditor']['sha256']}`  ",
        f"Corrected auditor tests: `{line['corrected_tests']['sha256']}`  ",
        f"Diagnosis files: **{len(line['diagnosis'])}**", '',
        f"Correction: `{line['correction']}`", '',
        'The correction authorizes only an independent numerical replay. It authorizes no source or product modification, '
        'producer rerun, tolerance or threshold change, BC admission, classifier fit, or model scoring.', ''])


def render_markdown(sources, tables):
    text = b.render_markdown(sources, tables)
    marker = '## Authority boundary\n'
    require(marker in text, 'accepted Markdown insertion point')
    return text.replace(marker, correction_markdown(sources) + '\n' + marker, 1)


def latex_hash(value):
    """Preserve a 64-hex digest while adding a break after each 8 characters."""
    require(b.hash_string(value), 'LaTeX provenance hash')
    return r'\allowbreak{}'.join(value[index:index + 8] for index in range(0, 64, 8))


def render_latex(sources, tables):
    text = b.render_latex(sources, tables)
    marker = r'\subsection*{Interpretation}'
    require(marker in text, 'accepted LaTeX insertion point')
    line = sources['correction_lineage']
    diagnosis_hash = value_hash(line['diagnosis'])
    block = '\n'.join((r'\subsection*{Corrected-audit provenance}',
        r'Correction authority: \texttt{' + latex_hash(line['correction_authority']['sha256']) + r'}\\',
        r'Original auditor: \texttt{' + latex_hash(line['original_auditor']['sha256']) + r'}\\',
        r'Original failed audit log: \texttt{' + latex_hash(line['original_failed_audit_log']['sha256']) + r'}\\',
        r'Corrected auditor: \texttt{' + latex_hash(line['corrected_auditor']['sha256']) + r'}\\',
        r'Corrected tests: \texttt{' + latex_hash(line['corrected_tests']['sha256']) + r'}\\',
        f"Diagnosis files: {len(line['diagnosis'])}; inventory hash: " +
            r'\texttt{' + latex_hash(diagnosis_hash) + r'}.\\',
        r'Correction: \texttt{' + b.latex_wrap(line['correction']) + r'}.\\',
        r'This correction changes no source, product, tolerance, threshold, admission, fit, or model score.', ''))
    return text.replace(marker, block + '\n' + marker, 1)


def write_new(path, data):
    return b.write_new(path, data)


def inventory(root):
    return b.inventory(root)


def publish(sources, output_root, tests_path):
    output = Path(output_root); result_root = Path(sources['result_root'])
    audit = Path(sources['audit_binding']['path'])
    authority = Path(sources['correction_lineage']['correction_authority']['path'])
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and output.parent.is_dir() and not output.parent.is_symlink()
            and not output.is_relative_to(result_root) and not result_root.is_relative_to(output)
            and not output.is_relative_to(audit.parent)
            and not output.is_relative_to(authority.parent), 'new external report root required')
    tables = machine_tables(sources)
    stage = Path(tempfile.mkdtemp(prefix='.bc-reserved-report-v2-', dir=output.parent))
    for name, rows in tables.items():
        write_new(stage / (name + '.json'), canonical({'version': VERSION, 'rows': rows}) + b'\n')
        write_new(stage / (name + '.csv'), b.csv_text(rows))
    write_new(stage / 'REPORT_EN.md', render_markdown(sources, tables))
    write_new(stage / 'REPORT.tex', render_latex(sources, tables))
    products = inventory(stage); gate = sources['scientific_gate_satisfied']
    commit = {'version': VERSION,
        'status': ('committed_valid_corrected_audit_scientific_gate_satisfied_not_admitted'
                   if gate else
                   'committed_valid_corrected_audit_scientific_gate_not_satisfied_not_admitted'),
        'independent_corrected_audit_valid': True, 'scientific_gate_satisfied': gate,
        'BC_admitted': False, 'automatic_admission_performed': False,
        'threshold_selection_performed': False, 'classifier_fits': 0, 'model_scoring': False,
        'reserved_audio_read': False, 'measurement_arrays_read': False,
        'producer_COMMIT': sources['commit_binding'],
        'independent_corrected_audit': sources['audit_binding'],
        'correction_lineage': sources['correction_lineage'],
        'audited_summary': sources['summary_binding'],
        'generator': binding(Path(__file__).resolve()), 'generator_tests': binding(tests_path),
        'accepted_reporter_v1': binding(Path(b.__file__).resolve(), BASE_SHA),
        'measurement_counts': sources['summary']['measurement_counts'],
        'scientific_check_count': 72,
        'scientific_checks_passed': sum(row['passed'] for row in tables['scientific_checks']),
        'products': products, 'products_sha256': value_hash(products),
        'interpretation': ('valid corrected independent audit; scientific gate satisfied; '
            'separate admission decision still required' if gate else
            'valid corrected independent audit; scientific gate failed; BC cannot be admitted from this result')}
    write_new(stage / 'COMMIT.json', canonical(commit) + b'\n')
    current = load_sources(sources['commit_binding']['path'], sources['commit_binding']['sha256'],
                           sources['audit_binding']['path'], sources['audit_binding']['sha256'])
    require(current == sources and inventory(stage) == products,
            'source/report/correction graph changed during publication')
    os.rename(stage, output)
    return verify_package(output, b.digest(output / 'COMMIT.json'))


def verify_package(root, commit_sha):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
            'safe report root')
    commit, commit_entry = read_json(root / 'COMMIT.json', commit_sha)
    require(commit.get('version') == VERSION and commit.get('status') in (
        'committed_valid_corrected_audit_scientific_gate_satisfied_not_admitted',
        'committed_valid_corrected_audit_scientific_gate_not_satisfied_not_admitted')
        and commit.get('independent_corrected_audit_valid') is True
        and commit.get('BC_admitted') is False and commit.get('automatic_admission_performed') is False
        and commit.get('threshold_selection_performed') is False
        and commit.get('classifier_fits') == 0 and commit.get('model_scoring') is False
        and commit.get('reserved_audio_read') is False and commit.get('measurement_arrays_read') is False,
        'nonauthorizing corrected report COMMIT scope')
    tests = commit.get('generator_tests')
    require(commit.get('generator') == binding(Path(__file__).resolve())
            and commit.get('accepted_reporter_v1') == binding(Path(b.__file__).resolve(), BASE_SHA)
            and isinstance(tests, dict) and tests == binding(tests.get('path', '')),
            'report generator/test/base binding changed')
    products = inventory(root)
    require(commit.get('products') == products and commit.get('products_sha256') == value_hash(products),
            'report product inventory')
    sources = load_sources(commit['producer_COMMIT']['path'], commit['producer_COMMIT']['sha256'],
                           commit['independent_corrected_audit']['path'],
                           commit['independent_corrected_audit']['sha256'])
    require(commit.get('correction_lineage') == sources['correction_lineage']
            and commit.get('audited_summary') == sources['summary_binding']
            and commit.get('measurement_counts') == sources['summary']['measurement_counts']
            and commit.get('scientific_gate_satisfied') is sources['scientific_gate_satisfied']
            and commit.get('scientific_checks_passed') ==
                sources['summary']['producer_margin_checks']['passed_checks']
            and commit.get('scientific_check_count') == 72,
            'report/source/correction semantic join')
    tables = machine_tables(sources)
    for name, rows in tables.items():
        require(read_json(root / (name + '.json'))[0] == {'version': VERSION, 'rows': rows}
                and (root / (name + '.csv')).read_text(encoding='utf-8') == b.csv_text(rows),
                'machine table reconstruction: ' + name)
    require((root / 'REPORT_EN.md').read_text(encoding='utf-8') == render_markdown(sources, tables)
            and (root / 'REPORT.tex').read_text(encoding='utf-8') == render_latex(sources, tables),
            'rendered report reconstruction')
    return {'status': commit['status'], 'commit': commit_entry,
            'independent_corrected_audit_valid': True,
            'scientific_gate_satisfied': sources['scientific_gate_satisfied'],
            'scientific_checks_passed': commit['scientific_checks_passed'],
            'scientific_check_count': 72, 'BC_admitted': False,
            'reserved_audio_read': False, 'measurement_arrays_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-commit', required=True)
    parser.add_argument('--producer-commit-sha256', required=True)
    parser.add_argument('--independent-corrected-audit', required=True)
    parser.add_argument('--independent-corrected-audit-sha256', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--tests', required=True)
    parser.add_argument('--mode', choices=('preflight', 'publish'), default='preflight')
    args = parser.parse_args()
    sources = load_sources(args.producer_commit, args.producer_commit_sha256,
                           args.independent_corrected_audit,
                           args.independent_corrected_audit_sha256)
    result = {'status': 'terminal_corrected_sources_verified_report_not_written',
              'independent_corrected_audit_valid': True,
              'scientific_gate_satisfied': sources['scientific_gate_satisfied'],
              'scientific_checks_passed': sources['summary']['producer_margin_checks']['passed_checks'],
              'scientific_check_count': 72, 'BC_admitted': False,
              'reserved_audio_read': False, 'measurement_arrays_read': False}
    if args.mode == 'publish':
        result = publish(sources, args.output_root, args.tests)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
