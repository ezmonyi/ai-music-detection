#!/usr/bin/env python3
"""Render a nonauthorizing report from terminal producer and independent audit.

Only JSON authorities are consumed. The report never opens reserved audio or
measurement arrays, never changes thresholds, and never performs admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


VERSION = 'summarize_bc_reserved_admission_v1'
PRODUCER_VERSION = 'run_bc_guitarset_reserved_admission_v1'
AUDITOR_VERSION = 'audit_bc_guitarset_reserved_admission_v1'
AUDIT_STATUS = 'passed_independent_reserved_numerical_and_accountability_replay_not_admitted'
EXPECTED = {'recordings': 90, 'float64': 630, 'precision': 270, 'codec': 540,
            'measurements': 1440, 'pools': 2880}
COUNT_EXPECTED = {'float64': 630, 'float32_precision_control': 270,
                  'codec_decoded': 540, 'all': 1440}
PRODUCER_SCOPE = {'reserved_recordings': 90, 'development_audio_read': False,
                  'unused_audio_read': False, 'classifier_fits': 0,
                  'model_scoring': False, 'BC_admitted': False,
                  'independent_numerical_replay_passed': False}
CODECS = ('mp3_128k', 'opus_96k')
SELECTED = ('baseline', 'closed_minus6db', 'independent_minus6db')
REPRESENTATIONS = ('accepted_float64', 'float32_control', *CODECS)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path, expected=None):
    candidate = Path(path)
    require(candidate.is_absolute() and candidate.resolve() == candidate
            and candidate.is_file() and not candidate.is_symlink(),
            'safe canonical regular file required: ' + str(candidate))
    result = {'path': str(candidate), 'bytes': candidate.stat().st_size,
              'sha256': digest(candidate)}
    require(expected is None or result['sha256'] == expected,
            'file SHA mismatch: ' + str(candidate))
    return result


def read_json(path, expected=None):
    entry = binding(path, expected)
    return json.loads(Path(entry['path']).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))), entry


def contrast_check_names(prefix):
    result = [prefix + '.paired_pool_support']
    for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
        result.extend(prefix + '.' + metric + suffix for suffix in (
            '.covered_recordings', '.equal_score', '.positive_recordings',
            '.all_players_positive', '.both_performances_positive'))
    return result


def expected_check_names():
    names = ['original.baseline.eligible_pools',
             'original.baseline.covered_recording_scalars',
             'original.baseline.every_player_performance_fraction']
    for level in ('minus6db', '0db'):
        names.extend(contrast_check_names('original.' + level))
    for condition in ('common_gain', 'polarity'):
        names.extend(('original.' + condition + '.full_grid_and_target_mask_exact',
                      'original.' + condition + '.maximum_common_finite_b2_error'))
    for codec in CODECS:
        names.extend((codec + '.baseline.eligible_pools',
                      codec + '.baseline.covered_recording_scalars',
                      codec + '.baseline.every_player_performance_fraction',
                      codec + '.baseline.mask_agreement_fraction'))
        for condition in SELECTED:
            names.extend((codec + '.' + condition + '.median_absolute_scalar_error',
                          codec + '.' + condition + '.p95_absolute_scalar_error'))
        names.extend(contrast_check_names(codec + '.minus6db'))
    names.append('processing.all_measurements_successful')
    require(len(names) == 72 and len(set(names)) == 72, 'internal exact72 check registry')
    return names


def validate_counts(counts):
    require(isinstance(counts, dict) and set(counts) == set(COUNT_EXPECTED),
            'exact measurement count groups')
    for group, expected in COUNT_EXPECTED.items():
        row = counts[group]
        require(set(row) == {'expected', 'attempted', 'successful',
                             'failed_after_attempt', 'skipped_before_attempt'}
                and row['expected'] == expected
                and row['attempted'] == row['successful'] + row['failed_after_attempt']
                and row['expected'] == row['successful'] + row['failed_after_attempt']
                    + row['skipped_before_attempt'], 'measurement accounting: ' + group)
    for key in ('expected', 'attempted', 'successful', 'failed_after_attempt',
                'skipped_before_attempt'):
        require(counts['all'][key] == sum(counts[group][key] for group in
                ('float64', 'float32_precision_control', 'codec_decoded')),
                'aggregate measurement accounting: ' + key)


def validate_checks(checks):
    require(isinstance(checks, dict)
            and checks.get('status') == 'producer_checks_only_not_authoritative_not_admitted'
            and checks.get('independent_replay_required') is True
            and checks.get('BC_admitted') is False,
            'producer scientific-check envelope')
    rows = checks.get('checks')
    require(isinstance(rows, list) and [row.get('name') for row in rows] == expected_check_names(),
            'exact ordered 72 scientific checks')
    require(all(set(row) == {'name', 'observed', 'relation', 'threshold', 'passed'}
                and isinstance(row['passed'], bool) for row in rows),
            'scientific check row schema')
    passed = sum(row['passed'] for row in rows)
    require(checks.get('check_count') == 72 and checks.get('passed_checks') == passed
            and checks.get('all_producer_checks_passed') is (passed == 72),
            'scientific check counts/boolean')
    return rows


def load_sources(producer_commit_path, producer_commit_sha, audit_path, audit_sha):
    require(hash_string(producer_commit_sha) and hash_string(audit_sha),
            'caller must pin exact producer COMMIT and audit receipt SHAs')
    commit, commit_entry = read_json(producer_commit_path, producer_commit_sha)
    root = Path(commit_entry['path']).parent
    require(Path(commit_entry['path']) == root / 'COMMIT.json'
            and commit.get('version') == PRODUCER_VERSION
            and commit.get('status') in (
                'committed_reserved_measurements_complete_pending_independent_replay_not_admitted',
                'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted')
            and all(commit.get(key) == value for key, value in PRODUCER_SCOPE.items())
            and commit.get('expected_measurements') == EXPECTED,
            'terminal nonadmitting producer COMMIT')
    audit, audit_entry = read_json(audit_path, audit_sha)
    require(not Path(audit_entry['path']).is_relative_to(root)
            and audit.get('version') == AUDITOR_VERSION and audit.get('status') == AUDIT_STATUS
            and audit.get('passed') is True and audit.get('BC_admitted') is False
            and audit.get('classifier_fits') == 0 and audit.get('model_scoring') is False
            and audit.get('thresholds_changed') is False
            and audit.get('final_admission_decision') is None
            and audit.get('separate_decision_required') is True
            and audit.get('result_COMMIT') == commit_entry,
            'passed independent audit authority required')
    summary_path = root / 'summary.json'
    summary, summary_entry = read_json(summary_path)
    product = commit.get('products', {}).get('summary.json')
    require(product == {key: summary_entry[key] for key in ('bytes', 'sha256')}
            and audit.get('summary') == summary_entry
            and commit.get('summary_sha256') == value_hash(summary)
            and summary.get('version') == PRODUCER_VERSION
            and summary.get('measurement_counts') == commit.get('measurement_counts')
            and audit.get('measurement_counts') == commit.get('measurement_counts')
            and audit.get('expected_denominators') == summary.get('expected_denominators')
            and audit.get('result_products_sha256') == commit.get('products_sha256')
            and audit.get('source_graph_sha256') == commit.get('source_graph_sha256'),
            'producer/auditor/summary hash and count join')
    counts = summary['measurement_counts']; validate_counts(counts)
    checks = validate_checks(summary.get('producer_margin_checks'))
    require(audit.get('independently_reconstructed_margin_checks') ==
            summary['producer_margin_checks'], 'audited numerical check report differs')
    complete = (counts['all']['successful'] == EXPECTED['measurements']
                and all(row.get('construction', {}).get('status') == 'success'
                        for row in summary.get('per_recording', []))
                and len(summary.get('per_recording', [])) == EXPECTED['recordings'])
    gate = complete and all(row['passed'] for row in checks)
    require(audit.get('successful_fixed_target_pools_recomputed') == counts['all']['successful'] * 2
            and audit.get('full_float64_grid_pools_recomputed') == counts['float64']['successful'] * 2
            and audit.get('planned_fixed_target_measurements') == EXPECTED['measurements']
            and audit.get('planned_fixed_target_pools') == EXPECTED['pools']
            and audit.get('all_scientific_requirements_satisfied_after_replay') is gate,
            'audit pool counts/scientific-gate classification')
    expected_status = ('committed_reserved_measurements_complete_pending_independent_replay_not_admitted'
                       if complete else
                       'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted')
    require(commit['status'] == expected_status, 'producer qualified terminal status')
    return {'commit': commit, 'commit_binding': commit_entry, 'audit': audit,
            'audit_binding': audit_entry, 'summary': summary,
            'summary_binding': summary_entry, 'result_root': str(root),
            'complete_measurements': complete, 'scientific_gate_satisfied': gate}


def coverage_rows(summary):
    original = summary['original_float64']['baseline_coverage']
    rows = [{'dataset_view': 'original_float64', 'representation': 'accepted_float64',
             'condition': 'baseline', **original}]
    coverage = summary['precision_and_codec']['baseline']['representation_coverage']
    require(set(coverage) == set(REPRESENTATIONS), 'exact baseline representation coverage')
    rows.extend({'dataset_view': 'precision_and_codec', 'representation': name,
                 'condition': 'baseline', **coverage[name]} for name in REPRESENTATIONS)
    return rows


def contrast_rows(summary):
    rows = []
    def add(view, representation, level, value):
        require(value.get('recording_denominator') == 90 and 'metrics' in value
                and 'per_recording' in value, 'contrast summary schema')
        for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
            entry = value['metrics'][metric]
            per = value['per_recording']
            rows.append({'dataset_view': view, 'representation': representation,
                'level': level, 'metric': metric,
                'recording_denominator': value['recording_denominator'],
                'covered_recordings': entry['covered_recordings'],
                'processing_failure_recordings': value['processing_failure_recordings'],
                'recording_mean': entry['recording_mean'],
                'paired_pool_denominator': value['paired_pool_denominator'],
                'paired_covered_pools': value['paired_covered_pools'],
                'equal_score_mean': entry['equal_score']['equal_group_mean'],
                'equal_player_rows': entry['equal_player_secondary']['rows'],
                'performance_strata': entry['performance_strata'],
                'positive_recordings': sum(row[metric] is not None and row[metric] > 0 for row in per),
                'per_recording': per})
    original = summary['original_float64']['levels']
    require(set(original) == {'minus6db', '0db'}, 'original contrast levels')
    for level in ('minus6db', '0db'):
        add('original_float64', 'accepted_float64', level, original[level])
    codec = summary['precision_and_codec']['condition_contrasts']
    require(set(codec) == set(REPRESENTATIONS), 'codec contrast representations')
    for representation in REPRESENTATIONS:
        add('precision_and_codec', representation, 'minus6db', codec[representation])
    return rows


def drift_rows(summary):
    result = []
    historical = summary['original_float64'].get('full_grid_historical_summary')
    if historical is not None:
        for condition in ('common_gain', 'polarity'):
            result.append({'kind': 'original_full_grid_nuisance', 'representation': 'accepted_float64',
                           'condition': condition, **historical['nuisance'][condition]})
    else:
        result.extend({'kind': 'original_full_grid_nuisance_unavailable',
                       'representation': 'accepted_float64', 'condition': condition,
                       'scientific_missingness': False, 'processing_failure': True}
                      for condition in ('common_gain', 'polarity'))
    block = summary['precision_and_codec']
    deltas = block['scalar_delta_distributions']
    precision = deltas['precision_float32_minus_float64']
    result.append({'kind': 'scalar_delta', 'representation': 'float32_control',
                   'reference': 'accepted_float64', 'condition': 'all_conditions',
                   **precision['all_conditions']})
    result.extend({'kind': 'scalar_delta', 'representation': 'float32_control',
                   'reference': 'accepted_float64', 'condition': condition,
                   **precision['by_condition'][condition]} for condition in SELECTED)
    for codec in CODECS:
        values = deltas['codec_decoded_minus_float32'][codec]
        result.append({'kind': 'scalar_delta', 'representation': codec,
                       'reference': 'float32_control', 'condition': 'all_conditions',
                       **values['all_conditions']})
        result.extend({'kind': 'scalar_delta', 'representation': codec,
                       'reference': 'float32_control', 'condition': condition,
                       **values['by_condition'][condition]} for condition in SELECTED)
    baseline = block['baseline']
    result.append({'kind': 'baseline_mask_transitions', 'representation': 'float32_control',
                   'reference': 'accepted_float64', 'condition': 'baseline',
                   **baseline['precision_pool_transition_counts']})
    result.extend({'kind': 'baseline_mask_transitions', 'representation': codec,
                   'reference': 'float32_control', 'condition': 'baseline',
                   **baseline['codec_pool_transition_counts'][codec]} for codec in CODECS)
    return result


def machine_tables(sources):
    summary = sources['summary']
    checks = summary['producer_margin_checks']['checks']
    return {'scientific_checks': [{'index': index + 1, **row}
                                   for index, row in enumerate(checks)],
            'coverage': coverage_rows(summary),
            'contrasts': contrast_rows(summary),
            'drift': drift_rows(summary)}


def short(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def md_escape(value):
    return short(value).replace('|', '\\|').replace('\n', ' ')


def latex_escape(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(character, character) for character in str(value))


def latex_wrap(value):
    """Escape first, then add TeX break opportunities without changing text."""
    result = latex_escape(value)
    for delimiter in (r'\_', '.', ',', ':', '/'):
        result = result.replace(delimiter, delimiter + r'\allowbreak{}')
    return result


def report_number(value):
    """Compact display only; JSON/CSV machine tables retain original values."""
    if value is None:
        return 'NA'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, '.6g')
    return str(value)


def render_markdown(sources, tables):
    counts = sources['summary']['measurement_counts']
    passed = sum(row['passed'] for row in tables['scientific_checks'])
    gate = sources['scientific_gate_satisfied']
    lines = ['# Reserved GuitarSet BC audit report', '',
        '**Status:** independent numerical/accountability audit passed. Scientific gate ' +
        ('was fully satisfied.' if gate else 'was not fully satisfied.'), '',
        'This report does not admit BC, select thresholds, fit a classifier, or score a model. '
        'A valid audit and a satisfied scientific gate are reported separately; neither is an automatic admission decision.', '',
        '## Measurement accountability', '',
        '| Group | Expected | Attempted | Successful | Failed after attempt | Skipped before attempt |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('float64', 'float32_precision_control', 'codec_decoded', 'all'):
        row = counts[name]
        lines.append(f"| {name} | {row['expected']} | {row['attempted']} | {row['successful']} | "
                     f"{row['failed_after_attempt']} | {row['skipped_before_attempt']} |")
    lines += ['', f'Independent audit validity: **passed**. Scientific checks: **{passed}/72**. '
              f'Scientific gate satisfied: **{str(gate).lower()}**.', '',
              '## All 72 scientific checks', '',
              '| # | Check | Observed | Relation | Threshold | Passed |',
              '|---:|---|---|---|---|---|']
    for row in tables['scientific_checks']:
        lines.append(f"| {row['index']} | {row['name']} | {md_escape(row['observed'])} | "
                     f"{row['relation']} | {md_escape(row['threshold'])} | {str(row['passed']).lower()} |")
    lines += ['', '## Coverage', '',
              '| View | Representation | Successful | Eligible pools | Missing pools | Covered scalars | Null scalars |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in tables['coverage']:
        lines.append(f"| {row['dataset_view']} | {row['representation']} | {row['successful_measurements']} | "
                     f"{row['eligible_pools']} | {row['scientifically_missing_pools']} | "
                     f"{row['covered_recording_scalars']} | {row['scientifically_null_recording_scalars']} |")
    lines += ['', '## Contrasts', '',
              '| View | Representation | Level | Metric | Covered recordings | Mean | Equal-score mean | Positive recordings | Covered pools |',
              '|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in tables['contrasts']:
        lines.append(f"| {row['dataset_view']} | {row['representation']} | {row['level']} | {row['metric']} | "
                     f"{row['covered_recordings']} | {row['recording_mean']} | {row['equal_score_mean']} | "
                     f"{row['positive_recordings']} | {row['paired_covered_pools']} |")
    lines += ['', '## Drift and nuisance behavior', '',
              '| Kind | Representation | Reference | Condition | Complete retained record |',
              '|---|---|---|---|---|']
    for row in tables['drift']:
        details = {key: value for key, value in row.items()
                   if key not in ('kind', 'representation', 'reference', 'condition')}
        lines.append(f"| {row['kind']} | {row['representation']} | {row.get('reference', '')} | "
                     f"{row['condition']} | {md_escape(details)} |")
    lines += ['', 'Scientific nulls remain distinct from processing failures. The same complete rows are available as JSON and CSV.', '',
              '## Authority boundary', '',
              f"Producer COMMIT: `{sources['commit_binding']['sha256']}`  ",
              f"Independent audit: `{sources['audit_binding']['sha256']}`  ",
              f"Summary: `{sources['summary_binding']['sha256']}`", '',
              'No reserved audio or measurement arrays were read while generating this report.']
    return '\n'.join(lines) + '\n'


def render_latex(sources, tables):
    gate = 'satisfied' if sources['scientific_gate_satisfied'] else 'not satisfied'
    lines = [r'\documentclass{article}', r'\usepackage[margin=1in]{geometry}',
             r'\usepackage{longtable}', r'\usepackage{array}', r'\usepackage{pdflscape}',
             r'\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}',
             r'\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}',
             r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
             r'\begin{document}',
             r'\section*{Reserved GuitarSet BC audit report}',
             'Independent numerical/accountability audit: passed. Scientific gate: ' + gate + r'.\\',
             r'This report performs no BC admission, threshold selection, classifier fit, or model scoring.',
             r'\subsection*{All 72 scientific checks}',
             r'\begin{longtable}{r p{5cm} p{4cm} p{2.5cm} r}',
             r'\# & Check & Observed & Relation and threshold & Passed\\\hline',
             r'\endfirsthead',
             r'\# & Check & Observed & Relation and threshold & Passed\\\hline',
             r'\endhead']
    for row in tables['scientific_checks']:
        relation = latex_wrap(row['relation'] + ' ' + short(row['threshold']))
        lines.append(f"{row['index']} & {latex_wrap(row['name'])} & "
                     f"{latex_wrap(short(row['observed']))} & {relation} & "
                     f"{'yes' if row['passed'] else 'no'} " + r'\\')
    lines += [r'\end{longtable}', r'\subsection*{Measurement accountability}',
              r'\begin{tabular}{lrrrrr}',
              r'Group & Expected & Attempted & Success & Failed & Skipped\\\hline']
    for name, row in sources['summary']['measurement_counts'].items():
        lines.append(f"{latex_wrap(name)} & {row['expected']} & {row['attempted']} & "
                     f"{row['successful']} & {row['failed_after_attempt']} & {row['skipped_before_attempt']} " + r'\\')
    lines += [r'\end{tabular}', r'\subsection*{Interpretation}',
              r'Scientific nulls and processing failures are retained separately in the machine tables. '
              r'A passed accountability audit is not an admission decision.',
              r'\begin{landscape}', r'\small',
              r'\subsection*{Coverage}',
              r'\begin{longtable}{L{3.0cm}L{3.2cm}L{2.2cm}R{1.6cm}R{1.6cm}R{1.6cm}R{1.6cm}R{1.6cm}}',
              r'View & Representation & Condition & Success & Eligible & Missing & Covered & Null\\\hline',
              r'\endfirsthead',
              r'View & Representation & Condition & Success & Eligible & Missing & Covered & Null\\\hline',
              r'\endhead']
    for row in tables['coverage']:
        lines.append(' & '.join((latex_wrap(row['dataset_view']), latex_wrap(row['representation']),
            latex_wrap(row['condition']), str(row['successful_measurements']), str(row['eligible_pools']),
            str(row['scientifically_missing_pools']), str(row['covered_recording_scalars']),
            str(row['scientifically_null_recording_scalars']))) + r'\\')
    lines += [r'\end{longtable}', r'\subsection*{Contrasts}',
              r'\begin{longtable}{L{3.0cm}L{3.2cm}L{2.0cm}L{4.2cm}R{1.5cm}R{1.7cm}R{1.7cm}R{1.7cm}}',
              r'View & Representation & Level & Metric & Covered & Mean & Score mean & Pools\\\hline',
              r'\endfirsthead',
              r'View & Representation & Level & Metric & Covered & Mean & Score mean & Pools\\\hline',
              r'\endhead']
    for row in tables['contrasts']:
        lines.append(' & '.join((latex_wrap(row['dataset_view']), latex_wrap(row['representation']),
            latex_wrap(row['level']), latex_wrap(row['metric']), str(row['covered_recordings']),
            latex_escape(report_number(row['recording_mean'])),
            latex_escape(report_number(row['equal_score_mean'])),
            str(row['paired_covered_pools']))) + r'\\')
    lines += [r'\end{longtable}', r'\subsection*{Drift and nuisance behavior}',
              r'\begin{longtable}{L{3.5cm}L{3.0cm}L{3.0cm}L{2.8cm}L{9.0cm}}',
              r'Kind & Representation & Reference & Condition & Complete retained record\\\hline',
              r'\endfirsthead',
              r'Kind & Representation & Reference & Condition & Complete retained record\\\hline',
              r'\endhead']
    for row in tables['drift']:
        details = {key: value for key, value in row.items()
                   if key not in ('kind', 'representation', 'reference', 'condition')}
        lines.append(' & '.join((latex_wrap(row['kind']), latex_wrap(row['representation']),
            latex_wrap(row.get('reference', '')), latex_wrap(row['condition']),
            latex_wrap(short(details)))) + r'\\')
    lines += [r'\end{longtable}', r'\end{landscape}', r'\end{document}', '']
    return '\n'.join(lines)


def csv_text(rows):
    keys = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: short(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue()


def write_new(path, data):
    path = Path(path)
    require(not path.exists(), 'report product overwrite forbidden')
    mode = 'xb' if isinstance(data, bytes) else 'x'
    kwargs = {} if isinstance(data, bytes) else {'encoding': 'utf-8', 'newline': ''}
    with path.open(mode, **kwargs) as stream:
        stream.write(data)


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'report symlink forbidden')
        if path.is_dir():
            continue
        require(path.is_file(), 'report regular files only')
        if path.name == 'COMMIT.json':
            continue
        entry = binding(path)
        result[path.relative_to(root).as_posix()] = {key: entry[key] for key in ('bytes', 'sha256')}
    return result


def publish(sources, output_root, tests_path):
    output = Path(output_root)
    result_root = Path(sources['result_root'])
    audit = Path(sources['audit_binding']['path'])
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and output.parent.is_dir() and not output.parent.is_symlink()
            and not output.is_relative_to(result_root) and not result_root.is_relative_to(output)
            and not output.is_relative_to(audit.parent), 'new external report root required')
    tables = machine_tables(sources)
    stage = Path(tempfile.mkdtemp(prefix='.bc-reserved-report-', dir=output.parent))
    try:
        for name, rows in tables.items():
            write_new(stage / (name + '.json'), canonical({'version': VERSION, 'rows': rows}) + b'\n')
            write_new(stage / (name + '.csv'), csv_text(rows))
        write_new(stage / 'REPORT_EN.md', render_markdown(sources, tables))
        write_new(stage / 'REPORT.tex', render_latex(sources, tables))
        products = inventory(stage)
        gate = sources['scientific_gate_satisfied']
        commit = {'version': VERSION,
            'status': ('committed_valid_audit_scientific_gate_satisfied_not_admitted'
                       if gate else
                       'committed_valid_audit_scientific_gate_not_satisfied_not_admitted'),
            'independent_audit_valid': True, 'scientific_gate_satisfied': gate,
            'BC_admitted': False, 'automatic_admission_performed': False,
            'threshold_selection_performed': False, 'classifier_fits': 0,
            'model_scoring': False, 'reserved_audio_read': False,
            'measurement_arrays_read': False,
            'producer_COMMIT': sources['commit_binding'],
            'independent_audit': sources['audit_binding'],
            'audited_summary': sources['summary_binding'],
            'generator': binding(Path(__file__).resolve()), 'generator_tests': binding(tests_path),
            'measurement_counts': sources['summary']['measurement_counts'],
            'scientific_check_count': 72,
            'scientific_checks_passed': sum(row['passed'] for row in tables['scientific_checks']),
            'products': products, 'products_sha256': value_hash(products),
            'interpretation': ('valid independent audit; scientific gate satisfied; '
                'separate admission decision still required' if gate else
                'valid independent audit; scientific gate failed; BC cannot be admitted from this result')}
        write_new(stage / 'COMMIT.json', canonical(commit) + b'\n')
        current = load_sources(sources['commit_binding']['path'], sources['commit_binding']['sha256'],
                               sources['audit_binding']['path'], sources['audit_binding']['sha256'])
        require(current == sources and inventory(stage) == products,
                'source/report graph changed during publication')
        os.rename(stage, output)
        return verify_package(output, digest(output / 'COMMIT.json'))
    except Exception:
        # Leave the hidden, nonterminal stage for forensic inspection; never overwrite it.
        raise


def verify_package(root, commit_sha):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir()
            and not root.is_symlink(), 'safe report root')
    commit, commit_entry = read_json(root / 'COMMIT.json', commit_sha)
    require(commit.get('version') == VERSION and commit.get('status') in (
        'committed_valid_audit_scientific_gate_satisfied_not_admitted',
        'committed_valid_audit_scientific_gate_not_satisfied_not_admitted')
        and commit.get('independent_audit_valid') is True
        and commit.get('BC_admitted') is False
        and commit.get('automatic_admission_performed') is False
        and commit.get('threshold_selection_performed') is False
        and commit.get('classifier_fits') == 0 and commit.get('model_scoring') is False
        and commit.get('reserved_audio_read') is False
        and commit.get('measurement_arrays_read') is False,
        'nonauthorizing report COMMIT scope')
    tests_binding = commit.get('generator_tests')
    require(commit.get('generator') == binding(Path(__file__).resolve())
            and isinstance(tests_binding, dict) and 'path' in tests_binding
            and tests_binding == binding(tests_binding['path']),
            'report generator/test binding changed')
    products = inventory(root)
    require(commit.get('products') == products and commit.get('products_sha256') == value_hash(products),
            'report product inventory')
    sources = load_sources(commit['producer_COMMIT']['path'], commit['producer_COMMIT']['sha256'],
                           commit['independent_audit']['path'], commit['independent_audit']['sha256'])
    require(commit['audited_summary'] == sources['summary_binding']
            and commit['measurement_counts'] == sources['summary']['measurement_counts']
            and commit['scientific_gate_satisfied'] is sources['scientific_gate_satisfied']
            and commit['scientific_checks_passed'] ==
                sources['summary']['producer_margin_checks']['passed_checks']
            and commit['scientific_check_count'] == 72,
            'report/source semantic join')
    tables = machine_tables(sources)
    for name, rows in tables.items():
        require(read_json(root / (name + '.json'))[0] == {'version': VERSION, 'rows': rows}
                and (root / (name + '.csv')).read_text(encoding='utf-8') == csv_text(rows),
                'machine table reconstruction: ' + name)
    require((root / 'REPORT_EN.md').read_text(encoding='utf-8') == render_markdown(sources, tables)
            and (root / 'REPORT.tex').read_text(encoding='utf-8') == render_latex(sources, tables),
            'rendered report reconstruction')
    return {'status': commit['status'], 'commit': commit_entry,
            'independent_audit_valid': True,
            'scientific_gate_satisfied': sources['scientific_gate_satisfied'],
            'scientific_checks_passed': commit['scientific_checks_passed'],
            'scientific_check_count': 72, 'BC_admitted': False,
            'reserved_audio_read': False, 'measurement_arrays_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-commit', required=True)
    parser.add_argument('--producer-commit-sha256', required=True)
    parser.add_argument('--independent-audit', required=True)
    parser.add_argument('--independent-audit-sha256', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--tests', required=True)
    parser.add_argument('--mode', choices=('preflight', 'publish'), default='preflight')
    args = parser.parse_args()
    sources = load_sources(args.producer_commit, args.producer_commit_sha256,
                           args.independent_audit, args.independent_audit_sha256)
    result = {'status': 'terminal_sources_verified_report_not_written',
              'independent_audit_valid': True,
              'scientific_gate_satisfied': sources['scientific_gate_satisfied'],
              'scientific_checks_passed': sources['summary']['producer_margin_checks']['passed_checks'],
              'scientific_check_count': 72, 'BC_admitted': False,
              'reserved_audio_read': False, 'measurement_arrays_read': False}
    if args.mode == 'publish':
        result = publish(sources, args.output_root, args.tests)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
