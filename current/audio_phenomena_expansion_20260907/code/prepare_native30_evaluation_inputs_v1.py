#!/usr/bin/env python3
"""Assemble the frozen native30 54-predictor evaluation input package.

Default preflight reads only metadata contracts/COMMITs and the schedule.  The
explicit assemble mode rehashes every committed F/H/SC and S/D/R/P product,
then reads per-item JSON receipts.  It never reads cohort audio, fits a model,
computes predictions, or authorizes evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid


VERSION = 'prepare_native30_evaluation_inputs_v1'
STATUS = 'assembled_not_evaluated_not_authorized'
EXPECTED = 3830
PLAN_SHA = '94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
SCREEN_SHA = 'ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87'
SCHEDULE_SHA = '4e12300a7fe43b4f3024c1cd9e675c25b0ade3af25dded0d688b8d8f2fcab7a6'
SCHEDULE_COMMIT_SHA = '5d0e329d9ef0a794e32acbf5bee06a4a09a58b453e1741ba6232fbfb32347233'
CODE_PINS = {
    'run_native30_fhsc_cohort_v1.py': 'dc26465459df133b89d6023bb1c732e91ebab5b63ff02143280b50c383e7b3cc',
    'run_native30_fhsc_pilot_v1.py': '0f4a2724081f617c38661c0eb3fad874379561d5525b022bd6601134651dedaa',
    'phase_features.py': 'f1598ecb513ae67431571e586e436840f80114ac2b6d04fce3a46e8eeb0a5ac5',
    'musical_features.py': '72d54ceb5266e2ff2f0d1ce5bdbcfa7e8c93611784a03e40d3255672020f8083',
    'stereo_candidate_v1.py': '132bd5fe225258b72b913238bd456e3b18279b6b5a664b2191b0292aede9481c',
    'plan_native30_evaluation_schedule_v1.py': 'ff6965b3f9c834ee9d39bbf0f994595f44fa2e261d9c7c6b60a153cb9f72ce24',
}
EXPANDED_SHA = '8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00'
SDRP_BACKENDS = {
    'extract_native30_sdrp_v1': {
        'runner_sha256': 'a170b673c484ff24472e9d479c86e71c345b8d1f36a3ff899c48405dfe63cddc',
        'contract_status': 'frozen_CPU_extraction_only',
        'item_status': 'measured_native30_S_D_R_P_not_admitted',
        'commit_status': 'committed_native30_S_D_R_P_not_admitted',
        'scope': {'classifier_fits': 0, 'cohort_admitted': False,
                  'neural_inference_performed': False, 'recovery_applied': False,
                  'M_predictor': False}},
    'extract_native30_sdrp_recovery_v1': {
        'runner_sha256': '32c861d3dbe17ac41f0ccfbdc82f63b33babb2d035b94b65fee914f7124107f9',
        'contract_status': 'frozen_CPU_recovery_aware_extraction_only',
        'item_status': 'measured_native30_S_D_R_P_with_recovery_provenance_not_admitted',
        'commit_status': 'committed_native30_S_D_R_P_with_recovery_provenance_not_admitted',
        'scope': {'classifier_fits': 0, 'cohort_admitted': False,
                  'neural_inference_performed': False, 'recovery_applied': False,
                  'M_predictor': False, 'recovery_evidence_consumed': True,
                  'recovery_provenance_is_predictor': False}},
}
IDENTITY = ('id', 'source_group', 'label', 'role', 'group_id', 'component_id')
SCOPE = {'classifier_fits': 0, 'predictions_computed': 0, 'cohort_admitted': False,
         'fitting_authorized': False, 'scoring_authorized': False, 'winner_selection': False,
         'threshold_tuning': False, 'M_predictor': False, 'S16_predictor': False,
         'eligibility_predictors': False}

FAMILY_CONFIG = {
    'S': ['s8__' + name for name in (
        'tilt_1_5k_db_oct', 'hf_tilt_5_7p5k_db_oct', 'hf_ratio_5_7p5_db',
        'sibilance_ratio_5_7p5_db', 'hf_flatness_5_7p5', 'hf_entropy_5_7p5',
        'hf_crest_5_7p5_db', 'fakeprint_peak_density_5_7p5_per_khz',
        'fakeprint_periodicity_5_7p5', 'hf_flux_5_7p5', 'hf_frame_similarity_5_7p5',
        'hf_power_sd_5_7p5_db', 'hf_mod_4_12_share_5_7p5',
        'sibilance_contrast_5_7p5_db', 'sibilance_burst_rate_5_7p5_hz')],
    'D': ['d__dynamics_span', 'd__dynamics_iqr', 'd__dynamics_adjacent_change'],
    'R': ['r__ibi_cv', 'r__tempo_tv', 'r__tempo_entropy'],
    'P': ['p__section_duration_cv', 'p__section_duration_entropy', 'p__section_bars_cv',
          'p__section_bars_offmode_fraction', 'p__section_duration_median', 'p__section_bars_median'],
    'F': [f'F_phase_residual_cvar_{region}' for region in ('all', 'attack', 'sustain', 'decay')]
         + [f'F_group_delay_iqr_ms_{region}' for region in ('all', 'attack', 'sustain', 'decay')]
         + [f'F_group_delay_cross_band_iqr_ms_{region}' for region in ('all', 'attack', 'sustain', 'decay')]
         + ['F_phase_residual_cvar_decay_minus_sustain',
            'F_group_delay_iqr_ms_decay_minus_sustain',
            'F_group_delay_cross_band_iqr_ms_decay_minus_sustain'],
    'H': ['H_pitch_class_entropy_norm', 'H_pc_token_entropy_norm', 'H_chroma_path_change_median',
          'H_chroma_path_change_iqr', 'H_pc_path_step_median', 'H_pc_path_large_step_rate'],
    'SC': [f'SC_{lo}_{hi}hz_{name}_median' for lo, hi in ((80, 500), (500, 2000), (2000, 6000))
           for name in ('abs_iid_db', 'side_energy_fraction')],
}
FEATURE_NAMES = [name for family in ('S', 'D', 'R', 'P', 'F', 'H', 'SC') for name in FAMILY_CONFIG[family]]
METADATA_SCHEMA = ['id', 'label', 'source_group', 'group_id', 'component_id', 'role',
                   'duration_view_s', 'native_sample_rate_hz', 'input_sha256',
                   'waveform_float32_sha256']
LINEAGE_SCHEMA = ['id', 'cohort_row_sha256', 'origin_plan_row_sha256', 'screen_row_sha256',
                  'component_sha256', 'source_input', 'waveform_float32_sha256', 'fhsc_receipt',
                  'fhsc_receipt_payload_sha256', 'fhsc_frame_product', 'sdrp_receipt',
                  'sdrp_receipt_payload_sha256', 'schedule_input_position']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path,
            'absolute unredirected path required: ' + str(path))
    return path


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def binding(path, expected=None):
    path = safe_path(path)
    require(path.is_file() and not path.is_symlink(), 'regular bound file required: ' + str(path))
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'bound file SHA mismatch: ' + str(path))
    return result


def write_new(path, value):
    path = Path(path); temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scalar(value, name):
    require(value is None or type(value) in (int, float), 'predictor must be numeric or null: ' + name)
    require(value is None or math.isfinite(value), 'nonfinite predictor: ' + name)
    return value


def indexed(rows, name):
    require(isinstance(rows, list), name + ' must be a list')
    result = {}
    for row in rows:
        ident = row.get('id')
        require(isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', ident), 'unsafe ID in ' + name)
        require(ident not in result, 'duplicate ID in ' + name)
        result[ident] = row
    return result


def root_files(root, exclusions=('writer.lock', 'COMMIT.json')):
    return {path.relative_to(root).as_posix(): path for path in root.rglob('*')
            if path.is_file() and path.relative_to(root).as_posix() not in exclusions}


def product_binding(root, relative, declared, deep):
    require(set(declared) == {'bytes', 'sha256'} and type(declared['bytes']) is int
            and hash_string(declared['sha256']), 'malformed committed product binding: ' + relative)
    path = safe_path(root / relative)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == declared['bytes'],
            'committed product missing/size changed: ' + relative)
    if deep:
        require(digest(path) == declared['sha256'], 'committed product hash changed: ' + relative)
    return path


def validate_measurement_root(root, contract_sha, commit_sha, kind, *, deep, expected=EXPECTED):
    root = safe_path(root)
    contract_entry, commit_entry = binding(root / 'contract.json', contract_sha), binding(root / 'COMMIT.json', commit_sha)
    contract, commit = read_json(contract_entry['path']), read_json(commit_entry['path'])
    require(value_hash(contract) == contract_sha and value_hash(commit) == commit_sha, 'noncanonical source contract/COMMIT')
    require(kind in ('fhsc', 'sdrp'), 'unknown measurement backend')
    if kind == 'fhsc':
        # The frozen producer records VERSION in contract.json, not COMMIT.json.
        expected_commit = {'status': 'committed_native30_F_H_SC_measurements_not_admitted',
                           'contract_sha256': contract_sha, 'completed': expected,
                           'products': commit.get('products'),
                           'all_bound_inputs_and_products_end_rehashed': True,
                           'classifier_fits': 0, 'cohort_admitted': False,
                           'no_neural_inference': True, 'M_is_diagnostic_only': True,
                           'source_selection_changed': False}
        require(contract.get('version') == 'run_native30_fhsc_cohort_v1'
                and commit == expected_commit, 'source measurement COMMIT scope/status')
    else:
        backend = SDRP_BACKENDS.get(contract.get('version'))
        require(backend is not None, 'unapproved S/D/R/P backend')
        expected_commit = {'version': contract['version'],
                           'status': backend['commit_status'],
                           'contract_sha256': contract_sha, 'completed': expected,
                           'products': commit.get('products'),
                           'all_inputs_products_runtime_end_rehashed': True,
                           **backend['scope']}
        require(commit == expected_commit, 'source measurement COMMIT scope/status')
    files = root_files(root)
    require(set(files) == set(commit.get('products', {})), 'source COMMIT product inventory changed')
    for relative, declared in commit['products'].items():
        product_binding(root, relative, declared, deep)
    return contract, commit, contract_entry, commit_entry


def validate_schedule(root, draft_sha, commit_sha):
    root = safe_path(root); draft_entry = binding(root / 'schedule_draft.json', draft_sha)
    commit_entry = binding(root / 'COMMIT.json', commit_sha)
    draft, commit = read_json(draft_entry['path']), read_json(commit_entry['path'])
    require(value_hash(draft) == draft_sha and value_hash(commit) == commit_sha, 'noncanonical schedule draft/COMMIT')
    require(commit == {'classifier_fits': 0, 'draft_sha256': draft_sha, 'fitting_authorized': False,
                       'products': {'schedule_draft.json': draft_entry},
                       'status': 'committed_metadata_draft_not_evaluation_authorization',
                       'version': 'plan_native30_evaluation_schedule_v1'}, 'schedule COMMIT changed')
    require(draft.get('version') == 'plan_native30_evaluation_schedule_v1'
            and draft.get('status') == 'new_protocol_metadata_draft_not_frozen_for_evaluation'
            and draft.get('classifier_fits') == 0 and draft.get('fitting_authorized') is False,
            'schedule remains a nonauthorizing metadata draft')
    return draft, draft_entry, commit_entry


def authority_bindings(cohort_path, cohort_sha, plan_path, screen_path, fhsc_entries, sdrp_entries,
                       schedule_entries, sdrp_contract, *, plan_sha=PLAN_SHA, screen_sha=SCREEN_SHA,
                       code_pins=CODE_PINS, expanded_sha=EXPANDED_SHA):
    here = Path(__file__).resolve().parent
    result = {'cohort_contract': binding(cohort_path, cohort_sha),
              'origin_plan': binding(plan_path, plan_sha), 'screen': binding(screen_path, screen_sha),
              'fhsc_contract': fhsc_entries[0], 'fhsc_commit': fhsc_entries[1],
              'sdrp_contract': sdrp_entries[0], 'sdrp_commit': sdrp_entries[1],
              'schedule_draft': schedule_entries[0], 'schedule_commit': schedule_entries[1],
              'assembler': binding(Path(__file__).resolve()),
              'tests': binding(here / ('test_' + VERSION + '.py'))}
    for name, expected in code_pins.items():
        result['code_' + Path(name).stem] = binding(here / name, expected)
    core = sdrp_contract['bindings']['core']
    require(core['sha256'] == expanded_sha, 'S/D/R/P feature definition pin changed')
    result['code_expanded_feature_definitions'] = binding(core['path'], expanded_sha)
    backend = SDRP_BACKENDS[sdrp_contract['version']]
    runner = sdrp_contract['bindings']['runner']
    require(Path(runner['path']).name == sdrp_contract['version'] + '.py'
            and runner['sha256'] == backend['runner_sha256'], 'S/D/R/P backend runner pin changed')
    result['code_sdrp_backend'] = binding(runner['path'], backend['runner_sha256'])
    return result


def validate_feature_lists(fhsc_contract):
    require(len(FEATURE_NAMES) == 54 and {k: len(v) for k, v in FAMILY_CONFIG.items()} ==
            {'S': 15, 'D': 3, 'R': 3, 'P': 6, 'F': 15, 'H': 6, 'SC': 6}, '54 predictor definition')
    require(fhsc_contract.get('F_measure_names') == FAMILY_CONFIG['F']
            and fhsc_contract.get('H_measure_names') == FAMILY_CONFIG['H']
            and fhsc_contract.get('SC_selected_six') == FAMILY_CONFIG['SC'], 'F/H/SC contract feature lists changed')


def load_authorities(cohort_path, cohort_sha, plan_path, screen_path,
                     fhsc_root, fhsc_contract_sha, fhsc_commit_sha,
                     sdrp_root, sdrp_contract_sha, sdrp_commit_sha,
                     schedule_root, schedule_sha=SCHEDULE_SHA, schedule_commit_sha=SCHEDULE_COMMIT_SHA,
                     *, deep=False, expected=EXPECTED, plan_sha=PLAN_SHA, screen_sha=SCREEN_SHA,
                     code_pins=CODE_PINS, expanded_sha=EXPANDED_SHA):
    try:
        require(hash_string(cohort_sha) and hash_string(fhsc_contract_sha) and hash_string(fhsc_commit_sha)
                and hash_string(sdrp_contract_sha) and hash_string(sdrp_commit_sha), 'caller SHA pins required')
        cohort_entry = binding(cohort_path, cohort_sha); cohort = read_json(cohort_entry['path'])
        plan_entry, screen_entry = binding(plan_path, plan_sha), binding(screen_path, screen_sha)
        plan, screen = read_json(plan_entry['path']), read_json(screen_entry['path'])
        require(value_hash(cohort) == cohort_sha and cohort.get('version') == 'run_native30_fhsc_cohort_v1'
                and cohort.get('status') == 'frozen_before_any_measurement_audio_reads'
                and cohort.get('expected_count') == expected, 'prepared cohort schema/count')
        rows = indexed(cohort['rows'], 'cohort rows')
        require(len(rows) == expected and list(rows) == sorted(rows), 'exact ordered cohort rows')
        fhsc, _, fhsc_contract_entry, fhsc_commit_entry = validate_measurement_root(
            fhsc_root, fhsc_contract_sha, fhsc_commit_sha, 'fhsc', deep=deep, expected=expected)
        sdrp, _, sdrp_contract_entry, sdrp_commit_entry = validate_measurement_root(
            sdrp_root, sdrp_contract_sha, sdrp_commit_sha, 'sdrp', deep=deep, expected=expected)
        schedule, schedule_entry, schedule_commit_entry = validate_schedule(schedule_root, schedule_sha, schedule_commit_sha)
        require(fhsc.get('rows') == cohort['rows'] and fhsc.get('expected_count') == expected
                and fhsc.get('status') == 'frozen_before_any_measurement_audio_reads'
                and fhsc.get('output_root') == str(safe_path(fhsc_root)), 'FHSC contract/cohort join')
        sdrp_rows = [row['cohort_row'] for row in sdrp.get('rows', [])]
        sdrp_backend = SDRP_BACKENDS[sdrp['version']]
        require(sdrp_rows == cohort['rows'] and sdrp.get('expected_count') == expected
                and sdrp.get('status') == sdrp_backend['contract_status']
                and sdrp.get('output_root') == str(safe_path(sdrp_root)), 'SDRP contract/cohort join')
        validate_feature_lists(fhsc)
        schedule_ids = schedule.get('input_population', {}).get('ids')
        require(schedule.get('prepared_contract_sha256') == cohort_sha and schedule_ids == list(rows),
                'schedule/prepared cohort exact ordered ID join')
        planned, screened = indexed(plan['rows'], 'origin plan'), indexed(screen['rows'], 'screen')
        require(set(rows) <= set(planned) and set(rows) <= set(screened), 'plan/screen omit cohort IDs')
        components = {component['component_id']: component for component in screen['components']}
        require(len(components) == len(screen['components']), 'duplicate screen component')
        bindings = authority_bindings(cohort_path, cohort_sha, plan_path, screen_path,
            (fhsc_contract_entry, fhsc_commit_entry), (sdrp_contract_entry, sdrp_commit_entry),
            (schedule_entry, schedule_commit_entry), sdrp, plan_sha=plan_sha, screen_sha=screen_sha,
            code_pins=code_pins, expanded_sha=expanded_sha)
        return {'cohort': cohort, 'rows': rows, 'plan': planned, 'screen': screened,
                'components': components, 'fhsc_contract_sha': fhsc_contract_sha,
                'sdrp_contract_sha': sdrp_contract_sha,
                'fhsc_contract': fhsc, 'sdrp_contract': sdrp, 'schedule': schedule,
                'bindings': bindings, 'fhsc_root': safe_path(fhsc_root), 'sdrp_root': safe_path(sdrp_root),
                'sdrp_backend': sdrp_backend,
                'fhsc_commit_sha': fhsc_commit_sha, 'sdrp_commit_sha': sdrp_commit_sha,
                'schedule_root': safe_path(schedule_root), 'schedule_sha': schedule_sha,
                'schedule_commit_sha': schedule_commit_sha, 'expected': expected}
    finally:
        pass


def receipt(root, ident):
    path = safe_path(root / 'items' / (ident + '.json')); envelope = read_json(path)
    require(set(envelope) == {'payload', 'receipt_sha256'}
            and envelope['receipt_sha256'] == value_hash(envelope['payload'])
            and digest(path) == value_hash(envelope), 'source item receipt envelope/hash: ' + ident)
    return envelope['payload'], binding(path), envelope['receipt_sha256']


def assemble_rows(authorities):
    metadata, features, lineage = [], [], []
    schedule_position = {ident: index for index, ident in enumerate(authorities['schedule']['input_population']['ids'])}
    sdrp_mapped = {row['cohort_row']['id']: row for row in authorities['sdrp_contract']['rows']}
    for ident, row in authorities['rows'].items():
        plan, screen = authorities['plan'][ident], authorities['screen'][ident]
        require(all(row[key] == plan[key] == screen[key] for key in IDENTITY)
                and row['origin_plan_row_sha256'] == value_hash(plan)
                and row['screen_row_sha256'] == value_hash(screen)
                and row['component_sha256'] == value_hash(authorities['components'][row['component_id']])
                and row['role'] == 'development', 'immutable plan/screen/component/cohort join: ' + ident)
        rate = plan['source_origin']['sample_rate_hz']
        require(type(rate) is int and rate > 0 and plan['source_origin']['channels'] == 2, 'native source rate/channel: ' + ident)
        fhsc, fhsc_entry, fhsc_payload_sha = receipt(authorities['fhsc_root'], ident)
        sdrp, sdrp_entry, sdrp_payload_sha = receipt(authorities['sdrp_root'], ident)
        require(fhsc.get('status') == 'measured_not_admitted' and fhsc.get('row') == row
                and fhsc.get('row_sha256') == value_hash(row)
                and fhsc.get('contract_sha256') == authorities['fhsc_contract_sha']
                and fhsc.get('classifier_fits') == 0 and fhsc.get('cohort_admitted') is False,
                'FHSC receipt identity/scope: ' + ident)
        mapped = sdrp_mapped[ident]
        require(sdrp.get('status') == authorities['sdrp_backend']['item_status']
                and sdrp.get('row') == mapped and mapped['cohort_row'] == row
                and mapped['native_sample_rate_hz'] == rate
                and sdrp.get('contract_sha256') == authorities['sdrp_contract_sha']
                and all(sdrp.get(key) == value for key, value in authorities['sdrp_backend']['scope'].items()),
                'SDRP receipt identity/rate/scope: ' + ident)
        extractor_row = mapped['extractor_row']
        require(extractor_row['item_id'] == ident and extractor_row['standardized_path'] == row['input']['path']
                and extractor_row['native_sample_rate_hz'] == str(rate)
                and extractor_row['duration'] == '30' and extractor_row['audio_offset_s'] == '0',
                'SDRP extractor mapping: ' + ident)
        measurement, legacy = fhsc.get('measurement'), sdrp.get('legacy_result')
        require(isinstance(measurement, dict) and isinstance(legacy, dict),
                'measurement payload schema: ' + ident)
        require(legacy.get('status') == legacy.get('feature_status') == 'complete'
                and legacy.get('errors') == '' and legacy.get('item_id') == ident
                and legacy.get('label') == row['label'] and legacy.get('source_id') == row['source_group']
                and legacy.get('group_id') == row['group_id'] and legacy.get('duration_sec') == 30
                and legacy.get('native_sample_rate_hz') == rate, 'legacy result identity/status: ' + ident)
        selected = {}
        for name in FAMILY_CONFIG['S'] + FAMILY_CONFIG['D'] + FAMILY_CONFIG['R'] + FAMILY_CONFIG['P']:
            selected[name] = scalar(legacy.get(name), name)
        require({key for key in legacy if key.startswith(('s8__', 'd__', 'r__', 'p__'))}
                == set(FAMILY_CONFIG['S'] + FAMILY_CONFIG['D'] + FAMILY_CONFIG['R'] + FAMILY_CONFIG['P']),
                'S/D/R/P predictor set changed: ' + ident)
        for family in ('F', 'H'):
            require(isinstance(measurement.get(family), dict)
                    and set(FAMILY_CONFIG[family]) <= set(measurement[family]),
                    family + ' primary predictor missing: ' + ident)
            for name in FAMILY_CONFIG[family]:
                selected[name] = scalar(measurement[family][name], name)
        require(isinstance(measurement.get('SC'), dict)
                and isinstance(measurement['SC'].get('features'), dict)
                and set(measurement['SC']['features']) == set(FAMILY_CONFIG['SC']),
                'SC predictor set changed: ' + ident)
        for name in FAMILY_CONFIG['SC']:
            selected[name] = scalar(measurement['SC']['features'][name], name)
        require(list(selected) == FEATURE_NAMES, 'predictor ordering changed')
        input_evidence = sdrp['inputs']['input']
        require({key: input_evidence[key] for key in ('path', 'bytes', 'sha256')} == row['input']
                and input_evidence.get('format') == 'WAV' and input_evidence.get('subtype') == 'FLOAT'
                and input_evidence.get('sample_rate_hz') == 44100 and input_evidence.get('channels') == 2
                and input_evidence.get('frames') == 1323000 and input_evidence.get('finite') is True
                and input_evidence.get('waveform_float32_sha256') == row['waveform_float32_sha256']
                and measurement['analysis_view_audit']['source_audio_sha256'] == row['input']['sha256'],
                'source input FLOAT/SHA lineage changed: ' + ident)
        frame_path = safe_path(authorities['fhsc_root'] / 'frames' / (ident + '.sc_frames.npz'))
        frame_entry = binding(frame_path)
        require(fhsc['frame_product'] == frame_entry, 'FHSC frame binding changed: ' + ident)
        metadata.append({'id': ident, 'label': row['label'], 'source_group': row['source_group'],
                         'group_id': row['group_id'], 'component_id': row['component_id'], 'role': row['role'],
                         'duration_view_s': 30, 'native_sample_rate_hz': rate,
                         'input_sha256': row['input']['sha256'],
                         'waveform_float32_sha256': row['waveform_float32_sha256']})
        features.append({'id': ident, **selected})
        lineage.append({'id': ident, 'cohort_row_sha256': value_hash(row),
                        'origin_plan_row_sha256': row['origin_plan_row_sha256'],
                        'screen_row_sha256': row['screen_row_sha256'],
                        'component_sha256': row['component_sha256'], 'source_input': row['input'],
                        'waveform_float32_sha256': row['waveform_float32_sha256'],
                        'fhsc_receipt': fhsc_entry, 'fhsc_receipt_payload_sha256': fhsc_payload_sha,
                        'fhsc_frame_product': frame_entry, 'sdrp_receipt': sdrp_entry,
                        'sdrp_receipt_payload_sha256': sdrp_payload_sha,
                        'schedule_input_position': schedule_position[ident]})
    return metadata, features, lineage


def rehash_sources(authorities):
    fhsc, fhsc_commit, fhsc_contract_entry, fhsc_commit_entry = validate_measurement_root(
        authorities['fhsc_root'], authorities['fhsc_contract_sha'], authorities['fhsc_commit_sha'],
        'fhsc', deep=True, expected=authorities['expected'])
    sdrp, sdrp_commit, sdrp_contract_entry, sdrp_commit_entry = validate_measurement_root(
        authorities['sdrp_root'], authorities['sdrp_contract_sha'], authorities['sdrp_commit_sha'],
        'sdrp', deep=True, expected=authorities['expected'])
    schedule, schedule_entry, schedule_commit_entry = validate_schedule(
        authorities['schedule_root'], authorities['schedule_sha'], authorities['schedule_commit_sha'])
    require(fhsc == authorities['fhsc_contract'] and sdrp == authorities['sdrp_contract']
            and schedule == authorities['schedule'], 'source authority changed during assembly')
    return {'fhsc_contract': fhsc_contract_entry, 'fhsc_commit': fhsc_commit_entry,
            'fhsc_products_sha256': value_hash(fhsc_commit['products']),
            'sdrp_contract': sdrp_contract_entry, 'sdrp_commit': sdrp_commit_entry,
            'sdrp_products_sha256': value_hash(sdrp_commit['products']),
            'schedule_draft': schedule_entry, 'schedule_commit': schedule_commit_entry}


def publish(authorities, output):
    output = safe_path(output)
    require(not output.exists() and output.parent.is_dir(), 'new evaluation-input output directory required')
    source_roots = (authorities['fhsc_root'], authorities['sdrp_root'], authorities['schedule_root'])
    require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in source_roots),
            'evaluation-input output overlaps a committed source root')
    source_rehash_before = rehash_sources(authorities)
    metadata, features, lineage = assemble_rows(authorities)
    bindings = authorities['bindings']
    contract = {'version': VERSION, 'status': STATUS, 'expected_count': len(metadata),
                'cohort_contract': bindings['cohort_contract'], 'family_config': FAMILY_CONFIG,
                'feature_names': FEATURE_NAMES, 'feature_count': 54,
                'sdrp_backend': authorities['sdrp_contract']['version'],
                'source_counts': dict(sorted(Counter(row['source_group'] for row in metadata).items())),
                'bindings': bindings, 'provenance_bindings_sha256': value_hash(bindings),
                'metadata_schema': METADATA_SCHEMA,
                'feature_schema': ['id', *FEATURE_NAMES],
                'lineage_schema': LINEAGE_SCHEMA,
                'lineage_policy': 'per-ID source input, plan/screen/component, committed FHSC/SDRP receipts and SC frame binding',
                'scientific_missingness': 'JSON null retained; no imputation or availability filtering', **SCOPE}
    output.mkdir()
    for name, value in (('contract.json', contract), ('metadata.json', metadata),
                        ('features.json', features), ('lineage.json', lineage)):
        write_new(output / name, value)
    products = {name: binding(output / name) for name in ('contract.json', 'metadata.json', 'features.json', 'lineage.json')}
    source_rehash_end = rehash_sources(authorities)
    require(source_rehash_before == source_rehash_end, 'source graph changed during assembly')
    commit = {'version': VERSION, 'status': STATUS, 'expected_count': len(metadata), 'products': products,
              'all_source_COMMIT_products_rehashed_before_assembly': True,
              'source_rehash_before': source_rehash_before,
              'source_rehash_end': source_rehash_end, 'source_rehash_end_sha256': value_hash(source_rehash_end),
              'feature_files_read': len(metadata) * 2, 'audio_files_opened': 0, **SCOPE}
    write_new(output / 'COMMIT.json', commit)
    return verify_package(output, digest(output / 'COMMIT.json'))


def verify_package(root, commit_sha):
    root = safe_path(root)
    require({path.name for path in root.iterdir()} ==
            {'contract.json', 'metadata.json', 'features.json', 'lineage.json', 'COMMIT.json'},
            'evaluation-input package root inventory')
    commit_entry = binding(root / 'COMMIT.json', commit_sha); commit = read_json(commit_entry['path'])
    require(value_hash(commit) == commit_sha and commit.get('version') == VERSION and commit.get('status') == STATUS
            and commit.get('all_source_COMMIT_products_rehashed_before_assembly') is True
            and commit.get('audio_files_opened') == 0
            and commit.get('source_rehash_before') == commit.get('source_rehash_end')
            and value_hash(commit.get('source_rehash_end')) == commit.get('source_rehash_end_sha256')
            and all(commit.get(key) == value for key, value in SCOPE.items()), 'package COMMIT schema/scope/hash')
    products = {name: binding(root / name) for name in ('contract.json', 'metadata.json', 'features.json', 'lineage.json')}
    require(commit.get('products') == products, 'package product binding changed')
    contract, metadata, features, lineage = [read_json(root / name) for name in
        ('contract.json', 'metadata.json', 'features.json', 'lineage.json')]
    count = contract.get('expected_count')
    require(contract.get('version') == VERSION and contract.get('status') == STATUS
            and contract.get('family_config') == FAMILY_CONFIG and contract.get('feature_names') == FEATURE_NAMES
            and contract.get('feature_count') == 54 and all(contract.get(k) == v for k, v in SCOPE.items()),
            'package contract schema/scope')
    require(count == commit.get('expected_count') == len(metadata) == len(features) == len(lineage), 'package row counts')
    ids = [row['id'] for row in metadata]
    require(ids == sorted(ids) == [row['id'] for row in features] == [row['id'] for row in lineage]
            and len(ids) == len(set(ids)), 'package ordered unique ID join')
    require(contract.get('metadata_schema') == METADATA_SCHEMA
            and contract.get('feature_schema') == ['id', *FEATURE_NAMES]
            and contract.get('lineage_schema') == LINEAGE_SCHEMA,
            'package declared row schemas')
    require(all(set(row) == set(METADATA_SCHEMA) for row in metadata), 'package exact metadata row schema')
    require(all(set(row) == {'id', *FEATURE_NAMES} for row in features), 'package exact feature row schema')
    require(all(set(row) == set(LINEAGE_SCHEMA) for row in lineage), 'package exact lineage row schema')
    require(all(all(value is None or type(value) in (int, float) and math.isfinite(value)
                        for key, value in row.items() if key != 'id') for row in features), 'package predictor scalar/null values')
    bindings = contract.get('bindings', {})
    require(value_hash(bindings) == contract.get('provenance_bindings_sha256'), 'package provenance binding map hash')
    require(contract.get('cohort_contract') == bindings.get('cohort_contract')
            and contract.get('sdrp_backend') in SDRP_BACKENDS
            and contract.get('source_counts') == dict(sorted(Counter(row['source_group'] for row in metadata).items()))
            and contract.get('scientific_missingness') == 'JSON null retained; no imputation or availability filtering',
            'package contract content summary')
    require(commit['source_rehash_end']['fhsc_commit'] == bindings['fhsc_commit']
            and commit['source_rehash_end']['sdrp_commit'] == bindings['sdrp_commit']
            and commit['source_rehash_end']['schedule_commit'] == bindings['schedule_commit'],
            'package source end-rehash authority join')
    required_bindings = {'cohort_contract', 'origin_plan', 'screen', 'fhsc_contract', 'fhsc_commit',
                         'sdrp_contract', 'sdrp_commit', 'schedule_draft', 'schedule_commit',
                         'assembler', 'tests', 'code_expanded_feature_definitions', 'code_sdrp_backend',
                         *('code_' + Path(name).stem for name in CODE_PINS)}
    require(set(bindings) == required_bindings, 'package authority binding inventory')
    for entry in bindings.values():
        require(binding(entry['path']) == entry, 'package authority binding changed: ' + entry['path'])
    reconstructed = load_authorities(
        bindings['cohort_contract']['path'], bindings['cohort_contract']['sha256'],
        bindings['origin_plan']['path'], bindings['screen']['path'],
        Path(bindings['fhsc_contract']['path']).parent, bindings['fhsc_contract']['sha256'],
        bindings['fhsc_commit']['sha256'], Path(bindings['sdrp_contract']['path']).parent,
        bindings['sdrp_contract']['sha256'], bindings['sdrp_commit']['sha256'],
        Path(bindings['schedule_draft']['path']).parent, bindings['schedule_draft']['sha256'],
        bindings['schedule_commit']['sha256'], deep=True, expected=count,
        plan_sha=bindings['origin_plan']['sha256'], screen_sha=bindings['screen']['sha256'])
    require(reconstructed['bindings'] == bindings, 'package reconstruction authority binding changed')
    require(reconstructed['sdrp_contract']['version'] == contract['sdrp_backend'],
            'package S/D/R/P backend declaration changed')
    expected_metadata, expected_features, expected_lineage = assemble_rows(reconstructed)
    require(metadata == expected_metadata and features == expected_features and lineage == expected_lineage,
            'package rows do not replay from committed source receipts')
    source_end = commit['source_rehash_end']
    require(set(source_end) == {'fhsc_contract', 'fhsc_commit', 'fhsc_products_sha256',
                                'sdrp_contract', 'sdrp_commit', 'sdrp_products_sha256',
                                'schedule_draft', 'schedule_commit'},
            'package source end-rehash schema')
    require(source_end['fhsc_contract'] == bindings['fhsc_contract']
            and source_end['sdrp_contract'] == bindings['sdrp_contract']
            and source_end['schedule_draft'] == bindings['schedule_draft']
            and source_end['fhsc_products_sha256'] == value_hash(
                read_json(bindings['fhsc_commit']['path'])['products'])
            and source_end['sdrp_products_sha256'] == value_hash(
                read_json(bindings['sdrp_commit']['path'])['products'])
            and commit.get('feature_files_read') == count * 2,
            'package source end-rehash content/count')
    return {'contract': contract, 'metadata': metadata, 'features': features, 'lineage': lineage,
            'commit': commit_entry, 'products': products}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'cohort-sha256', 'plan', 'screen', 'fhsc-root', 'fhsc-contract-sha256',
                 'fhsc-commit-sha256', 'sdrp-root', 'sdrp-contract-sha256', 'sdrp-commit-sha256',
                 'schedule-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--schedule-sha256', default=SCHEDULE_SHA)
    parser.add_argument('--schedule-commit-sha256', default=SCHEDULE_COMMIT_SHA)
    parser.add_argument('--mode', choices=('preflight', 'assemble'), default='preflight')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    authorities = load_authorities(args.cohort, args.cohort_sha256, args.plan, args.screen,
        args.fhsc_root, args.fhsc_contract_sha256, args.fhsc_commit_sha256,
        args.sdrp_root, args.sdrp_contract_sha256, args.sdrp_commit_sha256,
        args.schedule_root, args.schedule_sha256, args.schedule_commit_sha256,
        deep=args.mode == 'assemble')
    if args.mode == 'preflight':
        result = {'status': 'metadata_preflight_only_no_feature_receipts_audio_or_fits',
                  'rows': len(authorities['rows']), 'feature_count': 54, **SCOPE}
    else:
        require(args.output is not None, '--output required for assemble')
        proof = publish(authorities, args.output)
        result = {'status': STATUS, 'rows': len(proof['metadata']), 'commit': proof['commit'], **SCOPE}
    print(canonical(result).decode().strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
