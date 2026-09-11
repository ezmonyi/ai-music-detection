#!/usr/bin/env python3
"""Prospective reserved GuitarSet BC measurement producer.

Preflight is metadata-only.  ``run`` requires a separately pinned parent freeze
and is deliberately unable to admit BC: it commits measurements and producer
gate arithmetic for an independent numerical auditor to replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np


VERSION = 'run_bc_guitarset_reserved_admission_v1'
FREEZE_VERSION = 'bc_guitarset_reserved_admission_parent_freeze_v1'
DRAFT_COMMIT_SHA = '5d0dcb01974cb08f52ca9d7898a38be4a727f5a6776dd8780ec544493c69f4ac'
DRAFT_BUILDER_SHA = 'f27be0836c664b9649aa83877457ccece961378c41da2e9807dd378ff2f27ccd'
ORIGINAL_PRODUCER_SHA = 'dcb8bfd41b89e3ab59bd241214ab64e36dbb7a5264409898bf3e423a03de8d6e'
CODEC_PRODUCER_SHA = '2c607b1175694b6966b3ce5200c6e306f20c40dd00de8f0229e6236387cc013e'
PINS = {'bicoherence_audio_v1': 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1',
        'bicoherence_scalar_v1': 'b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85',
        'bicoherence_primitive_v2': '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5',
        'probe_bc_codec_roundtrip_v1': 'dbf1f548356046568329574ecbd57ddf99aab3164d666e0fa0b02b563d724696',
        'draft_bicoherence_guitarset_pilot_v2':
            'c7368fb060c2583b107ecf3b01d0ad173160cd546c1ee6219933f5d934d9ce0e'}
RATE, SAMPLES, POOLS = 16000, 128000, 2
CONDITIONS = ('baseline', 'common_gain', 'polarity', 'closed_minus6db',
              'independent_minus6db', 'closed_0db', 'independent_0db')
SELECTED = ('baseline', 'closed_minus6db', 'independent_minus6db')
CODECS = ('mp3_128k', 'opus_96k')
EXPECTED = {'recordings': 90, 'float64': 630, 'precision': 270, 'codec': 540,
            'measurements': 1440, 'pools': 2880}
SCOPE = {'reserved_recordings': 90, 'development_audio_read': False,
         'unused_audio_read': False, 'classifier_fits': 0, 'model_scoring': False,
         'BC_admitted': False, 'independent_numerical_replay_passed': False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


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
    require(candidate.is_absolute() and candidate.is_file() and not candidate.is_symlink(),
            'safe non-symlink file required')
    path = candidate.resolve()
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'binding SHA mismatch: ' + str(path))
    return result


def read_json(path, expected=None):
    entry = binding(path, expected)
    return json.loads(Path(entry['path']).read_text(encoding='utf-8')), entry


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def write_json(path, value):
    write_new(path, canonical(value) + b'\n')


def module_from(path, name, expected):
    path = Path(path).resolve()
    binding(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'module loader')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(binding(path, expected)['sha256'] == expected, 'module changed during import')
    return module


def load_frozen(code_root):
    code_root = Path(code_root).resolve()
    require(code_root.is_dir() and not code_root.is_symlink(), 'safe code root')
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    draft = module_from(code_root / 'draft_bc_guitarset_reserved_admission_v1.py',
                        '_reserved_draft_v1', DRAFT_BUILDER_SHA)
    codec = module_from(code_root / 'run_bc_guitarset_codec_development_v1.py',
                        '_reserved_codec_dev_v1', CODEC_PRODUCER_SHA)
    modules = codec.load_frozen(code_root)
    for key, expected in PINS.items():
        require(binding(code_root / (key + '.py'), expected)['sha256'] == expected,
                'frozen module pin: ' + key)
    original = module_from(code_root / 'bicoherence_guitarset_pilot_v1.py',
                           '_reserved_original_v1', ORIGINAL_PRODUCER_SHA)
    return {'draft': draft, 'codec': codec, 'original': original, **modules}


def verify_draft(draft_dir, commit_sha, modules):
    require(commit_sha == DRAFT_COMMIT_SHA, 'unexpected admission draft COMMIT')
    proof = modules['draft'].verify_draft(draft_dir, commit_sha)
    document, _ = read_json(Path(draft_dir).resolve() / 'draft.json', proof['draft']['sha256'])
    require(document['conditions']['float64'] == list(CONDITIONS)
            and document['conditions']['float32_precision_control'] == list(SELECTED)
            and document['conditions']['codec_decoded'] == list(SELECTED)
            and document['conditions']['codecs'] == list(CODECS)
            and document['expected_denominators']['total_BC_measurements'] == EXPECTED['measurements']
            and document['expected_denominators']['total_BC_pools'] == EXPECTED['pools']
            and document['margins']['currently_frozen'] is False,
            'reserved draft scientific contract')
    return document, proof['commit']


def verify_freeze(draft_dir, draft_sha, freeze_path, freeze_sha, modules):
    document, draft_commit = verify_draft(draft_dir, draft_sha, modules)
    freeze, freeze_entry = read_json(freeze_path, freeze_sha)
    expected_scope = {'reserved_audio_access_authorized': True,
                      'reserved_BC_measurement_authorized': True,
                      'codec_encoding_authorized': True,
                      'classifier_fits_authorized': False,
                      'model_scoring_authorized': False,
                      'BC_admission_authorized': False,
                      'threshold_changes_authorized': False}
    codec_proof = modules['codec_probe'].verify_result(
        Path(document['authorities']['synthetic_codec_COMMIT']['path']).parent,
        document['authorities']['synthetic_codec_COMMIT']['sha256'])
    require(freeze.get('version') == FREEZE_VERSION
            and freeze.get('status') == 'authorized_reserved_measurement_not_admission'
            and freeze.get('draft_COMMIT') == draft_commit
            and freeze.get('draft') == binding(Path(draft_dir).resolve() / 'draft.json')
            and freeze.get('output_root') == document['output_root']
            and freeze.get('authorized_scope') == expected_scope
            and freeze.get('expected_measurements') == EXPECTED
            and freeze.get('storage_estimate') == storage_estimate()
            and freeze.get('runtime') == modules['codec'].runtime_snapshot()
            and freeze.get('toolchain') == codec_proof['results']['toolchain']
            and freeze.get('codec_recipes') == codec_proof['results']['codec_recipes'],
            'parent freeze schema/authority')
    for key in ('runner', 'runner_tests', 'independent_auditor'):
        require(freeze.get(key) == binding(freeze[key]['path']), 'freeze code binding: ' + key)
    require(freeze['runner'] == binding(Path(__file__).resolve()), 'freeze runner differs')
    return document, freeze, freeze_entry


def storage_estimate():
    # Accepted development packages used ~9.05 GB together; reserve is the same shape.
    return {'basis': 'accepted 90-record original plus codec product graphs with 25 percent margin',
            'required_free_bytes': 12 * 1024**3}


def rehash_reserved_sources(document):
    actual = {}
    for row in document['roster']:
        entries = {}
        for key in ('source_audio', 'source_annotation'):
            expected = row[key]
            entry = binding(expected['path'])
            require(entry == expected, 'reserved source bytes changed: ' + row['item_id'] + '/' + key)
            entries[key] = entry
        actual[row['item_id']] = entries
    return value_hash(actual)


def decode_crop(row, original):
    import soundfile as sf
    expected = row['source_audio']
    require(binding(expected['path']) == expected, 'reserved source binding')
    info = sf.info(expected['path'])
    historical = row['historical_native_decode']
    require(info.format == historical['format'] and info.subtype == historical['subtype']
            and info.samplerate == historical['sample_rate_hz'] and info.channels == 1
            and info.frames == historical['decoded_frames'], 'reserved native format')
    samples, rate = sf.read(expected['path'], dtype='float64', always_2d=True)
    require(rate == historical['sample_rate_hz'] and samples.shape == (info.frames, 1)
            and np.isfinite(samples).all(), 'reserved native decode')
    native = samples[:, 0]
    require(hashlib.sha256(native.astype('<f8', copy=False).tobytes()).hexdigest()
            == row['historical_native_float64_pcm_sha256'], 'reserved native PCM hash')
    crop, provenance = original.drafting.standardize(native, rate)
    plan = row['prospective_preprocessing']
    require(crop.dtype == np.float64 and crop.shape == (SAMPLES,) and np.isfinite(crop).all()
            and provenance['native_frames'] == plan['native_frames']
            and provenance['native_sample_rate_hz'] == plan['native_sample_rate_hz']
            and provenance['resampled_frames'] == plan['expected_resampled_frames']
            and provenance['crop_start'] == plan['crop_start_sample']
            and provenance['crop_stop_exclusive'] == plan['crop_stop_sample_exclusive']
            and provenance['crop_frames'] == SAMPLES and provenance['up'] == plan['up']
            and provenance['down'] == plan['down'] and provenance['window'] == plan['window']
            and provenance['padtype'] == plan['padtype'] and provenance['cval'] == plan['cval']
            and provenance['resample_scope'] == plan['resample_scope'],
            'reserved crop differs from draft')
    return crop, provenance


def measure_float64(samples, prefix, modules, construction_status):
    original, extractor, scalar = (modules['original'], modules['bicoherence_audio_v1'],
                                   modules['bicoherence_scalar_v1'])
    original.save_waveform(str(prefix) + '.wav', samples)
    measured = extractor.extract(samples, RATE)
    measured['metadata']['construction_status'] = construction_status
    original.save_arrays(str(prefix) + '.arrays.npz', measured['arrays'])
    write_json(str(prefix) + '.metadata.json', measured['metadata'])
    reduction = modules['codec'].reduction_signature(scalar.reduce_metadata(
        measured['metadata'], crop={'start_sample': 0, 'stop_sample_exclusive': SAMPLES,
                                    'source_resampled_samples': SAMPLES}))
    return {'status': 'measured_reserved_not_admitted',
            'waveform': binding(str(prefix) + '.wav'),
            'metadata': binding(str(prefix) + '.metadata.json'),
            'arrays': binding(str(prefix) + '.arrays.npz'), 'scalar': reduction,
            'descriptor': original.descriptor(measured['metadata'])}


def retained_failure(path, error, status='processing_failure_not_scientific_missingness'):
    write_json(path, {'status': status, 'type': type(error).__name__, 'message': str(error),
                      'traceback': traceback.format_exc()})
    return binding(path)


def dependency_codec_receipt(row, condition, destination, failure):
    destination.mkdir(parents=True, exist_ok=False)
    statuses = {'float32_control': {'attempted': False, 'status': 'skipped_dependency_failure'},
                **{codec: {'attempted': False, 'status': 'skipped_dependency_failure'}
                   for codec in CODECS}}
    result = {key: row[key] for key in ('item_id', 'player_id', 'score_id', 'performance',
              'style_from_score_prefix', 'split_role')} | {'condition': condition,
              'source_float64_wav': None, 'source_metadata': failure, 'source_arrays': failure,
              'representations': {'accepted_float64': None, 'float32_control': None,
                                  **{codec: None for codec in CODECS}},
              'comparisons': {'float32_minus_float64': {
                  'direction': 'float32_minus_float64',
                  'status': 'processing_failure_not_scientific_missingness',
                  'pool_denominator': POOLS, 'mask_transition_counts': None, 'scalar_delta': None},
                  'decoded_minus_float32': {codec: {
                      'direction': 'decoded_minus_float32',
                      'status': 'processing_failure_not_scientific_missingness',
                      'pool_denominator': POOLS, 'mask_transition_counts': None,
                      'scalar_delta': None} for codec in CODECS}},
              'artifacts': {'dependency_failure': failure}, 'measurement_status': statuses}
    write_json(destination / 'receipt.json', result)
    return result


def process_item(row, item_dir, modules, recipes):
    require(row['split_role'] == 'reserved', 'only reserved roster may reach producer')
    item_dir.mkdir(parents=True, exist_ok=False)
    identity = {key: row[key] for key in ('item_id', 'player_id', 'score_id', 'performance',
                                         'style_from_score_prefix', 'split_role')}
    result = {**identity, 'source_audio': row['source_audio'], 'conditions': {},
              'condition_status': {}, 'codec_receipts': {}}
    built = None
    try:
        crop, provenance = decode_crop(row, modules['original'])
        built = modules['original'].constructions(crop, row['item_id'])
        modules['original'].save_arrays(item_dir / 'construction.arrays.npz', built['arrays'])
        write_json(item_dir / 'construction.json', built['metadata'] | {
            'actual_preprocessing': provenance,
            'prospective_preprocessing': row['prospective_preprocessing'],
            'native_pcm_sha256': row['historical_native_float64_pcm_sha256']})
        result['construction'] = {'status': 'success',
            'metadata': binding(item_dir / 'construction.json'),
            'arrays': binding(item_dir / 'construction.arrays.npz')}
    except Exception as error:
        built = None
        failure = retained_failure(item_dir / 'construction.failure.json', error)
        result['construction'] = {'status': 'processing_failure', 'failure': failure}
    for condition in CONDITIONS:
        prefix = item_dir / 'float64' / condition
        try:
            require(built is not None, 'construction unavailable')
            prefix.parent.mkdir(parents=True, exist_ok=True)
            measured = measure_float64(built['waveforms'][condition], prefix, modules,
                                       built['metadata']['status'])
            result['conditions'][condition] = measured
            result['condition_status'][condition] = {'attempted': True, 'status': 'success'}
        except Exception as error:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            failure = retained_failure(str(prefix) + '.failure.json', error,
                'processing_failure_not_scientific_missingness' if built is not None else
                'skipped_due_to_construction_failure_not_scientific_missingness')
            result['conditions'][condition] = {'status': 'failure_or_skip', 'failure': failure,
                                               'scalar': None, 'descriptor': None}
            result['condition_status'][condition] = {
                'attempted': built is not None,
                'status': 'measurement_failure' if built is not None else 'skipped_dependency_failure'}
    for condition in SELECTED:
        measured = result['conditions'][condition]
        destination = item_dir / 'codec' / condition
        if measured['scalar'] is None:
            receipt = dependency_codec_receipt(row, condition, destination, measured['failure'])
        else:
            source_row = {**identity, 'condition': condition,
                'source_float64_wav': measured['waveform'],
                'source_metadata': measured['metadata'], 'source_arrays': measured['arrays'],
                'accepted_float64_reduction': measured['scalar']}
            receipt = modules['codec'].process_row(source_row, destination, modules, recipes)
        result['codec_receipts'][condition] = binding(destination / 'receipt.json')
        result.setdefault('_codec_values', []).append(receipt)
    write_json(item_dir / 'receipt.json', {key: value for key, value in result.items()
                                           if key != '_codec_values'})
    return result


def outcome_counts(items):
    float_status = [item['condition_status'][condition] for item in items for condition in CONDITIONS]
    codec_receipts = [receipt for item in items for receipt in item['_codec_values']]
    float_counts = {'expected': len(float_status),
        'attempted': sum(row['attempted'] for row in float_status),
        'successful': sum(row['status'] == 'success' for row in float_status),
        'failed_after_attempt': sum(row['attempted'] and row['status'] != 'success' for row in float_status),
        'skipped_before_attempt': sum(not row['attempted'] for row in float_status)}
    require(float_counts['expected'] == sum(float_counts[key] for key in
            ('successful', 'failed_after_attempt', 'skipped_before_attempt')), 'float64 accounting')
    result = {'float64': float_counts}
    controls = [r['measurement_status']['float32_control'] for r in codec_receipts]
    codecs = [r['measurement_status'][name] for r in codec_receipts for name in CODECS]
    def count(rows):
        return {'expected': len(rows), 'attempted': sum(r['attempted'] for r in rows),
                'successful': sum(r['status'] == 'success' for r in rows),
                'failed_after_attempt': sum(r['attempted'] and r['status'] != 'success' for r in rows),
                'skipped_before_attempt': sum(not r['attempted'] for r in rows)}
    result['float32_precision_control'], result['codec_decoded'] = count(controls), count(codecs)
    result['all'] = {key: sum(result[group][key] for group in result)
                     for key in ('expected', 'attempted', 'successful',
                                 'failed_after_attempt', 'skipped_before_attempt')}
    require(result['all']['expected'] == EXPECTED['measurements'], 'total measurement accounting')
    return result


def original_summary(items, modules):
    codec = modules['codec']
    baseline = [item['conditions']['baseline']['scalar'] for item in items]
    levels = {}
    for level in ('minus6db', '0db'):
        rows = []
        for item in items:
            compared = codec.condition_contrast(item['conditions']['closed_' + level]['scalar'],
                                                item['conditions']['independent_' + level]['scalar'])
            rows.append({key: item[key] for key in
                         ('item_id', 'player_id', 'score_id', 'performance')} | compared)
        levels[level] = codec.summarize_contrasts(rows) | {'per_recording': rows}
    complete = all(item['conditions'][condition]['descriptor'] is not None
                   for item in items for condition in CONDITIONS)
    historical = None
    if complete:
        records = []
        for item in items:
            conditions = {condition: item['conditions'][condition]['descriptor']
                          for condition in CONDITIONS}
            records.append({key: item[key] for key in
                            ('item_id', 'player_id', 'score_id', 'performance',
                             'style_from_score_prefix')} | {'conditions': conditions,
                    'nuisance': {name: modules['original'].nuisance(conditions['baseline'], conditions[name])
                                 for name in ('common_gain', 'polarity')}})
        raw = modules['original'].aggregate(records)
        historical = {'version': VERSION, 'dataset_role': 'reserved',
            'reused_numerical_implementation': {
                'version': raw['version'], 'sha256': ORIGINAL_PRODUCER_SHA,
                'development_policy_not_represented_as_reserved_scope': True},
            **{key: raw[key] for key in
               ('recording_denominator', 'independent_replay_status', 'baseline', 'levels',
                'nuisance', 'per_recording')}}
    return {'baseline_coverage': codec.coverage_summary(baseline), 'levels': levels,
            'full_grid_historical_summary': historical,
            'full_grid_summary_available': complete}


def add_check(checks, name, observed, relation, threshold, passed):
    checks.append({'name': name, 'observed': observed, 'relation': relation,
                   'threshold': threshold, 'passed': bool(passed)})


def contrast_checks(checks, prefix, summary, minimum):
    add_check(checks, prefix + '.paired_pool_support', summary['paired_covered_pools'],
              '>=', 171, summary['paired_covered_pools'] >= 171)
    for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
        entry = summary['metrics'][metric]
        rows = entry['per_recording'] if 'per_recording' in entry else None
        # summarize_contrasts keeps per-recording rows at the enclosing level.
        rows = summary.get('per_recording', []) if rows is None else rows
        values = [row[metric] for row in rows if row[metric] is not None]
        add_check(checks, prefix + '.' + metric + '.covered_recordings', len(values),
                  '>=', 86, len(values) >= 86)
        add_check(checks, prefix + '.' + metric + '.equal_score',
                  entry['equal_score']['equal_group_mean'], '>=', minimum,
                  entry['equal_score']['equal_group_mean'] is not None and
                  entry['equal_score']['equal_group_mean'] >= minimum)
        add_check(checks, prefix + '.' + metric + '.positive_recordings',
                  sum(value > 0 for value in values), '>=', 72,
                  sum(value > 0 for value in values) >= 72)
        player = [row['mean'] for row in entry['equal_player_secondary']['rows']]
        add_check(checks, prefix + '.' + metric + '.all_players_positive', player,
                  'all >', 0, len(player) == 3 and all(value is not None and value > 0 for value in player))
        performances = [entry['performance_strata'][label]['mean'] for label in ('comp', 'solo')]
        add_check(checks, prefix + '.' + metric + '.both_performances_positive', performances,
                  'all >', 0, all(value is not None and value > 0 for value in performances))


def per_stratum_coverage(rows, getter):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['player_id'], row['performance'])].append(getter(row))
    result = []
    for (player, performance), reductions in sorted(groups.items()):
        valid = [value for value in reductions if value is not None]
        eligible = sum(value['eligible_pool_count'] for value in valid)
        result.append({'player_id': player, 'performance': performance,
                       'pool_denominator': len(reductions) * POOLS,
                       'eligible_pools': eligible,
                       'eligible_fraction': eligible / (len(reductions) * POOLS)})
    return result


def producer_margin_checks(items, original, codec_summary, counts, document):
    """Descriptive producer checks only; independent replay owns the gate."""
    checks = []
    baseline = original['baseline_coverage']
    add_check(checks, 'original.baseline.eligible_pools', baseline['eligible_pools'],
              '>=', 144, baseline['eligible_pools'] >= 144)
    add_check(checks, 'original.baseline.covered_recording_scalars',
              baseline['covered_recording_scalars'], '>=', 81,
              baseline['covered_recording_scalars'] >= 81)
    strata = per_stratum_coverage(items, lambda item: item['conditions']['baseline']['scalar'])
    add_check(checks, 'original.baseline.every_player_performance_fraction', strata,
              'all >=', .8, len(strata) == 6 and all(row['eligible_fraction'] >= .8 for row in strata))
    for level in ('minus6db', '0db'):
        contrast_checks(checks, 'original.' + level, original['levels'][level], .20)
    historical = original['full_grid_historical_summary']
    for condition in ('common_gain', 'polarity'):
        nuisance = historical['nuisance'][condition] if historical else None
        add_check(checks, 'original.' + condition + '.full_grid_and_target_mask_exact',
                  nuisance, 'exact', True, nuisance is not None and
                  nuisance['grid_eligibility_agreement_count'] == nuisance['grid_denominator'] and
                  nuisance['target_eligibility_agreement'] == nuisance['pool_denominator'])
        maximum = nuisance['maximum_finite_b2_difference'] if nuisance else None
        add_check(checks, 'original.' + condition + '.maximum_common_finite_b2_error',
                  maximum, '<=', 1e-10, maximum is not None and maximum <= 1e-10)
    codec_receipts = [receipt for item in items for receipt in item['_codec_values']]
    baseline_receipts = [row for row in codec_receipts if row['condition'] == 'baseline']
    for name in CODECS:
        coverage = codec_summary['baseline']['representation_coverage'][name]
        add_check(checks, name + '.baseline.eligible_pools', coverage['eligible_pools'],
                  '>=', 144, coverage['eligible_pools'] >= 144)
        add_check(checks, name + '.baseline.covered_recording_scalars',
                  coverage['covered_recording_scalars'], '>=', 81,
                  coverage['covered_recording_scalars'] >= 81)
        codec_strata = per_stratum_coverage(
            baseline_receipts, lambda row, key=name: row['representations'].get(key))
        add_check(checks, name + '.baseline.every_player_performance_fraction', codec_strata,
                  'all >=', .8, len(codec_strata) == 6 and
                  all(row['eligible_fraction'] >= .8 for row in codec_strata))
        transitions = codec_summary['baseline']['codec_pool_transition_counts'][name]
        agreement = ((transitions['eligible_to_eligible'] + transitions['missing_to_missing']) /
                     transitions['comparison_denominator'] / POOLS)
        add_check(checks, name + '.baseline.mask_agreement_fraction', agreement,
                  '>=', .95, agreement >= .95 and transitions['processing_failures'] == 0)
        for condition in SELECTED:
            delta = codec_summary['scalar_delta_distributions']['codec_decoded_minus_float32'][name][
                'by_condition'][condition]
            add_check(checks, name + '.' + condition + '.median_absolute_scalar_error',
                      delta['median_absolute_delta'], '<=', .02,
                      delta['median_absolute_delta'] is not None and
                      delta['median_absolute_delta'] <= .02)
            add_check(checks, name + '.' + condition + '.p95_absolute_scalar_error',
                      delta['p95_absolute_delta'], '<=', .10,
                      delta['p95_absolute_delta'] is not None and delta['p95_absolute_delta'] <= .10)
        contrast_checks(checks, name + '.minus6db',
                        codec_summary['condition_contrasts'][name], .20)
    add_check(checks, 'processing.all_measurements_successful', counts['all'], 'successful ==',
              EXPECTED['measurements'], counts['all']['successful'] == EXPECTED['measurements'])
    return {'status': 'producer_checks_only_not_authoritative_not_admitted',
            'margins_source_sha256': value_hash(document['margins']),
            'check_count': len(checks), 'passed_checks': sum(row['passed'] for row in checks),
            'all_producer_checks_passed': all(row['passed'] for row in checks),
            'independent_replay_required': True, 'BC_admitted': False, 'checks': checks}


def products(root):
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.name not in ('COMMIT.json', 'FAILED.json')}


def assemble_summary(items, document, modules):
    require(len(items) == 90 and [item['item_id'] for item in items] ==
            [row['item_id'] for row in document['roster']], 'complete reserved item order')
    codec_receipts = [receipt for item in items for receipt in item['_codec_values']]
    raw_codec_summary = modules['codec'].assemble_summary(codec_receipts)
    codec_summary = {'version': VERSION, 'dataset_role': 'reserved',
        'status': ('reserved_precision_and_codec_measurements_complete_not_admitted'
                   if raw_codec_summary['new_BC_measurements']['successful'] == 810 else
                   'reserved_precision_and_codec_measurements_with_retained_failures_not_admitted'),
        'reused_numerical_implementation': {
            'version': raw_codec_summary['version'], 'sha256': CODEC_PRODUCER_SHA,
            'development_scope_fields_removed': True},
        **{key: raw_codec_summary[key] for key in ('denominators', 'new_BC_measurements',
            'processing_failures', 'baseline', 'scalar_delta_distributions',
            'condition_contrasts')}}
    counts = outcome_counts(items)
    clean_items = [{key: value for key, value in item.items() if key != '_codec_values'}
                   for item in items]
    original = original_summary(items, modules)
    producer_checks = producer_margin_checks(items, original, codec_summary, counts, document)
    return {'version': VERSION,
            'status': ('reserved_measurements_complete_pending_independent_replay_not_admitted'
                       if counts['all']['successful'] == EXPECTED['measurements'] else
                       'reserved_measurements_with_retained_failures_pending_replay_not_admitted'),
            **SCOPE, 'measurement_counts': counts,
            'expected_denominators': document['expected_denominators'],
            'original_float64': original,
            'precision_and_codec': codec_summary,
            'margins': document['margins'],
            'producer_margin_checks': producer_checks,
            'producer_gate_status': 'descriptive_checks_only_not_authoritative_independent_replay_required',
            'independent_replay_required_before_gate': True,
            'final_admission_decision': None, 'per_recording': clean_items}


def run(draft_dir, draft_sha, freeze_path, freeze_sha, code_root):
    modules = load_frozen(code_root)
    document, freeze, freeze_entry = verify_freeze(
        draft_dir, draft_sha, freeze_path, freeze_sha, modules)
    output = Path(document['output_root'])
    require(output.is_absolute() and not output.exists(), 'new absolute output required')
    for root in (Path(draft_dir).resolve(), Path(code_root).resolve()):
        require(output != root and output not in root.parents and root not in output.parents,
                'source/output overlap')
    require(shutil.disk_usage(output.parent).free >= storage_estimate()['required_free_bytes'],
            'insufficient durable capacity')
    codec_proof = modules['codec_probe'].verify_result(
        Path(document['authorities']['synthetic_codec_COMMIT']['path']).parent,
        document['authorities']['synthetic_codec_COMMIT']['sha256'])
    recipes = dict(codec_proof['results']['codec_recipes'])
    recipes['_ffmpeg'] = codec_proof['results']['toolchain']['ffmpeg']['path']
    source_graph_start = rehash_reserved_sources(document)
    output.mkdir(parents=True, exist_ok=False)
    items = []
    try:
        write_json(output / 'frozen_draft.json', document)
        write_json(output / 'parent_freeze.json', freeze)
        write_json(output / 'storage_estimate.json', storage_estimate())
        for index, row in enumerate(document['roster']):
            items.append(process_item(row, output / 'items' / row['item_id'], modules, recipes))
            print(f'BC reserved {index + 1}/90 {row["item_id"]}', file=sys.stderr, flush=True)
        summary = assemble_summary(items, document, modules)
        write_json(output / 'summary.json', summary)
        end_document, end_freeze, end_entry = verify_freeze(
            draft_dir, draft_sha, freeze_path, freeze_sha, modules)
        codec_end = modules['codec_probe'].verify_result(
            Path(document['authorities']['synthetic_codec_COMMIT']['path']).parent,
            document['authorities']['synthetic_codec_COMMIT']['sha256'])
        require(end_document == document and end_freeze == freeze and end_entry == freeze_entry
                and codec_end == codec_proof and rehash_reserved_sources(document) == source_graph_start,
                'authority/source/toolchain changed during run')
        inventory = products(output)
        status = ('committed_reserved_measurements_complete_pending_independent_replay_not_admitted'
                  if summary['measurement_counts']['all']['successful'] == EXPECTED['measurements'] else
                  'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted')
        commit = {'version': VERSION, 'status': status, **SCOPE,
            'draft_COMMIT': binding(Path(draft_dir).resolve() / 'COMMIT.json'),
            'parent_freeze': freeze_entry, 'source_graph_sha256': source_graph_start,
            'measurement_counts': summary['measurement_counts'],
            'expected_measurements': EXPECTED, 'summary_sha256': value_hash(summary),
            'producer_gate_status': summary['producer_gate_status'],
            'independent_auditor': freeze['independent_auditor'],
            'products': inventory, 'products_sha256': value_hash(inventory)}
        write_json(output / 'COMMIT.json', commit)
        return {'commit': binding(output / 'COMMIT.json'), 'summary': summary}
    except BaseException as error:
        if not (output / 'FAILED.json').exists():
            write_json(output / 'FAILED.json', {'status': 'failed_no_COMMIT_outputs_preserved',
                'type': type(error).__name__, 'message': str(error),
                'completed_recordings': len(items), 'traceback': traceback.format_exc()})
        raise


def verify_result(output, commit_sha, code_root):
    """Integrity/summary replay; this is not the independent numerical audit."""
    candidate = Path(output)
    require(candidate.is_absolute() and candidate.is_dir() and not candidate.is_symlink(),
            'safe result root required')
    output = candidate.resolve()
    require(not (output / 'FAILED.json').exists(), 'FAILED result cannot be verified or complete')
    commit, commit_entry = read_json(output / 'COMMIT.json', commit_sha)
    require(commit.get('version') == VERSION and commit.get('status') in (
        'committed_reserved_measurements_complete_pending_independent_replay_not_admitted',
        'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted')
        and all(commit.get(key) == value for key, value in SCOPE.items())
        and commit.get('producer_gate_status') ==
            'descriptive_checks_only_not_authoritative_independent_replay_required',
        'reserved result COMMIT schema/scope')
    inventory = products(output)
    require(commit.get('products') == inventory
            and commit.get('products_sha256') == value_hash(inventory), 'result product graph')
    summary, summary_entry = read_json(output / 'summary.json')
    require(commit.get('summary_sha256') == value_hash(summary)
            and commit.get('measurement_counts') == summary.get('measurement_counts')
            and summary.get('final_admission_decision') is None
            and summary.get('BC_admitted') is False, 'result summary binding/scope')
    modules = load_frozen(code_root)
    draft_dir = Path(commit['draft_COMMIT']['path']).parent
    document, freeze, freeze_entry = verify_freeze(
        draft_dir, commit['draft_COMMIT']['sha256'], commit['parent_freeze']['path'],
        commit['parent_freeze']['sha256'], modules)
    require(freeze_entry == commit['parent_freeze']
            and json.loads((output / 'frozen_draft.json').read_text()) == document
            and json.loads((output / 'parent_freeze.json').read_text()) == freeze
            and rehash_reserved_sources(document) == commit['source_graph_sha256'],
            'result frozen authorities/source graph')
    items = []
    for row in document['roster']:
        item_path = output / 'items' / row['item_id'] / 'receipt.json'
        item = json.loads(item_path.read_text())
        require({key: item[key] for key in ('item_id', 'player_id', 'score_id', 'performance',
                'style_from_score_prefix', 'split_role')} == {key: row[key] for key in
                ('item_id', 'player_id', 'score_id', 'performance',
                 'style_from_score_prefix', 'split_role')}, 'result item identity')
        item['_codec_values'] = []
        for condition in SELECTED:
            receipt_path = output / 'items' / row['item_id'] / 'codec' / condition / 'receipt.json'
            require(item['codec_receipts'][condition] == binding(receipt_path),
                    'result codec receipt binding')
            item['_codec_values'].append(json.loads(receipt_path.read_text()))
        items.append(item)
    all_success = summary['measurement_counts']['all']['successful'] == EXPECTED['measurements']
    all_construction_success = all(item['construction']['status'] == 'success' for item in items)
    require((commit['status'] ==
             'committed_reserved_measurements_complete_pending_independent_replay_not_admitted')
            == (all_success and all_construction_success),
            'complete status requires every measurement and construction persistence')
    require(assemble_summary(items, document, modules) == summary,
            'result summary reconstruction')
    return {'status': 'verified_product_and_summary_graph_not_numerical_replay_not_admitted',
            'commit': commit_entry, 'summary': summary_entry, 'products': inventory,
            'measurement_counts': summary['measurement_counts'], 'BC_admitted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'run', 'verify'), default='preflight')
    parser.add_argument('--draft-dir', required=True)
    parser.add_argument('--draft-COMMIT-sha256', default=DRAFT_COMMIT_SHA)
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--parent-freeze')
    parser.add_argument('--parent-freeze-sha256')
    parser.add_argument('--result-COMMIT-sha256')
    args = parser.parse_args()
    modules = load_frozen(args.code_root)
    if args.mode == 'preflight':
        document, commit = verify_draft(args.draft_dir, args.draft_COMMIT_sha256, modules)
        print(json.dumps({'status': 'metadata_only_preflight_no_reserved_audio_access',
                          'draft_COMMIT': commit, 'reserved_rows': len(document['roster']), **SCOPE},
                         ensure_ascii=False, sort_keys=True))
        return
    if args.mode == 'verify':
        require(args.result_COMMIT_sha256, 'verify requires exact result COMMIT SHA')
        document, _ = verify_draft(args.draft_dir, args.draft_COMMIT_sha256, modules)
        print(json.dumps(verify_result(document['output_root'], args.result_COMMIT_sha256,
                                       args.code_root), ensure_ascii=False, sort_keys=True))
        return
    require(args.parent_freeze and args.parent_freeze_sha256,
            'run requires exact parent freeze path and SHA')
    print(json.dumps(run(args.draft_dir, args.draft_COMMIT_sha256,
                         args.parent_freeze, args.parent_freeze_sha256, args.code_root),
                     ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
