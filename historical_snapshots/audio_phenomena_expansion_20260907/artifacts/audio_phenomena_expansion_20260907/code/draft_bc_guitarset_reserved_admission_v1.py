#!/usr/bin/env python3
"""Build a nonauthorizing, metadata-only GuitarSet reserved admission draft."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


VERSION = 'bc_guitarset_reserved_admission_draft_v1'
SOURCE_DRAFT_SHA = '7632e91ddb3b2c25e0ebebf9b10357b7c1ddfaa96d4e79c603b28f5ef4ee97eb'
SOURCE_DRAFT_COMMIT_SHA = '56b31dc87c592534831e66f725d93c05548a2d40d07a179c70605f00f060667e'
SOURCE_COMMIT_SHA = 'aa33f0dfc55e3bbc6f48e618d1203c29e258608c309412d333f06c6e671b73dc'
DEVELOPMENT_COMMIT_SHA = 'cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c'
DEVELOPMENT_AUDIT_SHA = '61a926a92c764d6852608f9829395382e6593e56544d1434ec0f28af509ffc93'
CODEC_DEVELOPMENT_COMMIT_SHA = '9b38fb43457dbd3da03e46f12ff3e24e68f58e03e02eea6d6bfdd8d74fdc378b'
CODEC_DEVELOPMENT_AUDIT_SHA = 'aca0a828ed85f093158f532e5fb150217fa800f97e9bfe2e21cfb96ed85e77b3'
CODEC_SOURCE_COMMIT_SHA = '9e07622b5c45179e5086d66f2c6697c4a76bbe7e3b5b39c82ecaea0cb936909b'
CODEC_RESULTS_SHA = 'aec3e575a7283ee4c2ef33bf26d9a0f21de8fd4e2d6cd5a026e984da8847a0f8'
PINS = {'original_freeze': '769fe12892f07013361771bbabfb422a3a9ea23406029c910657efd9eef604a5',
    'codec_development_draft_COMMIT': '2f53c482e7934d23c120ab20343f3e66f377efd3626a8466608944e4f4ad30ed',
    'codec_development_freeze': '9405118a484d5578b73caff964538571920ae786ea9329e8aa846408ef85e53b',
    'original_producer': 'dcb8bfd41b89e3ab59bd241214ab64e36dbb7a5264409898bf3e423a03de8d6e',
    'codec_producer': '2c607b1175694b6966b3ce5200c6e306f20c40dd00de8f0229e6236387cc013e',
    'codec_auditor': 'd5d308e1dbc87365c04edc36bfc15e5377c5dd89dbe5d8db1da937fd79f99425',
    'audio': 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1',
    'scalar': 'b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85',
    'primitive': '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5',
    'codec_probe': 'dbf1f548356046568329574ecbd57ddf99aab3164d666e0fa0b02b563d724696'}
RATE = 16000
SAMPLES = 128000
POOLS = 2
CONDITIONS = ('baseline', 'common_gain', 'polarity', 'closed_minus6db',
              'independent_minus6db', 'closed_0db', 'independent_0db')
CODEC_CONDITIONS = ('baseline', 'closed_minus6db', 'independent_minus6db')
CODECS = ('mp3_128k', 'opus_96k')
SCOPE = {'reserved_recordings': 90, 'development_audio_read': False,
         'reserved_audio_read': False, 'unused_audio_read': False,
         'reserved_BC_measured': False, 'classifier_fits': 0,
         'model_scoring': False, 'BC_admitted': False,
         'thresholds_tuned_on_reserved': False, 'parent_freeze_required': True}


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


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def crop_plan(native_frames, native_rate):
    require(type(native_frames) is int and native_frames > 0
            and type(native_rate) is int and native_rate > 0, 'native support')
    divisor = math.gcd(native_rate, RATE)
    up, down = RATE // divisor, native_rate // divisor
    resampled = (native_frames * up + down - 1) // down
    require(resampled >= SAMPLES, 'reserved source too short for prospective fixed crop')
    start = (resampled - SAMPLES) // 2
    return {'native_frames': native_frames, 'native_sample_rate_hz': native_rate,
            'resampling_applied': native_rate != RATE, 'up': up, 'down': down,
            'window': ['kaiser', 5.0], 'padtype': 'constant', 'cval': 0.0,
            'resample_scope': 'entire_native_recording_before_crop',
            'expected_resampled_frames': resampled,
            'crop_rule': 'start=floor((resampled_frames-128000)/2)',
            'crop_start_sample': start, 'crop_stop_sample_exclusive': start + SAMPLES,
            'leading_context_excluded_samples': start,
            'trailing_context_excluded_samples': resampled - start - SAMPLES,
            'condition_samples': SAMPLES, 'complete_pool_samples': 2 * 64000,
            'complete_pool_count': POOLS, 'measurement_tail_discarded_samples': 0,
            'padding_samples': 0}


def reserved_roster(source_draft):
    require(source_draft.get('version') == 'draft_bicoherence_guitarset_pilot_v2'
            and source_draft['source']['commit']['sha256'] == SOURCE_COMMIT_SHA,
            'frozen source draft/version')
    split = source_draft['split']
    require(split['role_counts'] == {'development': 90, 'reserved': 90, 'unused': 180},
            'crossed split role counts')
    rows = [row for row in split['rows'] if row['split_role'] == 'reserved']
    require(len(rows) == 90 and [row['item_id'] for row in rows] == sorted(
        row['item_id'] for row in rows), 'reserved roster order/count')
    result = []
    source_root = Path(source_draft['source']['root'])
    for row in rows:
        source = row['source_record']
        require(source['item_id'] == row['item_id'] and source['class_label'] is None
                and source['role'] == 'external_measurement_control_only'
                and source['player_id'] == row['player_id']
                and source['score_id'] == row['score_id']
                and source['performance'] == row['performance'], 'reserved source identity')
        decoded = source['audio']['decoded']
        require(decoded['channels'] == 1 and decoded['nonfinite_samples'] == 0
                and decoded['float64_samples_checked_finite'] == decoded['decoded_frames']
                and decoded['header_frames'] == decoded['decoded_frames'], 'historical full-decode receipt')
        audio_product = source['audio']['materialized']
        annotation_product = source['annotation']['materialized']
        plan = crop_plan(decoded['decoded_frames'], decoded['sample_rate_hz'])
        result.append({key: row[key] for key in
            ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix',
             'split_role')} | {
            'source_audio': {'path': str(source_root / source['audio']['materialized_path']),
                             **audio_product},
            'source_annotation': {'path': str(source_root / source['annotation']['materialized_path']),
                                  **annotation_product},
            'historical_native_decode': decoded,
            'historical_native_float64_pcm_sha256': decoded['decoded_pcm_sha256'],
            'prospective_preprocessing': plan,
            'preprocessing_executed_in_this_draft': False,
            'condition_construction_seed_text': 'BC-GuitarSet-injection-20260907|' + row['item_id']})
    require({row['player_id'] for row in result} == set(split['reserved_player_ids'])
            and {row['score_id'] for row in result} == set(split['reserved_score_ids'])
            and Counter(row['performance'] for row in result) == {'comp': 45, 'solo': 45},
            'reserved crossed roster identities')
    return result


def margins():
    return {
        'provenance': 'development-informed engineering margins; not externally validated',
        'lock_required_before_reserved_measurement': True,
        'currently_frozen': False,
        'baseline_observability': {
            'eligible_target_pools_min': 144, 'target_pool_denominator': 180,
            'available_recording_scalars_min': 81, 'recording_denominator': 90,
            'each_player_by_performance_eligible_pool_fraction_min': 0.80,
            'strata_denominator': 6, 'scores_retained_with_denominators': True},
        'injection_support_each_level': {
            'levels': ['minus6db', '0db'], 'same_pool_pairs_min': 171,
            'same_pool_pair_denominator': 180, 'available_recording_contrasts_min': 86,
            'recording_denominator': 90},
        'sensitivity_each_level_both_endpoints': {
            'levels': ['minus6db', '0db'],
            'endpoints': ['operational_condition_median_difference',
                          'mean_of_same_pool_closed_minus_independent_differences'],
            'equal_score_aggregate_min_inclusive': 0.20,
            'strictly_positive_recordings_min': 72, 'recording_denominator': 90,
            'every_equal_player_aggregate_strictly_positive': True,
            'each_performance_aggregate_strictly_positive': True},
        'gain_and_polarity': {
            'conditions': ['common_gain', 'polarity'],
            'eligibility_mask_exact_agreement': True,
            'maximum_common_finite_absolute_b2_error_max_inclusive': 1e-10,
            'scopes': ['fixed_target', 'complete_228_cell_descriptive_grid'],
            'processing_failures_may_not_be_scientific_missingness': True},
        'each_codec': {
            'codecs': list(CODECS), 'baseline_target_mask_agreement_fraction_min': 0.95,
            'baseline_pool_comparison_denominator_each_codec': 180,
            'all_mask_transition_directions_reported': True,
            'scalar_error_conditions_evaluated_separately': list(CODEC_CONDITIONS),
            'scalar_pair_recording_denominator_per_condition_each_codec': 90,
            'median_absolute_scalar_error_max_inclusive': 0.02,
            'p95_absolute_scalar_error_max_inclusive': 0.10,
            'p95_quantile': {'method': 'linear', 'index': '(n-1)*0.95'},
            'original_float64_baseline_observability_gate_must_pass': True,
            'each_decoded_codec_baseline_observability': {
                'eligible_target_pools_min': 144, 'target_pool_denominator': 180,
                'available_recording_scalars_min': 81, 'recording_denominator': 90,
                'each_player_by_performance_eligible_pool_fraction_min': 0.80,
                'strata_denominator': 6},
            'minus6db_codec_injection_support_and_both_sensitivity_endpoints_must_pass': True,
            'minus6db_support_and_sensitivity_requirements': 'same unchanged requirements above'},
        'admission_rule': 'all declared requirements must pass after independent replay; any failure rejects admission without retuning'}


def expected_denominators():
    return {'reserved_recordings': 90,
        'float64_conditions': 630, 'float64_condition_pools': 1260,
        'selected_float32_precision_controls': 270, 'precision_control_pools': 540,
        'codec_encodes': 540, 'codec_decodes': 540,
        'codec_decoded_BC_measurements': 540, 'codec_decoded_pools': 1080,
        'total_BC_measurements': 1440, 'total_BC_pools': 2880,
        'precision_scalar_comparisons': 270, 'codec_scalar_comparisons': 540,
        'baseline_target_pool_denominator': 180,
        'paired_injection_pool_denominator_per_level': 180,
        'recording_contrast_denominator_per_level': 90,
        'codec_baseline_mask_comparisons': 360,
        'codec_minus6_recording_contrasts': 180,
        'gain_grid_cell_comparisons': 90 * 2 * 228,
        'polarity_grid_cell_comparisons': 90 * 2 * 228}


def build(artifact_root, output_root, bindings_extra):
    root = Path(artifact_root).resolve()
    source_draft, source_draft_binding = read_json(
        root / 'preregistration/bicoherence_guitarset_pilot_v1/draft.json', SOURCE_DRAFT_SHA)
    _, source_draft_commit = read_json(
        root / 'preregistration/bicoherence_guitarset_pilot_v1/COMMIT.json', SOURCE_DRAFT_COMMIT_SHA)
    source_commit_path = root / 'manifests/guitarset_validated_source_v2/COMMIT.json'
    if not source_commit_path.is_file():
        source_commit_path = Path(source_draft['source']['commit']['path'])
    source_commit, source_commit_binding = read_json(source_commit_path, SOURCE_COMMIT_SHA)
    codec_commit, codec_commit_binding = read_json(
        root / 'results/bc_codec_roundtrip_v1/COMMIT.json', CODEC_SOURCE_COMMIT_SHA)
    codec_results, codec_results_binding = read_json(
        root / 'results/bc_codec_roundtrip_v1/results.json', CODEC_RESULTS_SHA)
    codec_development_commit_path = (
        root / 'results/bicoherence_guitarset_codec_development_v1/COMMIT.json')
    if not codec_development_commit_path.is_file():
        codec_development_commit_path = (Path(source_draft['source']['root']).parent /
            'bicoherence_guitarset_codec_development_v1/COMMIT.json')
    require(source_commit.get('status') == 'committed' and len(source_commit.get('products', {})) == 725
            and source_draft['source']['products'] == source_commit['products'],
            'validated source inventory mirror')
    require({key: source_commit_binding[key] for key in ('bytes', 'sha256')} ==
            {key: source_draft['source']['commit'][key] for key in ('bytes', 'sha256')},
            'validated source COMMIT binding')
    require(codec_commit.get('results_sha256') == CODEC_RESULTS_SHA
            and codec_commit['products']['results.json'] == {
                'bytes': codec_results_binding['bytes'], 'sha256': codec_results_binding['sha256']},
            'synthetic codec results bound by pinned COMMIT')
    roster = reserved_roster(source_draft)
    authorities = {
        'source_crossed_draft': source_draft_binding,
        'source_crossed_draft_COMMIT': source_draft_commit,
        'validated_source_COMMIT': source_commit_binding,
        'development_result_COMMIT_mirror': binding(
            root / 'audit/guitarset_bc_producer_COMMIT_v1.json', DEVELOPMENT_COMMIT_SHA),
        'development_independent_audit': binding(
            root / 'audit/guitarset_bc_independent_replay_v1.json', DEVELOPMENT_AUDIT_SHA),
        'codec_development_result_COMMIT': binding(
            codec_development_commit_path, CODEC_DEVELOPMENT_COMMIT_SHA),
        'codec_development_independent_audit': binding(
            root / 'audit/bc_codec_independent_actual_v1.json', CODEC_DEVELOPMENT_AUDIT_SHA),
        'synthetic_codec_COMMIT': codec_commit_binding,
        'synthetic_codec_results': codec_results_binding,
        'original_development_parent_freeze': binding(
            root / 'preregistration/bicoherence_guitarset_parent_freeze_v1.json', PINS['original_freeze']),
        'codec_development_draft_COMMIT': binding(
            root / 'preregistration/bc_guitarset_codec_development_v1_r2/COMMIT.json',
            PINS['codec_development_draft_COMMIT']),
        'codec_development_parent_freeze': binding(
            root / 'preregistration/bc_guitarset_codec_development_parent_freeze_v1.json',
            PINS['codec_development_freeze']),
        'admission_review': binding(root / 'BC_ADMISSION_PROTOCOL_REVIEW_EN.md'),
        'development_results_report': binding(root / 'BC_GUITARSET_DEVELOPMENT_RESULTS_EN.md'),
        'codec_development_results_report': binding(root / 'BC_CODEC_DEVELOPMENT_RESULTS_EN.md'),
        'original_development_producer': binding(
            root / 'code/bicoherence_guitarset_pilot_v1.py', PINS['original_producer']),
        'codec_development_producer': binding(
            root / 'code/run_bc_guitarset_codec_development_v1.py', PINS['codec_producer']),
        'codec_development_auditor': binding(
            root / 'code/audit_bc_guitarset_codec_development_v1.py', PINS['codec_auditor']),
        'bicoherence_audio_v1': binding(root / 'code/bicoherence_audio_v1.py', PINS['audio']),
        'bicoherence_scalar_v1': binding(root / 'code/bicoherence_scalar_v1.py', PINS['scalar']),
        'bicoherence_primitive_v2': binding(root / 'code/bicoherence_primitive_v2.py', PINS['primitive']),
        'codec_probe': binding(root / 'code/probe_bc_codec_roundtrip_v1.py', PINS['codec_probe']),
    }
    authorities.update(bindings_extra)
    return {'version': VERSION,
        'status': 'prospective_reserved_admission_protocol_draft_not_frozen_not_authorized',
        **SCOPE, 'source_root': source_draft['source']['root'],
        'validated_source_COMMIT': source_draft['source']['commit'],
        'reserved_roster_source': 'unchanged crossed split frozen before development BC',
        'reserved_player_ids': source_draft['split']['reserved_player_ids'],
        'reserved_score_ids': source_draft['split']['reserved_score_ids'],
        'reserved_roster_sha256': value_hash(roster),
        'style_recording_counts': dict(sorted(Counter(
            row['style_from_score_prefix'] for row in roster).items())),
        'conditions': {'float64': list(CONDITIONS), 'float32_precision_control': list(CODEC_CONDITIONS),
                       'codec_decoded': list(CODEC_CONDITIONS), 'codecs': list(CODECS)},
        'construction': {'background_gain': 0.25, 'sample_rate_hz': RATE,
            'common_gain_factor': 0.1,
            'condition_samples': SAMPLES, 'pool_samples': 64000, 'pool_count': POOLS,
            'target_bins': [32, 48, 80], 'target_hz': [500, 750, 1250],
            'energy_fraction_min_inclusive': 1e-6,
            'phase_knot_count': 65, 'phase_knot_interval_seconds': 0.125,
            'injection_tone_amplitude': 0.2, 'injection_levels_db': [-6, 0],
            'mixture_normalization': False, 'mixture_clipping': False,
            'codec_input': 'explicit FLOAT64-to-FLOAT32 value cast of complete condition waveform',
            'codec_trim_pad_align_gain': False,
            'operation_order': [
                'verify bound native source bytes then decode mono FLOAT64',
                'resample entire native waveform to 16000 Hz with the declared polyphase policy',
                'take the deterministic centered 128000-sample crop without padding',
                'construct and retain all seven FLOAT64 conditions',
                'measure unchanged BC on every FLOAT64 condition',
                'cast each selected complete condition by value to FLOAT32 and measure the precision control',
                'write FLOAT32 WAV, encode with the bound recipe, decode to FLOAT32 16000-Hz mono WAV',
                'reject any decoded length other than 128000; do not trim, pad, align, or change gain',
                'measure unchanged BC on each valid decoded waveform']},
        'codec_recipes': codec_results['codec_recipes'],
        'expected_denominators': expected_denominators(), 'margins': margins(),
        'failure_policy': {'all_attempts_and_failures_retained': True,
            'processing_failure_is_not_scientific_missingness': True,
            'exact_decoded_length_required': SAMPLES, 'padding_or_trimming_forbidden': True,
            'denominators_never_reduced': True, 'any_unresolved_processing_failure_fails_admission': True,
            'qualified_COMMIT_with_failures_allowed_for_audit': True,
            'restart_or_retry_may_not_overwrite_outputs_or_change_waveforms': True},
        'aggregation': {'operational': 'closed condition median minus independent condition median',
            'historical': 'mean of same-pool closed minus independent differences',
            'order': 'pool -> recording -> equal score; equal player secondary; comp and solo strata',
            'missing_pairs_imputed': False, 'positive_means_strictly_greater_than_zero': True},
        'independent_replay': {'required_before_gate_evaluation': True,
            'must_not_import_reserved_producer_or_frozen_extractor_primitive_scalar': True,
            'reconstruct_native_decode_resample_crop_and_all_seven_float64_conditions': True,
            'reconstruct_exact_float32_cast_and_validate_codec_commands_decoded_lengths': True,
            'numerical_scope': 'fixed target for all 1440 measurements; full 228-cell grid for baseline/gain/polarity nuisance checks',
            'reconstruct_all_denominators_aggregates_margins_and_failure_states': True,
            'COMMIT_and_all_products_rehashed_before_and_after': True},
        'decision': {'reserved_measurement_requires_separate_parent_freeze': True,
            'gate_evaluation_occurs_only_after_independent_replay': True,
            'margins_may_not_change_after_reserved_access': True,
            'failure_action': 'BC not admitted; do not inspect AI/Human classifier performance',
            'pass_action': 'permits only a separately frozen grouped AI/Human BC evaluation'},
        'output_root': str(Path(output_root)), 'output_must_be_new_and_exclusive': True,
        'authorities': authorities, 'roster': roster,
        'implementation_gaps': [
            'reserved producer and synthetic tests do not yet exist',
            'independent reserved auditor including nuisance full-grid replay does not yet exist',
            'storage estimate and durable-capacity check are not yet frozen',
            'parent freeze publisher and exact run host/runtime receipt are not yet created',
            'reserved crop PCM hashes and condition product hashes cannot exist before authorized execution',
            'gate evaluator and nonoverwriting decision receipt do not yet exist',
            'no reserved numerical response has been externally validated because none has been accessed']}


def publish(document, directory):
    directory = Path(directory).resolve()
    require(not directory.exists(), 'new draft directory required')
    output = Path(document['output_root'])
    require(directory != output and directory not in output.parents and output not in directory.parents,
            'draft/output overlap')
    directory.mkdir(parents=True, exist_ok=False)
    write_json_new(directory / 'draft.json', document)
    commit = {'version': VERSION, 'status': 'committed_nonauthorizing_reserved_admission_draft',
              **SCOPE, 'draft': binding(directory / 'draft.json'),
              'validated_source_COMMIT_sha256': SOURCE_COMMIT_SHA,
              'development_result_COMMIT_sha256': DEVELOPMENT_COMMIT_SHA,
              'codec_development_result_COMMIT_sha256': CODEC_DEVELOPMENT_COMMIT_SHA,
              'reserved_audio_accessed': False, 'numerical_gate_evaluated': False}
    write_json_new(directory / 'COMMIT.json', commit)
    return {'draft': binding(directory / 'draft.json'), 'commit': binding(directory / 'COMMIT.json')}


def verify_draft(directory, commit_sha):
    """Verify the committed draft without opening any roster audio or annotations."""
    directory = Path(directory).resolve()
    commit, commit_binding = read_json(directory / 'COMMIT.json', commit_sha)
    require(commit.get('version') == VERSION
            and commit.get('status') == 'committed_nonauthorizing_reserved_admission_draft'
            and commit.get('reserved_audio_accessed') is False
            and commit.get('numerical_gate_evaluated') is False,
            'nonauthorizing draft COMMIT')
    draft_path = directory / 'draft.json'
    require(commit.get('draft') == binding(draft_path), 'draft binding')
    document, _ = read_json(draft_path, commit['draft']['sha256'])
    require(document.get('version') == VERSION
            and document.get('status') ==
                'prospective_reserved_admission_protocol_draft_not_frozen_not_authorized',
            'draft version/status')
    require(all(document.get(key) == value for key, value in SCOPE.items()), 'draft scope')
    require(document.get('margins') == margins()
            and document.get('expected_denominators') == expected_denominators(),
            'draft margins/denominators')
    roster = document.get('roster')
    require(isinstance(roster, list) and len(roster) == 90
            and [row.get('item_id') for row in roster] == sorted(
                row.get('item_id') for row in roster)
            and {row.get('split_role') for row in roster} == {'reserved'}
            and document.get('reserved_roster_sha256') == value_hash(roster),
            'draft reserved roster')
    require(all(row.get('preprocessing_executed_in_this_draft') is False
                for row in roster), 'draft must not claim reserved preprocessing')
    authorities = document.get('authorities')
    require(isinstance(authorities, dict) and authorities, 'draft authorities')
    for name, expected in authorities.items():
        require(expected == binding(expected['path']), 'authority changed: ' + name)
    return {'status': 'verified_nonauthorizing_reserved_admission_draft',
            'commit': commit_binding, 'draft': commit['draft'],
            'reserved_rows': len(roster), **SCOPE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--draft-dir', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--tests', required=True)
    parser.add_argument('--mode', choices=('preflight', 'publish'), default='preflight')
    args = parser.parse_args()
    extras = {'draft_builder': binding(Path(__file__).resolve()),
              'draft_builder_tests': binding(args.tests), 'protocol': binding(args.protocol)}
    document = build(args.artifact_root, args.output_root, extras)
    result = {'status': 'metadata_only_reserved_roster_preflight_no_audio_access',
              'reserved_rows': len(document['roster']), **SCOPE}
    if args.mode == 'publish':
        published = publish(document, args.draft_dir)
        result = verify_draft(args.draft_dir, published['commit']['sha256'])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
