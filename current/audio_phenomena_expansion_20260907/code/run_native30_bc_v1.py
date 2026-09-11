"""CPU-only fixed BC transfer producer. Default preflight reads metadata only.

Prepare is not execution authority. Run requires a separately caller-pinned parent
freeze. Scientific nulls retain rows; failures and orphan products are preserved.
No classifier, neural inference, codec construction, or source-dependent view.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import importlib
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent
VERSION = 'run_native30_bc_v1'
FEATURE = 'BC_b2_500_750_1250hz_center8s_median'
FREEZE_VERSION = 'native30-bc-parent-freeze-v1'
SCOPE = {'cpu_only': True, 'classifier_fits': 0, 'model_scoring': False,
         'classifier_admission': False, 'neural_inference': False, 'recovery_applied': False}
AUDIT_SHA = '637f07ec893ed64a85e4fcd4971da3b177ec4f9efa83315b76e4441cae5ec30e'
DECISION_SHA = '5f49573923d9ccbc1fcfc6e5b04a32b90b85c3f929e589d2f69d2969d020ca82'
RESERVED_COMMIT_SHA = 'a96b7ecc93f4bd0c44954b6c64afc0a9b4c782482ec025291a248dd35862b2e4'
PINS = {
    'materialize_native30_new1695_v1': '165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a',
    'run_native30_fhsc_cohort_v1': 'dc26465459df133b89d6023bb1c732e91ebab5b63ff02143280b50c383e7b3cc',
    'run_native30_inference_batches_v1': '84dce45426596888b3a4e3b20e4ad1c61dc976645506f269c7ec7990b12f3933',
    'validate_guitarset_archives_v2': 'cda25ca63e382a1363ade813cd114838099bee55ee60f346888f2b1cd2ff6d6c',
    'draft_bicoherence_guitarset_pilot_v2': 'c7368fb060c2583b107ecf3b01d0ad173160cd546c1ee6219933f5d934d9ce0e',
    'bicoherence_primitive_v2': '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5',
    'bicoherence_audio_v1': 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1',
    'bicoherence_scalar_v1': 'b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85',
    'summarize_bc_reserved_admission_v1': 'a1f736a3517335a2e6b07aa53d8ad7bd8ba272ab5cecca356ebaf8c9349b6280',
}
INPUT_FORMAT = {'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100, 'channels': 2, 'frames': 1323000}
VIEW = {'channel_policy': 'promote_float64_then_arithmetic_mean', 'global_dc_subtraction': False,
        'source_resampled_samples': 480000, 'resample_up': 160, 'resample_down': 441,
        'resample_window': ['kaiser', 5.0], 'resample_padtype': 'constant', 'resample_cval': 0.0,
        'resample_scope': 'whole_native30_before_crop', 'analysis_sample_rate_hz': 16000,
        'crop_start_sample': 176000, 'crop_stop_sample_exclusive': 304000,
        'condition_local_start_sample': 0, 'condition_local_stop_sample_exclusive': 128000,
        'baseline_gain': 0.25, 'pool_samples': 64000, 'pool_count': 2, 'discarded_tail_samples': 0,
        'analysis_duration_s': 8, 'cohort_duration_s': 30, 'normalization': False, 'clipping': False,
        'padding_short_inputs': False, 'channel_fallback': False,
        'intermediate_pcm_encoding': 'contiguous_little_endian_float64_mono'}
CROP = {'start_sample': 176000, 'stop_sample_exclusive': 304000, 'source_resampled_samples': 480000}
DECISION_SCOPE = {'external_measurement_gate_accepted': True,
    'prospective_native30_implementation_and_synthetic_tests_authorized': True,
    'actual_cohort_measurement_authorized_by_this_file': False,
    'classifier_fits_authorized_by_this_file': False, 'model_scoring_authorized_by_this_file': False,
    'threshold_changes_authorized': False, 'separate_measurement_freeze_required': True,
    'separate_fit_freeze_required': True}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(name):
    path = HERE / (name + '.py')
    require(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == PINS[name],
            'pinned source changed: ' + name)
    value = importlib.import_module(name)
    require(Path(value.__file__).resolve() == path, 'import origin changed: ' + name)
    return value


base = module('materialize_native30_new1695_v1')
physical = module('run_native30_fhsc_cohort_v1')
graph = module('run_native30_inference_batches_v1')


def numerical_modules():
    module('validate_guitarset_archives_v2')
    draft = module('draft_bicoherence_guitarset_pilot_v2')
    primitive = module('bicoherence_primitive_v2')
    audio = module('bicoherence_audio_v1')
    scalar = module('bicoherence_scalar_v1')
    require(audio.estimate is primitive.estimate, 'audio primitive callable changed')
    return draft, audio, scalar


def runtime_snapshot():
    import numpy as np
    import scipy
    import soundfile as sf
    result = base.runtime_binding(SimpleNamespace(np=np, scipy=scipy, sf=sf))
    require((np.__version__, scipy.__version__, sf.__version__) == ('1.26.4', '1.17.1', '0.14.0'),
            'fixed numerical runtime versions')
    files, origins = {}, {}
    for name in ('numpy', 'scipy', 'soundfile', 'cffi', '_cffi_backend'):
        loaded = importlib.import_module(name); path = Path(loaded.__file__).resolve()
        origins[name] = str(path); paths = [path]
        if hasattr(loaded, '__path__'):
            paths += list(path.parent.rglob('*'))
        for extra in (path.parent.with_name(name + '.libs'), path.parent.with_name(name + '.dylibs')):
            if extra.is_dir():
                paths += list(extra.rglob('*'))
        for current in sorted(set(paths)):
            if current.is_file() and '__pycache__' not in current.parts and current.suffix not in {'.pyc', '.pyo'}:
                files[str(current.resolve())] = base.file_binding(current.resolve())
    result.update(package_files=files, import_origins=origins,
                  thread_environment={k: os.environ.get(k) for k in
                                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    return result


def pinned_json(entry):
    physical.safe_path(entry['path']); graph._binding_matches(entry, audio=False)
    require(Path(entry['path']).suffix == '.json', 'JSON authority required')
    return base.read_json(entry['path'])


def accepted_gate(request):
    """Read terminal authority JSON only; no reserved arrays or audio are read."""
    reporter = module('summarize_bc_reserved_admission_v1')
    audit, decision, commit = (request[k] for k in ('audit', 'decision', 'producer_commit'))
    require(audit['sha256'] == AUDIT_SHA and decision['sha256'] == DECISION_SHA
            and commit['sha256'] == RESERVED_COMMIT_SHA, 'approved gate/decision exact pins required')
    decision_value = pinned_json(decision)
    for entry in (audit, commit):
        pinned_json(entry)
    require(decision_value.get('version') == 'bc_native30_transfer_decision_v1'
            and decision_value.get('status') == 'external_measurement_gate_accepted_prospective_transfer_implementation_only'
            and decision_value.get('scope') == DECISION_SCOPE and decision_value.get('prospective_cohort_rows') == 3830
            and decision_value.get('prospective_feature') == FEATURE
            and decision_value.get('independent_audit') == {k: audit[k] for k in ('path', 'sha256')}
            and decision_value.get('producer_commit_sha256') == commit['sha256']
            and decision_value.get('tolerances_and_scientific_margins_unchanged') is True,
            'implementation-only decision scope/feature/audit join')
    proof = reporter.load_sources(commit['path'], commit['sha256'], audit['path'], audit['sha256'])
    require(proof['complete_measurements'] is True and proof['scientific_gate_satisfied'] is True,
            'independent audit alone is not a passed scientific gate')
    return {'audit': audit, 'decision': decision, 'producer_commit': commit,
            'summary': proof['summary_binding'], 'scientific_gate_satisfied': True,
            'actual_execution_authorized': False}


def load_cohort(request, *, expected=physical.EXPECTED, plan_sha=physical.PLAN_SHA, screen_sha=physical.SCREEN_SHA):
    cohort = pinned_json(request['cohort_contract'])
    require(cohort.get('version') == physical.VERSION and
            cohort.get('status') == 'frozen_before_any_measurement_audio_reads' and
            base.value_hash(cohort) == request['cohort_contract']['sha256'] and
            cohort.get('input_format') == INPUT_FORMAT and cohort.get('expected_count') == expected['total'] and
            cohort.get('source_counts') == expected['sources'] and cohort.get('human') == expected['human'] and
            cohort.get('ai') == expected['ai'], 'prepared cohort schema/accounting')
    graph.validate_rows(cohort['rows'], expected)
    upstream = {k: base.file_binding(v['path']) for k, v in cohort['upstream_commits'].items()}
    require(all(v['sha256'] == cohort['upstream_commits'][k]['sha256'] for k, v in upstream.items()), 'intake COMMIT changed')
    graph.verify_source_graph({'plan': request['plan'], 'screen': request['screen'], 'upstream_commits': upstream},
                              cohort, physical, expected, plan_sha=plan_sha, screen_sha=screen_sha)
    for key, entry in cohort['bindings'].items():
        require(key == entry['path'], 'cohort binding path index')
        graph._binding_matches(entry, audio=False)
    plan_rows = physical.indexed(pinned_json(request['plan'])['rows'], 'origin plan')
    origins = {}
    for row in cohort['rows']:
        require(cohort['bindings'].get(row['input']['path']) == row['input'], 'input omitted from cohort graph')
        original = plan_rows[row['id']]
        require(base.value_hash(original) == row['origin_plan_row_sha256'] and
                original['source_origin']['sample_rate_hz'] in (44100, 48000), 'native sample-rate SHA join')
        origins[row['id']] = original['source_origin']
    return cohort, origins


def prepare(request, output):
    """Metadata only. CLI has no reduced-count or bypass switches."""
    require(set(request) == {'cohort_contract', 'plan', 'screen', 'audit', 'decision', 'producer_commit'}, 'request keys')
    gate = accepted_gate(request)
    cohort, origins = load_cohort(request)
    numerical_modules()
    output = physical.safe_path(output)
    bindings = {name: base.file_binding(HERE / (name + '.py')) for name in PINS}
    bindings.update(runner=base.file_binding(HERE / (VERSION + '.py')),
                    tests=base.file_binding(HERE / ('test_' + VERSION + '.py')))
    require(output.parent.is_dir() and not any(Path(e['path']).is_relative_to(output)
            for e in [*request.values(), *bindings.values(), *cohort['bindings'].values()]) and
            not any(output.is_relative_to(Path(root)) or Path(root).is_relative_to(output)
                    for root in [cohort['output_root'], *cohort['upstream_inventories']]), 'output overlaps immutable input')
    return {'version': VERSION, 'status': 'prepared_metadata_only_not_execution_authority',
            'output_root': str(output), 'request': request, 'gate': gate, 'code': bindings,
            'runtime': runtime_snapshot(), 'rows': cohort['rows'], 'native_origins': origins,
            'expected_count': 3830, 'source_counts': physical.SOURCES, 'input_format': INPUT_FORMAT,
            'view': VIEW, 'feature_names': [FEATURE], 'family': 'BC', 'audio_reads_in_preflight': 0,
            'grid_role': 'diagnostic_only_not_predictors', 'missingness': 'scalar_null_retains_row', **SCOPE}


def validate_freeze(path, digest):
    entry = base.file_binding(physical.safe_path(path))
    require(physical.hash_string(digest) and entry['sha256'] == digest, 'caller parent-freeze pin required')
    freeze = pinned_json(entry)
    require(freeze.get('version') == FREEZE_VERSION and freeze.get('status') == 'parent_frozen_for_native30_BC_measurement'
            and freeze.get('measurement_authorized') is True and all(freeze.get(k) == v for k, v in SCOPE.items()),
            'separate parent measurement freeze required; no fit authority')
    contract = pinned_json(freeze['prepared_contract'])
    require(base.value_hash(contract) == freeze['prepared_contract']['sha256'], 'canonical prepared contract')
    require(freeze.get('runner') == base.file_binding(HERE / (VERSION + '.py')) and
            freeze.get('tests') == base.file_binding(HERE / ('test_' + VERSION + '.py')) and
            freeze.get('decision') == contract['request']['decision'] and freeze.get('view') == VIEW and
            freeze.get('runtime') == contract['runtime'], 'freeze code/decision/view/runtime binding')
    require(prepare(contract['request'], contract['output_root']) == contract, 'prepared contract does not replay')
    require(not Path(entry['path']).is_relative_to(Path(contract['output_root'])) and
            not Path(freeze['prepared_contract']['path']).is_relative_to(Path(contract['output_root'])),
            'authorities must be outside output')
    return {'version': VERSION, 'status': 'parent_authorized_BC_measurement_only', 'prepared': contract,
            'parent_freeze': entry, 'prepared_contract': freeze['prepared_contract'], **SCOPE}


def check_inputs(run, *, audio):
    prepared = run['prepared']
    require(base.file_binding(run['parent_freeze']['path']) == run['parent_freeze'] and
            base.file_binding(run['prepared_contract']['path']) == run['prepared_contract'], 'execution authority changed')
    require(prepare(prepared['request'], prepared['output_root']) == prepared, 'runtime/metadata graph changed')
    if audio:
        cohort = pinned_json(prepared['request']['cohort_contract'])
        for entry in cohort['bindings'].values():
            graph._binding_matches(entry, audio=True)


@contextmanager
def source_locks(prepared):
    cohort = pinned_json(prepared['request']['cohort_contract'])
    with ExitStack() as stack:
        for root in sorted(cohort['upstream_inventories']):
            path = Path(root) / 'writer.lock'
            if path.exists():
                require(path.is_file() and not path.is_symlink(), 'unsafe source writer lock')
                stream = stack.enter_context(path.open('rb'))
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def pcm_hash(values):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()


def analysis_view(stereo):
    import numpy as np
    require(isinstance(stereo, np.ndarray) and stereo.shape == (1323000, 2) and stereo.dtype == np.float64
            and np.isfinite(stereo).all(), 'finite float64 exact30 stereo required')
    draft, _, _ = numerical_modules()
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        mono = stereo.mean(axis=1)
        crop, provenance = draft.standardize(mono, 44100)
        require(provenance['resampled_frames'] == 480000 and provenance['crop_start'] == 176000 and
                provenance['crop_stop_exclusive'] == 304000 and crop.shape == (128000,), 'fixed center8 crop support')
        wave = .25 * crop
        if np.any((crop != 0) & (wave == 0)):
            raise FloatingPointError('unsupported background-gain underflow')
        rms = float(np.sqrt(np.mean(wave**2)))
        if not np.isfinite(rms) or (rms == 0 and np.any(wave)):
            raise FloatingPointError('unsupported background RMS arithmetic')
    return wave, {**VIEW, 'input_format': INPUT_FORMAT, 'resampling': provenance,
                  'downmix_float64_sha256': pcm_hash(mono), 'crop_float64_sha256': pcm_hash(crop),
                  'analysis_float64_sha256': pcm_hash(wave), 'baseline_rms': rms,
                  'construction_status': 'ok' if rms > 0 else 'unsupported_zero_background_rms'}


def decode(row):
    import numpy as np
    import soundfile as sf
    physical.validate_audio(row)
    with sf.SoundFile(row['input']['path']) as stream:
        require((stream.format, stream.subtype, stream.samplerate, stream.channels, stream.frames) ==
                ('WAV', 'FLOAT', 44100, 2, 1323000), 'physical format changed before float64 decode')
        values = stream.read(dtype='float64', always_2d=True)
        require(values.shape == (1323000, 2) and not len(stream.read(1)) and np.isfinite(values).all(), 'float64 decode shape/EOF')
    require(hashlib.sha256(np.ascontiguousarray(values, dtype='<f4').tobytes()).hexdigest() == row['waveform_float32_sha256']
            and base.file_binding(row['input']['path']) == row['input'], 'input changed across decode')
    return values


def measure(row):
    _, audio, scalar = numerical_modules()
    wave, view = analysis_view(decode(row))
    result = audio.extract(wave, 16000)
    metadata = result['metadata']; metadata['construction_status'] = view['construction_status']
    reduced = scalar.reduce_metadata(metadata, crop=CROP)
    require(reduced['pool_count'] == 2 and reduced['discarded_tail_samples'] == 0, 'fixed two-pool support')
    require(base.file_binding(row['input']['path']) == row['input'], 'input changed during measurement')
    return view, metadata, result['arrays'], reduced


def validate_arrays(arrays, metadata):
    """Structural and metadata/array consistency audit; not numerical refitting."""
    import numpy as np
    shapes = {'window': ((1024,), 'float64'), 'frequency_bins': ((228, 3), 'int64'),
        'frequency_hz': ((228, 3), 'float64'), 'pool_start_samples': ((2,), 'int64'),
        'frame_offset_samples': ((247,), 'int64'), 'frame_start_samples': ((2, 247), 'int64'),
        'frame_means': ((2, 247), 'float64'), 'spectra': ((2, 247, 513), 'complex128'),
        'bin_coefficient_energy': ((2, 513), 'float64'), 'total_non_dc_coefficient_energy': ((2,), 'float64'),
        'bin_energy_fraction': ((2, 513), 'float64')}
    for name in ('energy_floor_mask', 'primitive_defined_mask', 'eligible_mask'):
        shapes[name] = ((2, 228), 'bool')
    for name in ('squared_bicoherence', 'biphase_radians', 'primitive_squared_bicoherence', 'primitive_biphase_radians'):
        shapes[name] = ((2, 228), 'float64')
    require(set(arrays) == set(shapes), 'full BC NPZ key set')
    nullable = {'bin_energy_fraction', 'squared_bicoherence', 'biphase_radians',
                'primitive_squared_bicoherence', 'primitive_biphase_radians'}
    for name, (shape, dtype) in shapes.items():
        a = arrays[name]
        require(a.shape == shape and a.dtype == np.dtype(dtype) and not np.isinf(a).any(), 'BC array shape/dtype/infinity: ' + name)
        require(name in nullable or np.isfinite(a).all(), 'unexpected BC array NaN: ' + name)
    require(np.array_equal(arrays['pool_start_samples'], [0, 64000]) and
            np.array_equal(arrays['frame_offset_samples'], np.arange(247) * 256) and
            np.array_equal(arrays['frame_start_samples'], np.array([0, 64000])[:, None] + np.arange(247)[None, :] * 256),
            'pool/frame coordinate arrays')
    for p, pool in enumerate(metadata['pools']):
        require(arrays['total_non_dc_coefficient_energy'][p] == pool['total_non_dc_coefficient_energy'], 'pool energy metadata')
        for c, cell in enumerate(pool['cells']):
            require(np.array_equal(arrays['frequency_bins'][c], cell['frequency_bins']), 'grid metadata')
            for name, field in (('eligible_mask', 'eligible'), ('energy_floor_mask', 'energy_floor_passed')):
                require(bool(arrays[name][p, c]) == cell[field], 'array/metadata mask')
            require(bool(arrays['primitive_defined_mask'][p, c]) == (cell['primitive']['status'] == 'ok'), 'primitive mask')
            for name, value in (('squared_bicoherence', cell['squared_bicoherence']),
                                ('biphase_radians', cell['biphase_radians']),
                                ('primitive_squared_bicoherence', cell['primitive']['squared_bicoherence']),
                                ('primitive_biphase_radians', cell['primitive']['biphase_radians'])):
                actual = arrays[name][p, c]
                require(np.isnan(actual) if value is None else actual == value, 'array/metadata scalar')


def paths(output, ident):
    return {name: output / folder / (ident + suffix) for name, folder, suffix in
            (('item', 'items', '.json'), ('metadata', 'metadata', '.json'), ('arrays', 'arrays', '.npz'))}


def write_arrays(path, arrays):
    import numpy as np
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    # Interrupted/failed binary publication is retained, never auto-overwritten.
    with temp.open('xb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.link(temp, path); temp.unlink()


def validate_view(view):
    require(all(view.get(k) == v for k, v in VIEW.items()) and view.get('input_format') == INPUT_FORMAT, 'fixed analysis view')
    p = view['resampling']
    require(p['native_sample_rate_hz'] == 44100 and p['native_frames'] == 1323000 and
            p['resampled_frames'] == 480000 and p['crop_start'] == 176000 and
            p['crop_stop_exclusive'] == 304000 and p['crop_frames'] == 128000 and
            p['crop_pcm_sha256'] == view['crop_float64_sha256'] and
            p['up'] == 160 and p['down'] == 441 and p['window'] == ['kaiser', 5.0] and
            p['padtype'] == 'constant' and p['cval'] == 0.0 and
            p['resample_scope'] == 'entire_native_recording_before_crop', 'resampling view provenance')
    require(all(physical.hash_string(view[k]) for k in
                ('downmix_float64_sha256', 'crop_float64_sha256', 'analysis_float64_sha256')), 'view PCM hashes')
    import math
    rms = view['baseline_rms']
    require(type(rms) is float and math.isfinite(rms) and rms >= 0 and
            view['construction_status'] == ('ok' if rms > 0 else 'unsupported_zero_background_rms'), 'baseline construction status')


def verify_receipt(output, row, run_sha, native_origin, *, audio=True):
    import numpy as np
    _, _, scalar = numerical_modules()
    names = paths(output, row['id'])
    require(all(path.is_file() and not path.is_symlink() for path in names.values()), 'regular nonsymlink item products required')
    envelope = base.read_json(names['item'])
    require(set(envelope) == {'payload', 'receipt_sha256'} and
            envelope['receipt_sha256'] == base.value_hash(envelope['payload']) and
            base.digest(names['item']) == base.value_hash(envelope), 'canonical item receipt')
    r = envelope['payload']
    require(r.get('version') == VERSION and r.get('status') == 'measured_BC_not_classifier_admitted' and
            r.get('id') == row['id'] and r.get('row') == row and r.get('row_sha256') == base.value_hash(row) and
            r.get('contract_sha256') == run_sha and r.get('native_origin') == native_origin and
            all(r.get(k) == v for k, v in SCOPE.items()), 'item lineage/scope')
    for name in ('metadata', 'arrays'):
        require(r[name] == base.file_binding(names[name]), 'measurement product binding')
    metadata = base.read_json(names['metadata'])
    require(base.digest(names['metadata']) == base.value_hash(metadata), 'canonical metadata')
    reduced = scalar.reduce_metadata(metadata, crop=CROP)
    require(reduced['pool_count'] == 2 and reduced['discarded_tail_samples'] == 0 and
            r['scalar'] == reduced and r['features'] == {FEATURE: reduced['median_squared_bicoherence']} and
            metadata['construction_status'] == r['analysis_view']['construction_status'], 'fixed scalar/features replay')
    validate_view(r['analysis_view'])
    with np.load(names['arrays'], allow_pickle=False) as archive:
        validate_arrays({k: archive[k] for k in archive.files}, metadata)
    if audio:
        # Physical/hash verification only; no BC or waveform-view numerical replay.
        physical.validate_audio(row)
    return r


def one(output, row, run_sha, native_origin):
    names = paths(output, row['id'])
    if names['item'].exists():
        return verify_receipt(output, row, run_sha, native_origin)
    require(not any(path.exists() or path.is_symlink() for path in names.values()) and
            not list((output / 'arrays').glob('.' + row['id'] + '.npz.*.tmp')), 'unreceipted products retained; review required')
    view, metadata, arrays, scalar = measure(row)
    validate_view(view); validate_arrays(arrays, metadata)
    write_arrays(names['arrays'], arrays); base.write_new(names['metadata'], metadata)
    require(base.file_binding(row['input']['path']) == row['input'], 'input changed before receipt publication')
    receipt = {'version': VERSION, 'status': 'measured_BC_not_classifier_admitted', 'id': row['id'],
               'row': row, 'row_sha256': base.value_hash(row), 'contract_sha256': run_sha,
               'native_origin': native_origin, 'analysis_view': view,
               'metadata': base.file_binding(names['metadata']), 'arrays': base.file_binding(names['arrays']),
               'scalar': scalar, 'features': {FEATURE: scalar['median_squared_bicoherence']}, **SCOPE}
    base.write_new(names['item'], {'payload': receipt, 'receipt_sha256': base.value_hash(receipt)})
    return receipt


def inventory(output, rows, complete=False):
    ids = {r['id'] for r in rows}; allowed = {'writer.lock', 'contract.json', 'manifest.json', 'COMMIT.json'}
    for ident in ids:
        allowed.update(str(p.relative_to(output)) for p in paths(output, ident).values())
    files = physical.tree_files(output)
    for name in files - allowed:
        p = Path(name)
        valid = (p.parent == Path('failures') and re.fullmatch(r'(.+)\.[0-9a-f]{32}\.json', p.name)
                 and p.name.rsplit('.', 2)[0] in ids)
        require(valid, 'unexpected/orphan product retained: ' + name)
    for path in output.rglob('*'):
        if path.is_dir():
            require(path.parent == output and path.name in {'items', 'metadata', 'arrays', 'failures'}, 'unexpected output directory')
    if complete:
        require(all(str(p.relative_to(output)) in files for i in ids for p in paths(output, i).values()), 'incomplete BC products')


def products(output):
    return {name: {k: v for k, v in base.file_binding(output / name).items() if k != 'path'}
            for name in sorted(physical.tree_files(output, ('writer.lock', 'COMMIT.json')))}


def manifest(output, rows, receipts, run_sha):
    by_source = {}
    for source in sorted({r['source_group'] for r in rows}):
        records = [r for r in receipts if r['row']['source_group'] == source]
        by_source[source] = {'rows': len(records), 'available': sum(r['features'][FEATURE] is not None for r in records),
            'missing': sum(r['features'][FEATURE] is None for r in records),
            'eligible_pool_counts': dict(sorted(Counter(str(r['scalar']['eligible_pool_count']) for r in records).items())),
            'pool_statuses': dict(sorted(Counter(p['target_status'] for r in records for p in r['scalar']['pools']).items()))}
    return {'version': VERSION, 'status': 'all_BC_rows_measured_nulls_retained_not_admitted',
            'contract_sha256': run_sha, 'count': len(rows), 'ids': [r['id'] for r in rows],
            'source_coverage': by_source, 'feature_names': [FEATURE], 'view': VIEW,
            'measurement_receipts': {r['id']: base.file_binding(paths(output, r['id'])['item']) for r in rows}, **SCOPE}


def verify_output(run, *, audio=True):
    prepared = run['prepared']; rows = prepared['rows']; output = physical.safe_path(prepared['output_root'])
    run_sha = base.value_hash(run)
    require(base.read_json(output / 'contract.json') == run and base.digest(output / 'contract.json') == run_sha, 'run contract binding')
    inventory(output, rows, complete=True)
    commit = base.read_json(output / 'COMMIT.json')
    require(commit.get('version') == VERSION and commit.get('status') == 'committed_native30_BC_measurements_not_admitted'
            and commit.get('contract_sha256') == run_sha and commit.get('completed') == len(rows) and
            commit.get('all_bound_inputs_and_products_end_rehashed') is True and
            all(commit.get(k) == v for k, v in SCOPE.items()) and commit['products'] == products(output), 'COMMIT complete inventory')
    receipts = [verify_receipt(output, row, run_sha, prepared['native_origins'][row['id']], audio=audio) for row in rows]
    expected = manifest(output, rows, receipts, run_sha)
    require(base.read_json(output / 'manifest.json') == expected and
            base.digest(output / 'manifest.json') == base.value_hash(expected), 'manifest replay/coverage')
    require(products(output) == commit['products'], 'products changed during verification')
    return {'status': 'verified_complete_native30_BC', 'count': len(rows), 'commit': base.file_binding(output / 'COMMIT.json'),
            'contract': base.file_binding(output / 'contract.json'), 'manifest': base.file_binding(output / 'manifest.json'),
            'feature_names': [FEATURE], 'source_coverage': expected['source_coverage']}


def run_contract(run):
    """Python fixtures may be reduced; production CLI reconstructs fixed3830."""
    prepared = run['prepared']; rows = prepared['rows']; output = physical.safe_path(prepared['output_root'])
    require(len(physical.indexed(rows, 'run')) == prepared['expected_count'] and rows == sorted(rows, key=lambda r: r['id']), 'run roster')
    with source_locks(prepared), base.writer_lock(output):
        check_inputs(run, audio=True)
        path = output / 'contract.json'; run_sha = base.value_hash(run)
        if path.exists():
            require(base.read_json(path) == run and base.digest(path) == run_sha, 'output contract conflict')
        else:
            require(physical.tree_files(output) == {'writer.lock'}, 'nonempty output without contract')
            base.write_new(path, run)
        for folder in ('items', 'metadata', 'arrays', 'failures'):
            (output / folder).mkdir(exist_ok=True)
        inventory(output, rows)
        if (output / 'COMMIT.json').exists():
            result = verify_output(run)
            check_inputs(run, audio=True)
            return result
        failures, receipts = [], []
        for index, row in enumerate(rows, 1):
            try:
                receipts.append(one(output, row, run_sha, prepared['native_origins'][row['id']]))
            except Exception as exc:
                failure = {'id': row['id'], 'contract_sha256': run_sha, 'row_sha256': base.value_hash(row),
                           'exception': type(exc).__name__, 'message': str(exc), 'scientific_null': False, **SCOPE}
                base.write_new(output / 'failures' / (row['id'] + '.' + uuid.uuid4().hex + '.json'), failure)
                failures.append(failure)
            if index % 25 == 0 or index == len(rows):
                print(base.canonical({'reviewed': index, 'expected': len(rows), 'successful': len(receipts), 'failed': len(failures)}).decode().strip(), flush=True)
        check_inputs(run, audio=True)
        if failures:
            return {'status': 'partial_no_COMMIT', 'completed': len(receipts), 'failed': len(failures)}
        receipts = [verify_receipt(output, row, run_sha, prepared['native_origins'][row['id']]) for row in rows]
        value = manifest(output, rows, receipts, run_sha); path = output / 'manifest.json'
        if path.exists():
            require(base.read_json(path) == value and base.digest(path) == base.value_hash(value), 'retained manifest conflict')
        else:
            base.write_new(path, value)
        inventory(output, rows, complete=True)
        bound = products(output)
        check_inputs(run, audio=True)
        require(products(output) == bound, 'products changed during final rehash')
        base.write_new(output / 'COMMIT.json', {'version': VERSION, 'status': 'committed_native30_BC_measurements_not_admitted',
            'contract_sha256': run_sha, 'completed': len(rows), 'products': bound,
            'all_bound_inputs_and_products_end_rehashed': True, **SCOPE})
        return verify_output(run)


def verify_completion(freeze_path, freeze_sha, commit_sha):
    run = validate_freeze(freeze_path, freeze_sha)
    with source_locks(run['prepared']):
        check_inputs(run, audio=True)
        output = Path(run['prepared']['output_root'])
        require(base.digest(output / 'COMMIT.json') == commit_sha, 'caller COMMIT pin mismatch')
        result = verify_output(run)
        check_inputs(run, audio=True)
        require(base.digest(output / 'COMMIT.json') == commit_sha, 'COMMIT changed during verification')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'prepare', 'run', 'verify'), default='preflight')
    parser.add_argument('--request', help='JSON object of six caller-pinned authority bindings')
    parser.add_argument('--request-sha256')
    parser.add_argument('--output')
    parser.add_argument('--prepared-output')
    parser.add_argument('--freeze'); parser.add_argument('--freeze-sha256'); parser.add_argument('--commit-sha256')
    args = parser.parse_args()
    if args.mode in ('run', 'verify'):
        require(args.freeze and args.freeze_sha256, 'explicit parent freeze and SHA required')
        result = (run_contract(validate_freeze(args.freeze, args.freeze_sha256)) if args.mode == 'run' else
                  verify_completion(args.freeze, args.freeze_sha256, args.commit_sha256))
    else:
        require(args.request and args.output and physical.hash_string(args.request_sha256), 'request/output/caller request SHA required')
        entry = base.file_binding(physical.safe_path(args.request))
        require(entry['sha256'] == args.request_sha256, 'request SHA mismatch')
        contract = prepare(pinned_json(entry), args.output)
        if args.mode == 'prepare':
            require(args.prepared_output, 'separate prepared output file required')
            path = physical.safe_path(args.prepared_output)
            require(not path.is_relative_to(Path(args.output)), 'prepared authority outside measurement output')
            base.write_new(path, contract)
        result = {'status': contract['status'], 'rows': contract['expected_count'], 'audio_reads': 0,
                  'prepared_contract_sha256': base.value_hash(contract), 'execution_authorized': False}
    print(base.canonical(result).decode().strip())


if __name__ == '__main__':
    main()
