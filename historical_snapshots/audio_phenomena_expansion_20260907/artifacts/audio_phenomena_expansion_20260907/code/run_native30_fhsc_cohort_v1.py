"""Frozen native30 F/H/SC measurement adapter; no admission or classifier.

Preflight reads metadata only. Prepare publishes a contract without opening any
audio. Run requires the caller-pinned prepared contract and both intake COMMIT
hashes. Missing scientific measurements are retained, never imputed or gated by
SC full-grid diagnostics. Interrupted products and failures are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import uuid

VERSION = 'run_native30_fhsc_cohort_v1'
PILOT_SHA = '0f4a2724081f617c38661c0eb3fad874379561d5525b022bd6601134651dedaa'
HELPER_SHA = '165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a'
PLAN_SHA = '94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
SCREEN_SHA = 'ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87'
SOURCES = {'ACE-Step': 400, 'FMA': 354, 'HeartMuLa': 366, 'MTG-Jamendo': 500,
           'Mureka_v9': 500, 'Suno': 400, 'Udio': 500, 'human_maestro_v3': 300,
           'human_medleydb': 168, 'human_moisesdb': 239, 'human_saraga_hindustani_v1': 103}
EXPECTED = {'plan': 3869, 'new': 1656, 'prior': 2174, 'excluded': 39,
            'total': 3830, 'human': 1664, 'ai': 2166, 'sources': SOURCES}
IDENTITY = ('id', 'source_group', 'label', 'role', 'group_id', 'component_id')
SCREEN_FIELDS = IDENTITY + ('input_occurrences', 'selected_metadata_input')
SCOPE = {'classifier_fits': 0, 'cohort_admitted': False, 'no_neural_inference': True,
         'M_is_diagnostic_only': True, 'source_selection_changed': False}
CLOSURE_STATUS = 'physical_intake_closed_with_explicit_exclusions'
CLOSURE_VERSION = 'finalize_native30_partial_intake_v3'
# All nonaudio files (including CSV/TSV metadata and runtime binaries) are
# rehashed in preflight. Only explicitly recognized media extensions defer IO.
AUDIO_SUFFIXES = {'.wav', '.wave', '.flac', '.mp3', '.ogg', '.opus', '.m4a', '.aac',
                  '.aif', '.aiff', '.wma', '.mp4', '.webm'}
VERSIONS = {'numpy': '1.26.4', 'scipy': '1.17.1', 'soundfile': '0.14.0',
            'librosa': '0.11.0', 'numba': '0.67.0', 'llvmlite': '0.49.0'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def pinned_module(name, expected):
    path = Path(__file__).resolve().with_name(name + '.py')
    require(path.is_file() and not path.is_symlink(), 'regular pinned module required')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'code pin mismatch: ' + name)
    result = importlib.import_module(name)
    require(Path(result.__file__).resolve() == path, 'module import path mismatch: ' + name)
    return result


base = pinned_module('materialize_native30_new1695_v1', HELPER_SHA)


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path,
            'absolute unredirected path required: ' + str(path))
    return path


def indexed(rows, name):
    require(isinstance(rows, list), name + ' rows must be list')
    out = {}
    for row in rows:
        ident = row['id']
        require(isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', ident), 'unsafe ID')
        require(ident not in out, 'duplicate ID: ' + name)
        out[ident] = row
    return out


def declaration(path, entry):
    path = safe_path(path)
    size = entry.get('bytes', entry.get('signature', [None, None, None])[2])
    require(hash_string(entry.get('sha256')) and type(size) is int and size >= 0, 'invalid product binding')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == size,
            'missing/changed product size: ' + str(path))
    return {'path': str(path), 'bytes': size, 'sha256': entry['sha256']}


def add_binding(bindings, entry):
    old = bindings.get(entry['path'])
    require(old is None or old == entry, 'conflicting product binding')
    bindings[entry['path']] = entry


def tree_files(root, exclude=()):
    root = safe_path(root)
    require(root.is_dir(), 'product root missing')
    result = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'product symlink forbidden')
        if path.is_dir():
            continue
        require(path.is_file(), 'nonregular product')
        name = path.relative_to(root).as_posix()
        if name not in exclude:
            result.add(name)
    return result


def bind_commit(root, digest, kind, bindings):
    root = safe_path(root)
    require(hash_string(digest), 'caller COMMIT SHA required')
    entry = base.file_binding(root / 'COMMIT.json')
    require(entry['sha256'] == digest, 'caller COMMIT SHA mismatch')
    add_binding(bindings, entry)
    commit = base.read_json(root / 'COMMIT.json')
    require(commit.get('status') == (CLOSURE_STATUS if kind == 'new' else 'committed_prior2174_DSP_only'),
            'upstream incomplete/wrong COMMIT status')
    products = commit['products']
    require(isinstance(products, dict) and 'manifest.json' in products, 'manifest must be committed')
    require(tree_files(root, ('COMMIT.json', 'writer.lock')) == set(products), 'COMMIT exhaustive product inventory')
    for name, value in products.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and name == relative.as_posix(), 'unsafe product name')
        require(kind == 'prior' or name in {'manifest.json', 'summary.json', 'upstream_bindings.json'}, 'closure product name')
        entry = declaration(root / name, value)
        if relative.suffix != '.wav':
            require(base.file_binding(entry['path']) == entry, 'upstream metadata product changed')
        else:
            require(kind == 'prior' and relative.parent == Path('audio'), 'unexpected audio product')
        add_binding(bindings, entry)
    if kind == 'new':
        require(set(products) == {'manifest.json', 'summary.json', 'upstream_bindings.json'}, 'closure product set')
        require(commit.get('version') == CLOSURE_VERSION
                and commit.get('all_bound_inputs_and_products_end_rehashed') is True, 'v3 closure with exhaustive rehash required')
    return commit, base.read_json(root / 'manifest.json')


def collect_upstream(value, bindings):
    """Closure v3 declares actual resolved path and bytes in every leaf."""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            entry = declaration(value['path'], value)
            if Path(entry['path']).suffix.lower() not in AUDIO_SUFFIXES:
                require(base.file_binding(entry['path']) == entry, 'closure bound metadata changed')
            add_binding(bindings, entry)
        else:
            for child in value.values():
                collect_upstream(child, bindings)
    elif isinstance(value, list):
        for child in value:
            collect_upstream(child, bindings)


def reconcile(plan, screen, new, prior, expected=EXPECTED):
    """Reduced expectations are Python test fixtures only, never CLI options."""
    require(plan.get('schema_version') == 'native30-origin-plan-v2'
            and plan.get('status') == 'draft_not_admitted_not_frozen_for_execution', 'plan schema/status')
    require(screen.get('classifier_fits') == 0 and screen.get('feature_extraction_authorized') is False, 'screen scope')
    require(plan['counts']['audio_files_opened'] == plan['counts']['classifier_fits'] == 0, 'plan scope')
    require(new.get('version') == CLOSURE_VERSION and new.get('status') == CLOSURE_STATUS,
            'new intake closure schema/status')
    require(prior.get('status') == 'all_prior2174_DSP_materialized_not_cohort_admitted', 'prior manifest status')
    for document in (new, prior):
        require(document.get('classifier_fits') == 0 and document.get('cohort_admitted') is False
                and document.get('feature_extraction_authorized') is False, 'upstream scope')
    plans, screened = indexed(plan['rows'], 'plan'), indexed(screen['rows'], 'screen')
    components = {r['component_id']: r for r in screen['components']}
    require(len(components) == len(screen['components']), 'duplicate screen components')
    fresh, old = indexed(new['records'], 'new'), indexed(prior['records'], 'prior')
    excluded = indexed(new['excluded_short_records'], 'exclusions')
    require(len(plans) == expected['plan'] and len(fresh) == expected['new'] and len(old) == expected['prior']
            and len(excluded) == expected['excluded'], 'cohort exact counts')
    require(not (set(fresh) & set(old) or set(fresh) & set(excluded) or set(old) & set(excluded))
            and set(fresh) | set(old) | set(excluded) == set(plans), 'cohort IDs must partition the plan')
    require(set(new['eligible_ids']) == set(fresh) and len(new['eligible_ids']) == len(fresh)
            and set(new['excluded_short_ids']) == set(excluded) and len(new['excluded_short_ids']) == len(excluded)
            and set(new['all_attempted_ids']) == set(fresh) | set(excluded)
            and len(new['all_attempted_ids']) == len(fresh) + len(excluded), 'closure ID accountability')
    require(new['eligible_count'] == len(fresh) and new['excluded_short_count'] == len(excluded)
            and new['attempted'] == len(fresh) + len(excluded) and prior['count'] == len(old), 'manifest declared counts')
    selected = []
    for ident, row in sorted(plans.items()):
        s = screened[ident]
        require(all(row[k] == s[k] for k in SCREEN_FIELDS), 'plan/screen identity/role/group mismatch')
        require(row['role'] == 'development' and row['label'] in {'0', '1'} and row['group_id']
                and s['duration_exposure_candidate'] is True and not s['exclusion_reasons'], 'excluded/protected row')
        component = components[row['component_id']]
        require(ident in component['members'] and ident in component['candidate_screen_members']
                and not component['protected_relationships'], 'protected/component membership mismatch')
        if ident in excluded:
            x = excluded[ident]
            require(row['origin_set'] == 'native30_evidence_v2' and row['source_group'] == 'FMA' and row['label'] == '0'
                    and all(x[k] == row[k] for k in IDENTITY), '39 exclusions must be original FMA human rows')
            require(type(x['actual_frames']) is int and type(x['required_frames']) is int
                    and 0 <= x['actual_frames'] < x['required_frames'] == 30 * x['native_rate_hz'], 'short exclusion evidence')
            continue
        origin = 'new' if ident in fresh else 'prior'
        require(row['origin_set'] == ('native30_evidence_v2' if origin == 'new' else 'prior60'), 'origin family swapped')
        receipt = (fresh if origin == 'new' else old)[ident]
        require(all(receipt[k] == row[k] == receipt['row'][k] for k in IDENTITY), 'receipt identity/role/group mismatch')
        require(receipt['row_sha256'] == base.value_hash(receipt['row'])
                and receipt['status'] == 'materialized_DSP_only', 'receipt row/status')
        native = receipt['row']['native_evidence']
        require(native['sha256'] == row['source_origin']['sha256'] and native['channels'] == 2
                and native['sample_rate_hz'] == row['source_origin']['sample_rate_hz'], 'native provenance mismatch')
        if origin == 'prior':
            require(receipt['row'].get('origin_plan_row_sha256') == base.value_hash(row)
                    and receipt['row'].get('screen_row_sha256') == base.value_hash(s)
                    and receipt['row'].get('component_sha256') == base.value_hash(component), 'prior plan/screen/component binding')
        require(receipt['waveform_bit_exact_roundtrip'] is True
                and hash_string(receipt['waveform_float32_sha256'])
                and receipt['waveform_float32_sha256'] == receipt['audit']['output_waveform_float32_sha256'], 'output PCM receipt')
        selected.append({**{k: row[k] for k in IDENTITY}, 'origin_family': origin,
                         'origin_plan_row_sha256': base.value_hash(row), 'screen_row_sha256': base.value_hash(s),
                         'component_sha256': base.value_hash(component), 'producer_receipt_sha256': base.value_hash(receipt),
                         'input': {'path': receipt['standardized_path'], 'bytes': receipt['file_bytes'], 'sha256': receipt['file_sha256']},
                         'waveform_float32_sha256': receipt['waveform_float32_sha256']})
    require(len(selected) == expected['total'] and dict(Counter(r['source_group'] for r in selected)) == expected['sources']
            and Counter(r['label'] for r in selected) == {'0': expected['human'], '1': expected['ai']}, 'combined source/label roster')
    return selected


def bind_receipts(root, manifest, bindings, origin):
    contract_path = root / 'contract.json'
    contract_binding = base.file_binding(contract_path)
    add_binding(bindings, contract_binding)
    contract = base.read_json(contract_path)
    require(manifest['contract_sha256'] == contract_binding['sha256'], 'producer contract binding')
    contract_rows = indexed(contract['rows'], 'producer contract')
    for receipt in manifest['records']:
        ident = receipt['id']
        item = root / 'items' / (ident + '.json')
        require(str(item) in bindings, 'unbound producer receipt')
        envelope = base.read_json(item)
        require(set(envelope) == {'payload', 'receipt_sha256'} and envelope['payload'] == receipt
                and envelope['receipt_sha256'] == base.value_hash(receipt), 'manifest/receipt payload mismatch')
        require(receipt['contract_sha256'] == contract_binding['sha256'] and receipt['row'] == contract_rows[ident],
                'producer contract row mismatch')
        wav = root / 'audio' / (ident + '.wav')
        require(receipt['standardized_path'] == str(wav), 'producer output path mismatch')
        expected = declaration(wav, {'bytes': receipt['file_bytes'], 'sha256': receipt['file_sha256']})
        require(bindings.get(str(wav)) == expected, 'WAV missing from exhaustive product bindings')
    if origin == 'new':
        require(not (root / 'COMMIT.json').exists() and not (root / 'manifest.json').exists(), 'original partial run was mutated')
    return contract


def measurement_runtime(pilot):
    dsp = pinned_module('standardize_native30_v1', pilot.PINS['standardize_native30_v1.py'])
    runtime = base.runtime_binding(dsp)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, 'frozen measurement numerical runtime versions')
    # Bind executed Python sources and extension binaries, including lazy librosa
    # modules and numerical/JIT libraries absent from the older DSP-only helper.
    package_files, import_paths = {}, {}
    for name in ('librosa', 'numba', 'llvmlite', 'numpy', 'scipy', 'sklearn', 'soxr',
                 'joblib', 'threadpoolctl', 'decorator', 'lazy_loader', 'msgpack',
                 'pooch', 'audioread', 'soundfile', 'cffi', 'packaging', 'platformdirs',
                 'requests', 'urllib3', 'certifi', 'charset_normalizer', 'idna'):
        package = importlib.import_module(name)
        module_path = Path(package.__file__).resolve()
        import_paths[name] = str(module_path)
        paths = [module_path]
        if hasattr(package, '__path__'):
            paths += list(module_path.parent.rglob('*'))
            # Wheel-supplied BLAS/OpenMP/shared libraries live beside packages.
            for candidate in (module_path.parent.with_name(name + '.libs'),
                              module_path.parent.with_name(name + '.dylibs')):
                if candidate.is_dir():
                    paths += list(candidate.rglob('*'))
        for path in sorted(set(paths)):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
                package_files[str(path)] = base.file_binding(path)
    runtime.update(measurement_versions=versions, measurement_package_files=package_files, measurement_import_paths=import_paths,
                   thread_environment={k: os.environ.get(k) for k in
                                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})
    return runtime


def prepare(new_root, new_sha, prior_root, prior_sha, plan_path, screen_path, output):
    new_root, prior_root, plan_path, screen_path, output = map(safe_path,
        (new_root, prior_root, plan_path, screen_path, output))
    bindings = {}
    for path, expected in ((plan_path, PLAN_SHA), (screen_path, SCREEN_SHA)):
        entry = base.file_binding(path)
        require(entry['sha256'] == expected, 'fixed plan/screen SHA mismatch')
        add_binding(bindings, entry)
    nc, new = bind_commit(new_root, new_sha, 'new', bindings)
    pc, prior = bind_commit(prior_root, prior_sha, 'prior', bindings)
    require(nc['eligible'] == EXPECTED['new'] and nc['excluded_short'] == EXPECTED['excluded']
            and pc['completed'] == EXPECTED['prior'], 'committed intake counts')
    upstream = base.read_json(new_root / 'upstream_bindings.json')
    collect_upstream(upstream, bindings)
    original = safe_path(new['original_root'])
    require(nc['original_root'] == str(original) and original != new_root, 'separate closure original root')
    bind_receipts(original, new, bindings, 'new')
    bind_receipts(prior_root, prior, bindings, 'prior')
    rows = reconcile(base.read_json(plan_path), base.read_json(screen_path), new, prior)
    # The producer and closure cover every successful and failed product. Extra
    # original files cannot become untracked inputs through a later resume.
    original_inventory = tree_files(original, ('writer.lock',))
    require(all(str(original / name) in bindings for name in original_inventory), 'unbound original partial product')
    require({str(original / 'items' / (r['id'] + '.json')) for r in rows if r['origin_family'] == 'new'}
            == set(upstream['original_success_products']) - {r['input']['path'] for r in rows if r['origin_family'] == 'new'},
            'closure exhaustive success receipt/WAV set')
    pilot = pinned_module('run_native30_fhsc_pilot_v1', PILOT_SHA)
    for name, digest in pilot.PINS.items():
        module = pinned_module(Path(name).stem, digest)
        add_binding(bindings, base.file_binding(module.__file__))
    add_binding(bindings, base.file_binding(pilot.__file__))
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name('test_' + VERSION + '.py')):
        add_binding(bindings, base.file_binding(path))
    runtime = measurement_runtime(pilot)
    import audio_inputs, phase_features, musical_features, stereo_candidate_v1
    require(output.parent.is_dir() and all(not Path(p).is_relative_to(output) for p in bindings)
            and all(not output.is_relative_to(root) for root in (new_root, prior_root, original)), 'unsafe output overlap')
    contract = {'version': VERSION, 'status': 'frozen_before_any_measurement_audio_reads',
                'output_root': str(output), 'expected_count': EXPECTED['total'], 'rows': rows, 'bindings': bindings,
                'runtime': runtime, 'source_counts': SOURCES, 'human': EXPECTED['human'], 'ai': EXPECTED['ai'],
                'upstream_commits': {'new': {'path': str(new_root / 'COMMIT.json'), 'sha256': new_sha},
                                     'prior': {'path': str(prior_root / 'COMMIT.json'), 'sha256': prior_sha}},
                'upstream_inventories': {str(new_root): sorted(tree_files(new_root, ('writer.lock',))),
                                        str(prior_root): sorted(tree_files(prior_root, ('writer.lock',))),
                                        str(original): sorted(original_inventory)},
                'F_H_view': audio_inputs.CONFIG, 'F_config': phase_features.CONFIG,
                'F_measure_names': list(phase_features.FEATURE_NAMES), 'H_measure_names': list(musical_features.H_MEASURE_NAMES),
                'SC_config': pilot.clean(stereo_candidate_v1.CONFIG), 'SC_selected_six': pilot.SC_KEYS,
                'SC_full_grid': 'diagnostic_only_separate_from_selected_six', 'M_min_context_seconds': musical_features.M_MIN_DURATION_SEC,
                'input_format': {'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100, 'channels': 2, 'frames': 1323000},
                'preflight_audio_open_count': 0, 'input_audio_SHA_verification': 'deferred_to_explicit_run_then_end_rehashed',
                'upstream_source_rehash_scope': 'all_closure_declared_sources_and_all_prior_COMMIT_products; no_native_DSP_replay',
                'resume_policy': 'verify_receipt_input_SHA_PCM_format_and_frame_product; retain_orphans_fail_closed', **SCOPE}
    check_contract(contract, audio=False)
    return contract, pilot


def check_contract(contract, *, audio):
    for entry in contract['bindings'].values():
        current = declaration(entry['path'], entry)
        if audio or Path(entry['path']).suffix.lower() not in AUDIO_SUFFIXES:
            require(base.file_binding(entry['path']) == current, 'bound input changed: ' + entry['path'])
    for root, inventory in contract.get('upstream_inventories', {}).items():
        require(sorted(tree_files(root, ('writer.lock',))) == inventory, 'upstream inventory changed')
    runtime = contract.get('runtime')
    if runtime:
        base.recheck({'python': runtime['executable'], 'libsndfile': runtime['libsndfile_binary'],
                      **runtime['module_files'], **runtime['measurement_package_files']})
        require({name: importlib.metadata.version(name) for name in VERSIONS} == runtime['measurement_versions'], 'runtime versions changed')
        require({name: str(Path(importlib.import_module(name).__file__).resolve()) for name in runtime['measurement_import_paths']}
                == runtime['measurement_import_paths'], 'runtime module import paths changed')
        require({k: os.environ.get(k) for k in runtime['thread_environment']} == runtime['thread_environment'], 'runtime thread configuration changed')


def validate_audio(row):
    import numpy as np
    import soundfile as sf
    path = safe_path(row['input']['path'])
    require(base.file_binding(path) == row['input'], 'measurement input bytes changed')
    with sf.SoundFile(path) as stream:
        require((stream.format, stream.subtype, stream.samplerate, stream.channels, stream.frames) ==
                ('WAV', 'FLOAT', 44100, 2, 1323000), 'exact FLOAT stereo native30 WAV required')
        pcm = stream.read(dtype='float32', always_2d=True)
        require(pcm.shape == (1323000, 2) and not len(stream.read(1, dtype='float32', always_2d=True))
                and np.isfinite(pcm).all(), 'input PCM shape/finite/EOF')
    require(hashlib.sha256(np.ascontiguousarray(pcm, dtype='<f4').tobytes()).hexdigest() == row['waveform_float32_sha256'],
            'input PCM hash mismatch')


def validate_measurement(result, row, keys):
    import phase_features
    import musical_features
    for group, names in (('F', phase_features.FEATURE_NAMES), ('H', musical_features.H_MEASURE_NAMES)):
        require(set(names) <= set(result[group]) and all(k.startswith(group + '_') for k in result[group]),
                'malformed ' + group + ' measurement keys')
        require(all(result[group][k] is None or type(result[group][k]) in (int, float) for k in names),
                'malformed scientific measurement value')
    require(result['M_diagnostic_not_predictor']['M_status'] in {'missing_short_duration', 'missing_low_energy'}, 'M cannot be predictor')
    sc = result['SC']
    require(set(sc['features']) == set(keys), 'SC selected six changed')
    missing = [k for k in keys if sc['features'][k] is None]
    require(sc['selected_six_finite_count'] == 6 - len(missing) and sc['selected_six_missing'] == missing
            and sc['selected_six_status'] == ('complete' if not missing else 'missing' if len(missing) == 6 else 'partial'), 'SC missingness mismatch')
    audit = result['analysis_view_audit']
    require(audit['source_audio_path'] == row['input']['path'] and audit['source_audio_sha256'] == row['input']['sha256']
            and audit['source_sample_rate'] == 44100 and audit['source_channels'] == 2 and audit['source_total_frames'] == 1323000
            and audit['crop_start_frame'] == 0 and audit['crop_frames'] == 1323000
            and audit['analysis_sr'] == 16000 and audit['analysis_frames'] == 480000, 'F/H analysis view changed')
    require(isinstance(result['F']['F_status'], str) and isinstance(result['H']['H_status'], str)
            and isinstance(sc['full_grid_diagnostic_status'], str) and sc['frame_count'] == 1288, 'measurement statuses/frame count')
    base.canonical(result)  # Reject nonfinite JSON; legitimate missingness is null.


FRAME_KEYS = {'iid_db', 'signed_real_coherence', 'magnitude_coherence', 'side_energy_fraction', 'ipd_increment_rad'}
MASK_KEYS = {'energy_valid', 'two_sided_valid', 'ipd_phase_valid', 'ipd_increment_valid'}


def validate_frames(arrays):
    import numpy as np
    require(set(arrays) == FRAME_KEYS | MASK_KEYS, 'SC frame array names')
    for name, array in arrays.items():
        require(array.shape == (5, 1288) and array.dtype == (np.dtype('bool') if name in MASK_KEYS else np.dtype('float64')),
                'SC frame shape/dtype')
        if name in FRAME_KEYS:
            require(not np.isinf(array).any(), 'infinite SC frame measurement')


def publish_frames(path, arrays):
    import numpy as np
    validate_frames(arrays)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    # Preserve temporary/orphan files on failure; never infer they are usable.
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as archive:
        decoded = {k: archive[k] for k in archive.files}
        validate_frames(decoded)
        require(all(np.array_equal(decoded[k], arrays[k], equal_nan=True) for k in arrays), 'SC NPZ roundtrip')
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_receipt(output, row, contract_sha, keys):
    import numpy as np
    item = output / 'items' / (row['id'] + '.json')
    require(item.is_file() and not item.is_symlink(), 'regular measurement receipt required')
    envelope = base.read_json(item)
    require(set(envelope) == {'payload', 'receipt_sha256'} and base.value_hash(envelope['payload']) == envelope['receipt_sha256'], 'measurement receipt hash')
    receipt = envelope['payload']
    require(receipt['contract_sha256'] == contract_sha and receipt['row'] == row
            and receipt['row_sha256'] == base.value_hash(row) and receipt['status'] == 'measured_not_admitted'
            and all(receipt.get(k) == v for k, v in SCOPE.items()), 'measurement receipt frozen lineage/scope')
    validate_audio(row)
    validate_measurement(receipt['measurement'], row, keys)
    path = output / 'frames' / (row['id'] + '.sc_frames.npz')
    require(receipt['frame_product'] == base.file_binding(path), 'SC frame product changed')
    with np.load(path, allow_pickle=False) as archive:
        validate_frames({k: archive[k] for k in archive.files})
    return receipt


def one(output, row, contract_sha, pilot):
    ident = row['id']
    item, frames = output / 'items' / (ident + '.json'), output / 'frames' / (ident + '.sc_frames.npz')
    if item.exists() or item.is_symlink():
        return verify_receipt(output, row, contract_sha, pilot.SC_KEYS)
    require(not frames.exists() and not frames.is_symlink()
            and not list((output / 'frames').glob('.' + ident + '.sc_frames.npz.*.tmp')), 'unreceipted SC product retained; review required')
    validate_audio(row)
    measurement, sc = pilot.measure(Path(row['input']['path']))
    validate_measurement(measurement, row, pilot.SC_KEYS)
    require(base.file_binding(row['input']['path']) == row['input'], 'input changed during measurement')
    require(not (set(sc['per_frame']) & set(sc['valid_masks'])), 'SC arrays overlap')
    publish_frames(frames, {**sc['per_frame'], **sc['valid_masks']})
    receipt = {'status': 'measured_not_admitted', 'row': row, 'row_sha256': base.value_hash(row),
               'contract_sha256': contract_sha, 'measurement': measurement,
               'frame_product': base.file_binding(frames), **SCOPE}
    base.write_new(item, {'payload': receipt, 'receipt_sha256': base.value_hash(receipt)})
    return receipt


def inventory(output, rows, complete=False):
    ids = {r['id'] for r in rows}
    permitted = {'writer.lock', 'contract.json', 'manifest.json', 'COMMIT.json', 'items', 'frames', 'failures', 'runs'}
    require({p.name for p in output.iterdir()} <= permitted, 'unexpected measurement root entry')
    for path in output.iterdir():
        require(not path.is_symlink() and (path.is_dir() if path.name in {'items', 'frames', 'failures', 'runs'} else path.is_file()),
                'unsafe measurement root entry')
    for directory in ('items', 'frames', 'failures', 'runs'):
        for path in (output / directory).iterdir():
            require(path.is_file() and not path.is_symlink(), 'unsafe measurement product')
            if directory == 'items':
                valid = path.name in {i + '.json' for i in ids}
            elif directory == 'frames':
                match = re.fullmatch(r'\.(.+)\.sc_frames\.npz\.[0-9a-f]{32}\.tmp', path.name)
                valid = path.name in {i + '.sc_frames.npz' for i in ids} or (not complete and match and match[1] in ids)
            elif directory == 'failures':
                match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.json', path.name)
                valid = match and match[1] in ids
            else:
                valid = re.fullmatch(r'[0-9a-f]{32}\.json', path.name)
            require(valid, 'unexpected measurement ' + directory + ' product')
        if complete and directory in {'items', 'frames'}:
            suffix = '.json' if directory == 'items' else '.sc_frames.npz'
            require({p.name for p in (output / directory).iterdir()} == {i + suffix for i in ids}, 'incomplete measurement products')


def products(output):
    return {name: {k: v for k, v in base.file_binding(output / name).items() if k != 'path'}
            for name in sorted(tree_files(output, ('writer.lock', 'COMMIT.json')))}


def freeze(contract):
    output = safe_path(contract['output_root'])
    with base.writer_lock(output):
        check_contract(contract, audio=False)
        path = output / 'contract.json'
        if path.exists():
            require(not path.is_symlink() and base.read_json(path) == contract, 'prepared contract conflict')
        else:
            require({p.name for p in output.iterdir()} == {'writer.lock'}, 'nonempty output without prepared contract')
            base.write_new(path, contract)
    return base.digest(path)


def run_contract(contract, pilot, contract_sha):
    """Python API allows reduced synthetic contracts; CLI always prepares3830."""
    require(hash_string(contract_sha) and base.value_hash(contract) == contract_sha, 'caller prepared contract SHA mismatch')
    output, rows = safe_path(contract['output_root']), contract['rows']
    require(len(indexed(rows, 'measurement contract')) == contract['expected_count']
            and rows == sorted(rows, key=lambda r: r['id']), 'contract exact deterministic rows')
    with base.writer_lock(output):
        require(base.digest(output / 'contract.json') == contract_sha and base.read_json(output / 'contract.json') == contract,
                'explicit prepare required before run')
        check_contract(contract, audio=True)
        for name in ('items', 'frames', 'failures', 'runs'):
            (output / name).mkdir(exist_ok=True)
        inventory(output, rows)
        commit_path = output / 'COMMIT.json'
        committed = commit_path.exists()
        if committed:
            commit = base.read_json(commit_path)
            require(commit['contract_sha256'] == contract_sha and commit['completed'] == len(rows)
                    and commit['status'] == 'committed_native30_F_H_SC_measurements_not_admitted'
                    and commit['products'] == products(output), 'existing measurement COMMIT changed')
        run_id, receipts, failures = uuid.uuid4().hex, [], []
        for index, row in enumerate(rows, 1):
            try:
                receipt = (verify_receipt(output, row, contract_sha, pilot.SC_KEYS) if committed else
                           one(output, row, contract_sha, pilot))
                receipts.append(receipt)
            except Exception as exc:
                failure = {'id': row['id'], 'row_sha256': base.value_hash(row), 'contract_sha256': contract_sha,
                           'run_id': run_id, 'exception': type(exc).__name__, 'message': str(exc)}
                failures.append(failure)
                if not committed:
                    base.write_new(output / 'failures' / (row['id'] + '.' + run_id + '.json'), failure)
            if index % 25 == 0 or index == len(rows):
                print(base.canonical({'event': 'measurement_progress', 'reviewed': index, 'expected': len(rows),
                                      'measured_or_verified': len(receipts), 'failed': len(failures)}).decode().strip(), flush=True)
        check_contract(contract, audio=True)
        inventory(output, rows, complete=not failures)
        if committed:
            require(not failures and base.read_json(commit_path)['products'] == products(output), 'committed measurement verification failed')
            return {'status': 'verified_existing_COMMIT', 'completed': len(rows), 'commit_sha256': base.digest(commit_path)}
        summary = {'status': 'partial_no_COMMIT' if failures else 'all_items_measured_scientific_missingness_retained',
                   'expected': len(rows), 'completed': len(receipts), 'failed': len(failures), 'failures': failures,
                   'contract_sha256': contract_sha, 'run_id': run_id, **SCOPE}
        base.write_new(output / 'runs' / (run_id + '.json'), summary)
        if failures:
            return summary
        # Reopen every receipt and NPZ; do not commit merely because the workers
        # returned successfully. Missing metric values remain legitimate results.
        receipts = [verify_receipt(output, row, contract_sha, pilot.SC_KEYS) for row in rows]
        measurements = [r['measurement'] for r in receipts]
        availability = {name: dict(Counter(r[group][field] for r in measurements)) for name, group, field in
                        [('F', 'F', 'F_status'), ('H', 'H', 'H_status'), ('SC_selected_six', 'SC', 'selected_six_status'),
                         ('SC_full_grid_diagnostic', 'SC', 'full_grid_diagnostic_status'),
                         ('M_diagnostic_only', 'M_diagnostic_not_predictor', 'M_status')]}
        manifest = {'version': VERSION, 'status': 'all_items_measured_scientific_missingness_retained',
                    'contract_sha256': contract_sha, 'count': len(rows), 'ids': [r['id'] for r in rows],
                    'availability': availability, 'source_counts': dict(Counter(r['source_group'] for r in rows)),
                    'measurement_receipts': {r['id']: base.file_binding(output / 'items' / (r['id'] + '.json')) for r in rows}, **SCOPE}
        path = output / 'manifest.json'
        if path.exists():
            require(base.read_json(path) == manifest, 'retained manifest conflict')
        else:
            base.write_new(path, manifest)
        check_contract(contract, audio=True)
        inventory(output, rows, complete=True)
        bound_products = products(output)
        require(products(output) == bound_products, 'measurement products changed during final rehash')
        base.write_new(commit_path, {'status': 'committed_native30_F_H_SC_measurements_not_admitted',
                                    'contract_sha256': contract_sha, 'completed': len(rows), 'products': bound_products,
                                    'all_bound_inputs_and_products_end_rehashed': True, **SCOPE})
        return {**summary, 'availability': availability, 'commit_sha256': base.digest(commit_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('new-closure', 'new-commit-sha256', 'prior', 'prior-commit-sha256', 'plan', 'screen', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('preflight', 'prepare', 'run'), default='preflight')
    parser.add_argument('--contract-sha256')
    args = parser.parse_args()
    if args.mode == 'run':
        require(hash_string(args.contract_sha256), 'run requires caller-pinned --contract-sha256 from prepare')
    contract, pilot = prepare(args.new_closure, args.new_commit_sha256, args.prior, args.prior_commit_sha256,
                              args.plan, args.screen, args.output)
    if args.mode == 'run':
        result = run_contract(contract, pilot, args.contract_sha256)
    else:
        digest = freeze(contract) if args.mode == 'prepare' else base.value_hash(contract)
        result = {'status': 'prepared_no_audio_opened' if args.mode == 'prepare' else 'metadata_preflight_only_no_audio_reads_or_writes',
                  'contract_sha256': digest, 'expected': EXPECTED['total'], 'human': EXPECTED['human'], 'ai': EXPECTED['ai'],
                  'source_counts': SOURCES, 'audio_open_count': 0, **SCOPE}
    print(base.canonical(result).decode().strip(), flush=True)
    return 1 if result['status'] == 'partial_no_COMMIT' else 0


if __name__ == '__main__':
    raise SystemExit(main())
