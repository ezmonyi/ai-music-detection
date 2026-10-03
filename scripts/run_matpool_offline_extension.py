"""Append an immutable 26-group shard to the verified 24-group MatPool cache.

No offline optimizer, proxy, changed guard, or overwritten first-shard files.
The dated original controller is loaded only after its source hash is checked.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/ai-music-rl-20261003')
PROVISION = ROOT / 'provisioning'
FIRST = ROOT / 'data/acestep_grpo_offline_g10_24x4_20261003'
FIRST_PLAN = ROOT / 'data/offline_persistent_plan_20261003'
OUTPUT = ROOT / 'data/acestep_grpo_offline_g10_extension_26x4_20261003'
PLAN = ROOT / 'data/offline_50_group_plan_20261003'
STATE = PROVISION / 'offline_extension_26_pipeline_state.json'
BASE_SCRIPT = PROVISION / 'run_offline_persistent_pipeline.py'
BASE_SHA = '5ee1f85fbb78c7212014277581d42e1b838b39e9569593d4eca2c10155eb496a'
BEHAVIOR_SHA = '909c003d33c9a150c6ac947969ccf139f4f4ae52d9a1d8ab5f8e36b1958047e4'
COUNT, OFFSET, GROUP_SIZE = 26, 126, 4
OWNED_STATE = False


def read_json(path):
    return json.loads(Path(path).read_text())


def load_base(path=BASE_SCRIPT):
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Original controller source changed')
    spec = importlib.util.spec_from_file_location('verified_initial_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_base(module):
    module.OUTPUT, module.PLAN, module.STATE = OUTPUT, PLAN, STATE
    module.COUNT, module.OFFSET, module.GROUP_SIZE = COUNT, OFFSET, GROUP_SIZE
    return module


def combine_verified_shards(first, extension, first_manifest, extension_manifest):
    for receipt, expected in ((first, 24), (extension, COUNT)):
        if (receipt.get('status') != 'complete_persistent_dataset_all_members_sha256_verified'
                or receipt.get('prompt_groups') != expected
                or receipt.get('candidates') != expected * GROUP_SIZE
                or receipt.get('paired_wavs') != expected * GROUP_SIZE * 2
                or receipt.get('optimizer_updates') != 0
                or receipt.get('proxy_used') is not False):
            raise ValueError('A shard is incomplete or violates the frozen-policy contract')
    ids = []
    for manifest, count, offset in ((first_manifest, 24, 102), (extension_manifest, COUNT, OFFSET)):
        if (manifest['prompt_groups'] != count or manifest['group_size'] != GROUP_SIZE
                or manifest['selection_offset'] != offset
                or manifest['checkpoint_sha256'] != BEHAVIOR_SHA
                or manifest['trajectory_dtype'] != 'float32'
                or manifest['solver_steps'] != 50
                or manifest['guards_changed'] is not False
                or manifest['optimizer_updates'] != 0
                or manifest['reward_families'] != ['S', 'R', 'F']):
            raise ValueError('Shard manifest differs from the approved protocol')
        if len(manifest['prompt_ids']) != count:
            raise ValueError('Wrong prompt membership count')
        ids.extend(manifest['prompt_ids'])
    if len(set(ids)) != 50:
        raise ValueError('Offline shards overlap')
    for key in ('config_sha256', 'validation_data_sha256', 'test_data_sha256'):
        if first_manifest[key] != extension_manifest[key]:
            raise ValueError('Shard configuration or held-out boundary changed')
    return {'status': 'complete_50_groups_all_members_sha256_verified',
            'prompt_groups': 50, 'candidates': 200, 'paired_wavs': 400,
            'optimizer_updates': 0, 'behavior_checkpoint_sha256': BEHAVIOR_SHA,
            'first_shard': first, 'extension_shard': extension,
            'bytes_verified': first['bytes_verified'] + extension['bytes_verified'],
            'files_verified': first['files_verified'] + extension['files_verified'],
            'original_probe_groups_retained_separately': 2,
            'drive_upload_required': False, 'proxy_used': False,
            'snapshot_complete': False, 'full_local_backup_complete': False}


def validate_storage_basis(files, config, selected):
    """Use already serialized fixed-shape records, not a pre-probe percentage."""
    wavs = [f['bytes'] for f in files if f['path'].endswith('.wav')]
    traces = [f['bytes'] for f in files if f['path'].endswith('_trajectory.pt')]
    conditions = [f['bytes'] for f in files if f['path'].endswith('/condition.pt')]
    metadata = [f['bytes'] for f in files if f['path'].endswith('/group.json')]
    if (len(wavs) != 192 or set(wavs) != {11520088}
            or len(traces) != 96 or set(traces) != {9796301}
            or len(conditions) != 24 or max(conditions) > 3_000_000
            or len(metadata) != 24 or max(metadata) > 400_000
            or config['sampling']['duration_s'] != 30
            or config['sampling']['steps'] != 50
            or config['model']['precision'] != 'bfloat16'
            or len(selected) != COUNT
            or any(r['duration_s'] != 30 or r['lyrics'] != '' for r in selected)):
        raise ValueError('Measured fixed-shape storage contract changed')


def capacity_projection(module, groups, free_bytes, *, online_complete=False, completed_test_wavs=0):
    plan = module.capacity_projection(groups, free_bytes)
    if not 0 <= completed_test_wavs <= 100 or (completed_test_wavs and groups != 100):
        raise ValueError('Invalid completed-test storage credit')
    # Exact verified WAV/FP32 dimensions, generous 3MB condition and 400KB group
    # JSON bounds, complete adapter copy, plus 64MiB for global metadata. The
    # separate 4GiB overall safety reserve and 26GB snapshot estimate stay fixed.
    raw = COUNT * (GROUP_SIZE * (2*module.WAV_BYTES + module.TRAJECTORY_BYTES)
        + 3_000_000 + 400_000) + 169603855
    plan['pre_probe_20pct_dataset_budget_bytes'] = plan['new_dataset_budget_bytes']
    plan['new_dataset_budget_bytes'] = raw + 64*2**20
    plan['data_budget_basis'] = 'verified_fixed_shape_serialization_plus_bounded_conditions_and_metadata'
    plan['completed_test_wavs_storage_credit'] = completed_test_wavs
    plan['remaining_online_worst_case_bytes'] -= completed_test_wavs * module.WAV_BYTES
    if online_complete:
        if groups != 100:
            raise ValueError('Cannot omit future online storage before 100 groups')
        # Only after actual final-test summary AND all 100 WAVs exist; all their
        # bytes are already reflected in disk_usage, so do not count them twice.
        plan['remaining_online_worst_case_bytes'] = 0
    plan['needed_free_bytes'] = (plan['new_dataset_budget_bytes']
        + plan['remaining_online_worst_case_bytes']
        + plan['environment_snapshot_budget_bytes'] + plan['safety_reserve_bytes'])
    plan['capacity_gate_passed'] = free_bytes >= plan['needed_free_bytes']
    plan['online_outputs_confirmed_complete'] = online_complete
    return plan


def online_outputs_complete(phase):
    if phase != 'training_and_test_complete_pending_backup_and_snapshot':
        return False
    main = read_json(ROOT / 'runs/a6000-srf-pilot-main-v2/summary.json')
    test_dir = ROOT / 'runs/a6000-srf-final-test-v2'
    test = read_json(test_dir / 'summary.json')
    if (main['groups_completed'] != 100 or main['optimizer_updates'] <= 0
            or test['count'] != 50 or len(list(test_dir.rglob('*.wav'))) != 100):
        raise ValueError('Online-complete phase lacks actual training/test files')
    return True


def main():
    global OWNED_STATE
    if STATE.exists() or OUTPUT.exists() or PLAN.exists():
        raise FileExistsError('Never duplicate or overwrite the extension')
    base = configure_base(load_base())
    first = read_json(FIRST_PLAN / 'PERSISTENT_VERIFICATION.json')
    first_manifest = read_json(FIRST / 'COLLECTION_MANIFEST.json')
    if (first['status'] != 'complete_persistent_dataset_all_members_sha256_verified'
            or first['prompt_groups'] != 24
            or hashlib.sha256((FIRST / 'FILES_SHA256.json').read_bytes()).hexdigest()
                != first['manifest_sha256']):
        raise ValueError('The completed first shard changed or is not verified')
    from music_detector.rl.offline_io import select_prompts
    data = ROOT / 'data/musiccaps_a6000_curated_500_20261003'
    _, selected = select_prompts(data/'train.jsonl', data/'validation.jsonl', data/'test.jsonl',
        offset=OFFSET, count=COUNT, online_groups=100)
    validate_storage_basis(read_json(FIRST/'FILES_SHA256.json')['files'],
        read_json(PROVISION/'acestep_a6000_srf_v2_resolved.json'), selected)
    OWNED_STATE = True
    started = time.monotonic()
    last_observation = None
    while True:
        online = read_json(PROVISION / 'pipeline_state_v2.json')
        if online['phase'] == 'failed':
            raise RuntimeError('Online pipeline failed; preserve and inspect')
        latest = json.loads((ROOT / 'runs/a6000-srf-pilot-main-v2/metrics.jsonl').read_text().splitlines()[-1])
        complete = online_outputs_complete(online['phase'])
        closed_wavs = 0
        if latest['groups_completed'] == 100:
            test_dir = ROOT / 'runs/a6000-srf-final-test-v2'
            closed_wavs = sum(p.stat().st_size == base.WAV_BYTES for p in test_dir.glob('*.wav'))
        capacity = capacity_projection(base, latest['groups_completed'],
            shutil.disk_usage(ROOT).free, online_complete=complete, completed_test_wavs=closed_wavs)
        if capacity['capacity_gate_passed']:
            break
        observation = (latest['groups_completed'], online['phase'], capacity['free_bytes'] // (16*2**20))
        if observation != last_observation:
            base.state('waiting_for_persistent_headroom', capacity=capacity,
                       target_total_groups=50, new_groups=COUNT, no_gpu_worker_started=True)
            last_observation = observation
        if complete:
            raise RuntimeError('Final online files complete but persistent snapshot/safety headroom is insufficient')
        if time.monotonic() - started > 7200:
            raise TimeoutError('Persistent capacity wait exceeded two hours; no collection launched')
        time.sleep(30)
    PLAN.mkdir(exist_ok=False)
    shutil.copy2(__file__, PLAN / Path(__file__).name)
    base.write_json(PLAN / 'CAPACITY_AT_LAUNCH.json', capacity)
    base.write_json(PLAN / 'USER_SCOPE.json', {'cumulative_prompt_groups': 50,
        'existing_immutable_groups': 24, 'new_groups': COUNT, 'new_candidates': 104,
        'new_paired_wavs': 208, 'selection': 'sorted train[126:152]',
        'original_probe_retained_separately': True, 'persistent_expansion_authorized': False,
        'temporary_vpn_permission': False, 'offline_optimizer_updates': 0,
        'online_sampler_reward_guards_unchanged': True})
    data = ROOT / 'data/musiccaps_a6000_curated_500_20261003'
    env = dict(os.environ, PYTHONPATH='/root/music-rl-20261003/offline-runtime/src')
    for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
        env.pop(key, None)
    log = PROVISION / 'offline_extension_26_collect.log'
    args = [sys.executable, '-u', '-m', 'music_detector.rl.offline_collect',
        '--config', str(PROVISION / 'acestep_a6000_srf_v2_resolved.json'),
        '--train-data', str(data / 'train.jsonl'),
        '--validation-data', str(data / 'validation.jsonl'), '--test-data', str(data / 'test.jsonl'),
        '--checkpoint', str(ROOT / 'runs/a6000-srf-pilot-main-v2/checkpoints/group_000010.pt'),
        '--checkpoint-sha256', BEHAVIOR_SHA, '--offset', str(OFFSET), '--count', str(COUNT),
        '--output', str(OUTPUT), '--main-state', str(PROVISION / 'pipeline_state_v2.json'),
        '--monitor-dir', str(ROOT / 'runs/a6000-srf-offline-extension26/tensorboard')]
    with log.open('x') as stream:
        process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
        base.state('collecting_extension_26', pid=process.pid, log=str(log),
                   capacity=capacity, target_total_groups=50)
        code = process.wait()
    if code:
        raise RuntimeError(f'Extension collector exited {code}; preserve all files')
    base.state('verifying_extension_and_preserved_first_shard')
    extension = base.verify_dataset()
    base.write_json(PLAN / 'PERSISTENT_VERIFICATION.json', extension)
    # Re-read every first-shard member at completion, not merely its old receipt.
    base.OUTPUT, base.COUNT, base.OFFSET = FIRST, 24, 102
    try:
        first_fresh = base.verify_dataset()
    finally:
        configure_base(base)
    if first_fresh != first:
        raise ValueError('Preserved first-shard verification changed')
    cumulative = combine_verified_shards(first_fresh, extension, first_manifest,
        read_json(OUTPUT / 'COLLECTION_MANIFEST.json'))
    base.write_json(PLAN / 'CUMULATIVE_VERIFICATION.json', cumulative)
    base.state('offline_50_groups_complete_all_members_verified', summary=cumulative)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OWNED_STATE:
            load_base().write_json(STATE, {'phase': 'failed', 'utc_unix': time.time(),
                'output': str(OUTPUT), 'error_type': type(error).__name__, 'message': str(error),
                'target_total_groups': 50, 'new_groups': COUNT})
        raise
