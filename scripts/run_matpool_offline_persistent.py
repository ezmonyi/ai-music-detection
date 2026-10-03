"""User-reduced, no-proxy collection directly onto the existing persistent disk."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/ai-music-rl-20261003')
PROVISION = ROOT / 'provisioning'
OUTPUT = ROOT / 'data/acestep_grpo_offline_g10_24x4_20261003'
PLAN = ROOT / 'data/offline_persistent_plan_20261003'
STATE = PROVISION / 'offline_persistent_pipeline_state.json'
METRICS = ROOT / 'runs/a6000-srf-pilot-main-v2/metrics.jsonl'
COUNT, OFFSET, GROUP_SIZE = 24, 102, 4
WAV_BYTES, TRAJECTORY_BYTES, CHECKPOINT_BYTES = 11520088, 9796301, 508991837
SNAPSHOT_BUDGET_BYTES, SAFETY_RESERVE_BYTES = 26_000_000_000, 4 * 2**30
OWNED_STATE = False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def state(phase, **values):
    result = {'phase': phase, 'utc_unix': time.time(), 'output': str(OUTPUT), **values}
    write_json(STATE, result)
    print(json.dumps(result), flush=True)


def capacity_projection(groups, free_bytes):
    if not 0 <= groups <= 100:
        raise ValueError('Unexpected online group count')
    remaining = 100 - groups
    future_checkpoints = (10 - groups // 10) * CHECKPOINT_BYTES
    future_validation = (10 - groups // 10) * 20 * WAV_BYTES
    future_test = 100 * WAV_BYTES
    # Worst case: every remaining candidate/base pair is retained by a guard.
    future_training_audio = remaining * GROUP_SIZE * 2 * WAV_BYTES
    future_online = future_checkpoints + future_validation + future_test + future_training_audio + 128 * 2**20
    dataset_raw = COUNT * (GROUP_SIZE * (2 * WAV_BYTES + TRAJECTORY_BYTES) + 3_000_000 + 400_000)
    dataset_budget = int((dataset_raw + 169603855 + 16 * 2**20) * 1.2)
    needed = future_online + dataset_budget + SNAPSHOT_BUDGET_BYTES + SAFETY_RESERVE_BYTES
    return {'free_bytes': free_bytes, 'new_dataset_budget_bytes': dataset_budget,
            'remaining_online_worst_case_bytes': future_online,
            'environment_snapshot_budget_bytes': SNAPSHOT_BUDGET_BYTES,
            'safety_reserve_bytes': SAFETY_RESERVE_BYTES, 'needed_free_bytes': needed,
            'capacity_gate_passed': free_bytes >= needed, 'online_groups_observed': groups}


def verify_dataset():
    from music_detector.rl.offline_archive import hashes, safe_member, verified_group
    manifest = json.loads((OUTPUT / 'COLLECTION_MANIFEST.json').read_text())
    summary = json.loads((OUTPUT / 'SUMMARY.json').read_text())
    final_state = json.loads((OUTPUT / 'collection_state.json').read_text())
    if (manifest['prompt_groups'] != COUNT or manifest['selection_offset'] != OFFSET or
            summary['groups_completed'] != COUNT or summary['candidate_count'] != COUNT * GROUP_SIZE or
            summary['audio_files'] != COUNT * GROUP_SIZE * 2 or summary['optimizer_updates'] != 0 or
            final_state['phase'] != 'complete' or final_state['groups_completed'] != COUNT):
        raise ValueError('Reduced collection is not complete')
    for index in range(COUNT):
        verified_group(OUTPUT, index, manifest)
    files = json.loads((OUTPUT / 'FILES_SHA256.json').read_text())['files']
    names = {entry['path'] for entry in files}
    if len(names) != len(files):
        raise ValueError('Duplicate member in final source manifest')
    actual = {str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob('*') if p.is_file()}
    if actual != names | {'FILES_SHA256.json', 'collection_state.json'}:
        raise ValueError('Missing, extra or incomplete dataset members')
    for entry in files:
        digest = hashes(safe_member(OUTPUT, entry['path']))
        if any(digest[key] != entry[key] for key in ('bytes', 'sha256')):
            raise ValueError('Persistent file SHA-256/size mismatch: ' + entry['path'])
    wavs = list(OUTPUT.glob('groups/*/*.wav'))
    trajectories = list(OUTPUT.glob('groups/*/*_trajectory.pt'))
    if len(wavs) != COUNT * GROUP_SIZE * 2 or len(trajectories) != COUNT * GROUP_SIZE:
        raise ValueError('Missing WAV or trajectory')
    generated_metadata = {name: hashes(OUTPUT / name) for name in ('FILES_SHA256.json', 'collection_state.json')}
    return {'status': 'complete_persistent_dataset_all_members_sha256_verified',
            'prompt_groups': COUNT, 'candidates': len(trajectories), 'paired_wavs': len(wavs),
            'optimizer_updates': 0,
            'bytes_verified': sum(e['bytes'] for e in files) + sum(v['bytes'] for v in generated_metadata.values()),
            'files_verified': len(files) + len(generated_metadata),
            'generated_metadata': generated_metadata,
            'manifest_sha256': generated_metadata['FILES_SHA256.json']['sha256'],
            'snapshot_complete': False, 'drive_upload_required': False, 'proxy_used': False}


def main():
    global OWNED_STATE
    if STATE.exists() or OUTPUT.exists() or PLAN.exists():
        raise FileExistsError('Never duplicate or overwrite a reduced collection')
    online = json.loads((PROVISION / 'pipeline_state_v2.json').read_text())
    if online['phase'] == 'failed':
        raise RuntimeError('Online run failed; inspect before collection')
    latest = json.loads(METRICS.read_text().splitlines()[-1])
    capacity = capacity_projection(latest['groups_completed'], shutil.disk_usage(ROOT).free)
    print(json.dumps({'capacity': capacity}), flush=True)
    if not capacity['capacity_gate_passed']:
        raise RuntimeError('Insufficient headroom for online worst case, snapshot and safety reserve')
    PLAN.mkdir(exist_ok=False)
    shutil.copy2(__file__, PLAN / Path(__file__).name)
    write_json(PLAN / 'CAPACITY_AT_LAUNCH.json', capacity)
    write_json(PLAN / 'USER_SCOPE.json', {'user_requested_reduced_persistent_storage': True,
        'temporary_vpn_permission': False, 'original_300_group_plan_superseded_for_this_run': True,
        'new_prompt_groups': COUNT, 'new_candidates': COUNT * GROUP_SIZE,
        'new_paired_wavs': COUNT * GROUP_SIZE * 2, 'existing_probe_groups_retained_separately': 2,
        'selection': f'sorted train[{OFFSET}:{OFFSET + COUNT}]',
        'online_sampler_reward_guards_unchanged': True, 'offline_optimizer_updates': 0})
    OWNED_STATE = True
    log = PROVISION / 'offline_persistent_collect.log'
    env = dict(os.environ, PYTHONPATH='/root/music-rl-20261003/offline-runtime/src')
    for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
        env.pop(key, None)
    data = ROOT / 'data/musiccaps_a6000_curated_500_20261003'
    args = [sys.executable, '-u', '-m', 'music_detector.rl.offline_collect',
        '--config', str(PROVISION / 'acestep_a6000_srf_v2_resolved.json'),
        '--train-data', str(data / 'train.jsonl'), '--validation-data', str(data / 'validation.jsonl'),
        '--test-data', str(data / 'test.jsonl'),
        '--checkpoint', str(ROOT / 'runs/a6000-srf-pilot-main-v2/checkpoints/group_000010.pt'),
        '--checkpoint-sha256', '909c003d33c9a150c6ac947969ccf139f4f4ae52d9a1d8ab5f8e36b1958047e4',
        '--offset', str(OFFSET), '--count', str(COUNT), '--output', str(OUTPUT),
        '--main-state', str(PROVISION / 'pipeline_state_v2.json'),
        '--monitor-dir', str(ROOT / 'runs/a6000-srf-offline-persistent-24/tensorboard')]
    with log.open('x') as stream:
        process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
        state('collecting_reduced_persistent_dataset', pid=process.pid, log=str(log), capacity=capacity)
        code = process.wait()
    if code:
        raise RuntimeError(f'Collector exited {code}; preserve all files and inspect')
    state('verifying_persistent_dataset')
    receipt = verify_dataset()
    write_json(PLAN / 'PERSISTENT_VERIFICATION.json', receipt)
    state('collection_complete_persistent_sha256_verified', summary=receipt)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OWNED_STATE:
            state('failed', error_type=type(error).__name__, message=str(error))
        raise
