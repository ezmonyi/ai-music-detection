"""Detached server-side private backup. Authentication exists only in process memory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = 'EZMONYI/music-ai-human-interpretable-results'
BASE = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RUN = BASE / 'hf_backup_20260909'
HF = '/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903/venv/bin/hf'
AUDIO = ['*.wav', '*.flac', '*.mp3', '*.m4a', '*.ogg', '*.opus', '*.aif', '*.aiff', '*.mid', '*.midi']
EXCLUDE = ['**/.cache/**', '**/.git/**', '**/venv/**', '**/.venv/**', '**/site-packages/**',
           '**/__pycache__/**', '**/node_modules/**', '**/writer.lock']

def save(value):
    temporary = RUN / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(RUN / 'status.json')

def worker(token, kind='remote'):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    if api.whoami()['name'].lower() != 'ezmonyi':
        raise RuntimeError('Authenticated account is not EZMONYI')
    if not api.dataset_info(REPO).private:
        raise RuntimeError('Private backup refuses a public repository')
    state = {'status': 'running', 'pid': os.getpid(), 'repo': REPO, 'private': True,
             'completed_jobs': [], 'failed_jobs': [], 'started_at': time.time()}
    state_path = RUN / 'status.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        state['completed_jobs'] = previous.get('completed_jobs', [])
    completed = {j['name'] for j in state['completed_jobs']}
    env = dict(os.environ, HF_TOKEN=token, HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
    jobs = []
    data = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
    for name in ('native30_sdrp_operational_v1', 'native30_bc_v2'):
        root = data / name
        jobs.append((name + '_root', str(root), 'snapshots/2026-09-09/remote_results/' + name, ['*.json'], ['items/*', 'metadata/*', 'arrays/*', 'runs/*', 'failures/*']))
        for folder in ('items', 'metadata', 'arrays', 'runs', 'failures'):
            if (root / folder).is_dir() and any((root / folder).iterdir()):
                jobs.append((name + '_' + folder, str(root / folder), 'snapshots/2026-09-09/remote_results/' + name + '/' + folder, ['*.json', '*.npz'], []))
    plan = json.loads((BASE / 'code/music_backup_remote_plan.json').read_text())
    for number, entry in enumerate(plan['roots']):
        if Path(entry['path']).is_dir():
            jobs.append(('music_' + str(number), entry['path'], entry['repo_prefix'], AUDIO, EXCLUDE))
    if kind == 'local':
        jobs = [('local_results', str(RUN / 'results'), 'snapshots/2026-09-09/results', ['*.tar.gz', '*.json', '*.md'], []),
                ('root_card', str(RUN / 'results/README.md'), 'README.md', [], []),
                ('local_music', str(RUN / 'music'), 'music/local', AUDIO, EXCLUDE)]
    state['planned_jobs'] = len(jobs)
    save(state)
    for name, source, prefix, includes, excludes in jobs:
        if name in completed:
            continue
        if kind == 'local':
            marker = RUN / ('music.ready' if name == 'local_music' else 'results.ready')
            while not marker.is_file():
                state.update(status='waiting_for_local_transfer', current_job=name, marker=str(marker), updated_at=time.time())
                save(state)
                time.sleep(15)
            state['status'] = 'running'
        state.update(current_job=name, current_source=source, updated_at=time.time())
        save(state)
        command = [HF, 'upload', REPO, source, prefix, '--type', 'dataset',
                   '--commit-message', 'Server research backup: ' + name, '--format', 'json']
        for pattern in includes:
            command += ['--include', pattern]
        for pattern in excludes:
            command += ['--exclude', pattern]
        log = RUN / (name + '.log')
        for attempt in range(1, 4):
            print(json.dumps({'job': name, 'attempt': attempt, 'source': source}), flush=True)
            with log.open('ab') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                state['completed_jobs'].append({'name': name, 'source': source, 'prefix': prefix,
                                               'status': 'hf_cli_upload_success', 'finished_at': time.time()})
                break
            tail = log.read_bytes()[-16384:].decode(errors='replace').lower()
            if any(s in tail for s in ('storage limit', 'storage quota', '402', 'payment required', '401 unauthorized', '403 forbidden')):
                state.update(status='blocked_capacity_or_authorization', current_log=str(log), updated_at=time.time())
                save(state)
                return
            if attempt < 3:
                time.sleep(30 * attempt)
        else:
            state['failed_jobs'].append({'name': name, 'log': str(log), 'exit_code': result.returncode})
        save(state)
    state.update(status='uploads_finished_pending_full_inventory_verification' if not state['failed_jobs'] else 'partial_upload_failures', updated_at=time.time())
    save(state)

def main():
    global RUN
    payload = json.load(sys.stdin)
    token = payload['token']
    kind = payload.get('kind', 'remote')
    if kind not in ('remote', 'local'):
        raise ValueError('Unknown backup kind')
    if kind == 'local':
        RUN = BASE / 'hf_backup_local_20260909'
    RUN.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid:
        print(json.dumps({'started_pid': pid, 'status': str(RUN / 'status.json'), 'log': str(RUN / 'controller.log')}), flush=True)
        return
    os.setsid()
    with open(os.devnull, 'rb') as source, (RUN / 'controller.log').open('ab', buffering=0) as target:
        os.dup2(source.fileno(), 0)
        os.dup2(target.fileno(), 1)
        os.dup2(target.fileno(), 2)
        try:
            worker(token, kind)
        except BaseException as exc:
            save({'status': 'controller_failed', 'error_type': type(exc).__name__,
                  'message': str(exc).replace(token, '[REDACTED]'), 'pid': os.getpid()})
        finally:
            os._exit(0)

if __name__ == '__main__':
    main()
