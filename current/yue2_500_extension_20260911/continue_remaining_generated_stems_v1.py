"""Wait for the pilot queue, then preserve 3,900 additional generated stems."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from resume_processed_upload_v1 import retry_delay

BASE = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT = BASE / 'remaining_generated_stem_upload_queue_v1'
DEPENDENCIES = ((3285628, 'hf_diffrhythm_stems_v1'), (3285628, 'hf_ishizaka_stems_v1'))
TARGETS = ('suno', 'open_models', 'udio')


def event(**values):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time_unix=time.time(), **values)) + '\n')


def worker(token):
    import publish_remaining_generated_stems_v1 as publisher
    from bounded_hf_http_v1 import install
    install()
    while True:
        pending = []
        for pid, name in DEPENDENCIES:
            if (BASE / name / 'COMMIT.json').exists():
                continue
            if not Path(f'/proc/{pid}').exists():
                event(status='stopped_dependency_terminal_without_commit', dependency=name, pid=pid)
                return 1
            pending.append(name)
        if not pending:
            break
        time.sleep(30)
    event(status='dependencies_completed')
    for name in TARGETS:
        target_out = BASE / f'hf_remaining_{name}_stems_v1'
        if (target_out / 'COMMIT.json').exists():
            event(status='already_completed', target=name)
            continue
        target_out.mkdir(exist_ok=True)
        with (target_out / 'resume.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for attempt in range(4):
                event(status='starting', target=name, attempt=attempt+1)
                try:
                    publisher.worker(token, name)
                    assert (target_out / 'COMMIT.json').exists()
                    event(status='uploaded_pending_independent_acceptance', target=name)
                    break
                except Exception as error:
                    delay = retry_delay(error)
                    event(status='worker_error', target=name, attempt=attempt+1,
                          error_type=type(error).__name__, retry_delay_s=delay)
                    if delay is None or attempt == 3:
                        return 1
                    time.sleep(delay)
    event(status='queue_uploads_complete_independent_acceptance_still_required')
    return 0


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    token = json.load(sys.stdin)['token']
    pid = os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid, status='waiting_for_current_publishers')), flush=True)
    else:
        os.setsid()
        fd = os.open(OUT / 'worker.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(fd, 1); os.dup2(fd, 2)
        null = os.open('/dev/null', os.O_RDONLY); os.dup2(null, 0)
        event(status='waiting_for_dependencies', targets=list(TARGETS))
        os._exit(worker(token))

