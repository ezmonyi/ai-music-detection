"""Resume two confirmed-terminal publishers after an explicit HF cooldown."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time

TARGETS = {'open_models': ('publish_open_model_test_views_v1', 3221467),
           'fma': ('publish_fma_test_views_v1', 3225239),
           'aime': ('publish_aime_test_views_v1', 3222621)}


def retry_delay(error):
    """Honor longer server delays and distinguish hourly commit quotas."""
    response = getattr(error, 'response', None)
    if getattr(response, 'status_code', None) != 429:
        return None
    retry = response.headers.get('Retry-After', '360')
    floor = 3900 if 'commits' in str(error).lower() else 360
    return max(floor, int(retry)) if retry.isdigit() else floor


def main(target, cooldown_override=None):
    name, prior_pid = TARGETS[target]
    assert not Path(f'/proc/{prior_pid}').exists(), 'Prior process still exists; inspect before resuming'
    module = importlib.import_module(name)
    assert not (module.OUT/'COMMIT.json').exists()
    lock = (module.OUT/'resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token = json.load(sys.stdin)['token']
    cooldown = cooldown_override if cooldown_override is not None else (3900 if target == 'aime' else 360)
    assert cooldown >= 360
    pid = os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid,target=target,initial_cooldown_s=cooldown)),flush=True)
        return
    os.setsid()
    fd = os.open(module.OUT/f'resume_{os.getpid()}.log', os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.dup2(fd,1); os.dup2(fd,2)
    null = os.open('/dev/null',os.O_RDONLY); os.dup2(null,0)
    from bounded_hf_http_v1 import install
    install()
    time.sleep(cooldown)
    for attempt in range(4):
        try:
            module.worker(token)
            os._exit(0)
        except Exception as error:
            response = getattr(error,'response',None)
            status = getattr(response,'status_code',None)
            print(json.dumps(dict(attempt=attempt+1,error_type=type(error).__name__,status=status)),flush=True)
            delay = retry_delay(error)
            if delay is None or attempt == 3: os._exit(1)
            time.sleep(delay)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('target',choices=TARGETS)
    p.add_argument('--cooldown-seconds', type=int)
    args=p.parse_args()
    main(args.target, args.cooldown_seconds)
