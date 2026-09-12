"""Resume Saraga through HTTP/LFS after the original Xet worker is stopped."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import fcntl
import json
import sys
from huggingface_hub.utils._runtime import is_xet_available
import publish_saraga_originals_v1 as publication


def main():
    assert not is_xet_available()
    assert not (publication.OUT/'COMMIT.json').exists()
    # Refuse to duplicate the diagnosed original worker.
    path = '/proc/3193400/cmdline'
    if os.path.exists(path):
        with open(path, 'rb') as stream:
            assert b'publish_saraga_originals_v1.py' not in stream.read()
    token = json.load(sys.stdin)['token']
    lock = (publication.OUT/'http_resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid, transport='HTTP/LFS', xet_disabled=True)), flush=True)
        return
    os.setsid()
    log = os.open(publication.OUT/f'http_resume_{os.getpid()}.log', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.dup2(log, 1); os.dup2(log, 2)
    null = os.open('/dev/null', os.O_RDONLY); os.dup2(null, 0)
    try: publication.worker(token)
    except Exception as exc:
        print('Worker failed: '+type(exc).__name__, flush=True)
        os._exit(1)
    os._exit(0)


if __name__ == '__main__': main()
