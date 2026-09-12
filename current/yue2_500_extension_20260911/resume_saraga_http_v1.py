"""Resume Saraga through HTTP/LFS after the original Xet worker is stopped."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import fcntl
import json
import sys
import time
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
    attempt = 0
    while True:
        try:
            publication.publish(token)
            break
        except Exception as exc:
            attempt += 1
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', None)
            kind = type(exc).__name__
            print(json.dumps(dict(error_type=kind, http_status=status, attempt=attempt)), flush=True)
            transient = status in (429, 500, 502, 503, 504) or kind in (
                'ReadTimeout', 'ConnectTimeout', 'ConnectionError', 'RemoteProtocolError',
                'TimeoutError', 'ReadTimeoutError')
            if not transient or attempt >= 4:
                os._exit(1)
            delay = 3900 if status == 429 else 30*attempt
            retry = response.headers.get('Retry-After', '') if response is not None else ''
            if retry.isdigit(): delay = max(delay, int(retry))
            print(f'Resume from verified receipts after {delay} seconds', flush=True)
            time.sleep(delay)
    os._exit(0)


if __name__ == '__main__': main()
