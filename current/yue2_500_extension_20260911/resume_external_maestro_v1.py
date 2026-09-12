"""Resume the terminated external MAESTRO publisher with tested MIDI validation."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import fcntl
import json
import sys
import publish_external_maestro_v1 as publication
from bounded_hf_http_v1 import install


def main():
    install()
    assert not (publication.OUT/'COMMIT.json').exists()
    original='/proc/3204274/cmdline'
    if os.path.exists(original):
        with open(original,'rb') as f:assert b'publish_external_maestro_v1.py' not in f.read()
    token=json.load(sys.stdin)['token']
    lock=(publication.OUT/'resume.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid,module='publish_external_maestro_v1')),flush=True);return
    os.setsid()
    log=os.open(publication.OUT/f'resume_{os.getpid()}.log',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.dup2(log,1);os.dup2(log,2)
    null=os.open('/dev/null',os.O_RDONLY);os.dup2(null,0)
    try:publication.worker(token)
    except Exception as exc:
        root=exc
        while root.__context__ is not None:root=root.__context__
        print(json.dumps(dict(error_type=type(root).__name__,http_status=getattr(getattr(root,'response',None),'status_code',None))),flush=True)
        os._exit(1)
    os._exit(0)


if __name__=='__main__':main()
