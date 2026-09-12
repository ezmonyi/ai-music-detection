"""Detached resume with process lock and HF-directed commit rate-limit backoff."""
import argparse
import fcntl
import importlib
import json
import os
import sys
import time
from huggingface_hub import HfApi


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('module',choices=['publish_aime_originals_v1','publish_open_model_audio_v1','publish_mureka_originals_v1','publish_suno_originals_v1','publish_mtt_clips_v1','publish_early_fma_originals_v1','publish_early_suno_originals_v1'])
    args=parser.parse_args()
    module=importlib.import_module(args.module)
    token=json.load(sys.stdin)['token']
    module.OUT.mkdir(exist_ok=True)
    lock=(module.OUT/'resume.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid,module=args.module)),flush=True)
        return
    os.setsid()
    log=os.open(module.OUT/f'resume_{os.getpid()}.log',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.dup2(log,1);os.dup2(log,2)
    null=os.open('/dev/null',os.O_RDONLY);os.dup2(null,0)
    original=HfApi.create_commit
    def create_commit(self,*a,**kw):
        while True:
            try:return original(self,*a,**kw)
            except Exception as exc:
                response=getattr(exc,'response',None)
                if response is None or response.status_code != 429:raise
                delay=3900
                retry=response.headers.get('Retry-After','')
                if retry.isdigit():delay=max(delay,int(retry))
                print(f'HTTP 429: respecting server limit; retry in {delay} seconds',flush=True)
                time.sleep(delay)
    HfApi.create_commit=create_commit
    try:
        print('Resume scheduled after known hourly commit limit; wait 3900 seconds',flush=True)
        time.sleep(3900)
        module.worker(token)
    except Exception as exc:
        # Do not print exception URLs, headers or authentication material.
        print('Worker failed: '+type(exc).__name__,flush=True)
        os._exit(1)
    os._exit(0)


if __name__=='__main__':main()
