"""Append verified live TensorBoard bytes to a private local project mirror."""
from datetime import datetime, timezone
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parent
PREFIX=Path('/Users/yi/miniforge3/envs/matpool-machine-provider')
MONITOR=Path('/Users/yi/Documents/projects/music/operations/rl_live_monitor_20261031')
MIRROR=MONITOR/'runtime/tb_mirror/runs'


def append_verified(root,entry):
    relative=Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts or not relative.name.startswith('events.out.tfevents.'):
        raise ValueError('Unsafe TensorBoard mirror path')
    path=Path(root)/relative;path.parent.mkdir(parents=True,exist_ok=True)
    old=path.read_bytes() if path.exists() else b''
    if len(old)!=entry['offset'] or hashlib.sha256(old).hexdigest()!=entry['prefix_sha256']:
        raise ValueError('Local event prefix changed; do not splice unrelated runs')
    delta=base64.b64decode(entry['delta_base64'],validate=True)
    if len(old)+len(delta)!=entry['total_bytes'] or hashlib.sha256(old+delta).hexdigest()!=entry['snapshot_sha256']:
        raise ValueError('TensorBoard delta size/SHA mismatch')
    if delta:
        with path.open('ab') as stream:stream.write(delta);stream.flush()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['snapshot_sha256']:
        raise ValueError('Event mirror readback checksum mismatch')


def main():
    assert Path(sys.prefix).resolve()==PREFIX.resolve()
    MIRROR.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();failures=0
    while time.monotonic()-started<48*3600:
        offsets={str(p.relative_to(MIRROR)):p.stat().st_size for p in MIRROR.rglob('events.out.tfevents.*') if p.is_file()}
        code='''
from pathlib import Path
import base64,gzip,hashlib,json,subprocess
root=Path('/root/music-offline-grpo1000-v2-20261031-2128889/runs')
work=Path('/mnt/ai-music-offline-grpo1000-v2-20261031-2128889')
offsets=OFFSETS
files=[]
for p in root.rglob('events.out.tfevents.*'):
    assert not p.is_symlink() and root.resolve() in p.resolve().parents
    n=p.stat().st_size;assert n<32*1024*1024
    with p.open('rb') as stream:data=stream.read(n)
    key=str(p.relative_to(root));offset=offsets.get(key,0);assert offset<=len(data)
    files.append({'path':key,'offset':offset,'total_bytes':len(data),
        'prefix_sha256':hashlib.sha256(data[:offset]).hexdigest(),'snapshot_sha256':hashlib.sha256(data).hexdigest(),
        'delta_base64':base64.b64encode(data[offset:]).decode()})
value={'files':files,'node_id':2128889,'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}
for name in ('pipeline_state.json','PIPELINE_EXIT.json'):
    path=work/name
    if path.exists():value[name]=json.loads(path.read_text())
print('MUSIC_MONITOR:'+base64.b64encode(gzip.compress(json.dumps(value).encode())).decode())
'''.replace('OFFSETS',repr(offsets))
        wrapper='import base64;exec(base64.b64decode('+repr(base64.b64encode(code.encode()).decode())+'))'
        env={k:v for k,v in os.environ.items() if not k.startswith('MATPOOL_') and k!='MATPILOT_CONFIRMED_MATPOOL_ACTION'}
        try:
            result=subprocess.run([str(PREFIX/'bin/matcli'),'node','exec','--node-id','2128889','--idle-timeout','45',
                '--command','python3 -c '+shlex.quote(wrapper)],env=env,capture_output=True,text=True,timeout=90)
            lines=[line for line in result.stdout.splitlines() if line.startswith('MUSIC_MONITOR:')]
            if result.returncode or len(lines)!=1:raise RuntimeError('Live event snapshot unavailable')
            value=json.loads(gzip.decompress(base64.b64decode(lines[0].removeprefix('MUSIC_MONITOR:'))))
            for entry in value['files']:append_verified(MIRROR,entry)
            status={'utc':datetime.now(timezone.utc).isoformat(),'node_id':2128889,'mirror_event_files':len(value['files']),
                'mirrored_bytes':sum(p.stat().st_size for p in MIRROR.rglob('events.out.tfevents.*') if p.is_file()),
                'gpu':value['gpu'],'pipeline':value.get('pipeline_state.json'), 'all_received_event_snapshots_sha256_verified':True}
            (MONITOR/'runtime/LATEST_MONITOR_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
            failures=0
            if 'PIPELINE_EXIT.json' in value:
                (MONITOR/'runtime/MIRROR_COMPLETE.json').write_text(json.dumps(status,indent=2)+'\n');return
        except Exception as error:
            failures+=1
            (MONITOR/'runtime/MIRROR_ERROR.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                'error_type':type(error).__name__,'error':str(error),'consecutive_failures':failures},indent=2)+'\n')
        time.sleep(30)


if __name__=='__main__':main()
