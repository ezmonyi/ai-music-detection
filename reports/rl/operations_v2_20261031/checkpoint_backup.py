"""SHA-verified binary transfer over official matcli, never raw SSH credentials."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parent
PREFIX=Path('/Users/yi/miniforge3/envs/matpool-machine-provider')
DEST=Path('/Users/yi/Documents/projects/music/rl_results/offline_grpo1000_v2_20261031_2128889/checkpoint_backup')
WORK='/mnt/ai-music-offline-grpo1000-v2-20261031-2128889'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def receive(stream,destination,*,probe=False,on_file=None,expected_files=None):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    current=None;writer=None;members=[];seen=set();finished=False
    try:
        for line in stream:
            if line.startswith(b'MUSIC_FILE_BEGIN:'):
                if current is not None:raise ValueError('Nested binary-transfer frame')
                current=json.loads(line[len(b'MUSIC_FILE_BEGIN:'):])
                relative=Path(current['path'])
                if relative.is_absolute() or '..' in relative.parts or current['path'] in seen:
                    raise ValueError('Unsafe or duplicate backup path')
                allowed=(str(relative)=='route_probe.bin') if probe else (
                    len(relative.parts)==4 and relative.parts[0]=='runs' and relative.parts[1] in ('offline-smoke','offline-main')
                    and relative.parts[2]=='checkpoints' and relative.suffix=='.pt')
                if expected_files is not None:
                    expected=expected_files.get(str(relative))
                    allowed=expected is not None and all(current.get(k)==expected.get(k) for k in ('bytes','sha256'))
                minimum=0 if expected_files is not None else 1
                if not allowed or type(current['bytes']) is not int or not minimum<=current['bytes']<=2*2**30:
                    raise ValueError('Unapproved file or oversized checkpoint')
                target=destination/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if any(p.is_symlink() for p in [target,*target.parents] if p==destination or destination in p.parents):raise ValueError('Unsafe backup symlink')
                if shutil.disk_usage(destination).free<current['bytes']+2*2**30:raise RuntimeError('Insufficient Mac space for authenticated backup')
                current.update(target=target,partial=target.with_name(target.name+'.partial'),count=0,digest=hashlib.sha256())
                if target.exists():
                    if target.stat().st_size!=current['bytes'] or sha(target)!=current['sha256']:
                        raise FileExistsError('Preserve differing existing artifact')
                    writer=None
                else:writer=current['partial'].open('xb')
            elif line.startswith(b'MUSIC_DATA:'):
                if current is None:raise ValueError('Binary data outside a file frame')
                data=base64.b64decode(line[len(b'MUSIC_DATA:'):].strip(),validate=True)
                current['count']+=len(data)
                if current['count']>current['bytes']:raise ValueError('File payload exceeded its declared size')
                current['digest'].update(data)
                if writer is not None:writer.write(data)
            elif line.startswith(b'MUSIC_FILE_END'):
                if current is None:raise ValueError('File end without header')
                if writer is not None:writer.close();writer=None
                if current['count']!=current['bytes'] or current['digest'].hexdigest()!=current['sha256']:
                    raise ValueError('Binary file size/SHA256 mismatch; partial evidence preserved')
                if not current['target'].exists():current['partial'].rename(current['target'])
                if sha(current['target'])!=current['sha256']:raise ValueError('Local file readback mismatch')
                receipt={k:current[k] for k in ('path','bytes','sha256')}
                receipt['local_path']=str(current['target']);members.append(receipt);seen.add(current['path'])
                if on_file:on_file(receipt)
                current=None
            elif line.startswith(b'MUSIC_TRANSFER_DONE:'):
                end=json.loads(line[len(b'MUSIC_TRANSFER_DONE:'):])
                if current is not None or end['files_sent']!=len(members):raise ValueError('Incomplete binary transfer')
                finished=True;break
            # Only protocol-marked binary is accepted. Login banners and
            # bounded keepalives are ignored and never become artifact bytes.
        if not finished or current is not None:raise RuntimeError('Transport ended before the authenticated completion marker')
        return members
    finally:
        if writer is not None:writer.close()


def cli_command(code):
    wrapper='import base64;exec(base64.b64decode('+repr(base64.b64encode(code.encode()).decode())+'))'
    return [str(PREFIX/'bin/matcli'),'node','exec','--node-id','2128889','--idle-timeout','120','--command','python3 -u -c '+shlex.quote(wrapper)]


SENDER='''
import base64,hashlib,json,os,sys,time
from pathlib import Path
def send(path,relative,*,limit=None):
    length=min(path.stat().st_size,limit) if limit else path.stat().st_size
    h=hashlib.sha256()
    with path.open('rb') as stream:
        remaining=length
        while remaining:
            data=stream.read(min(1024*1024,remaining));assert data
            h.update(data);remaining-=len(data)
    print('MUSIC_FILE_BEGIN:'+json.dumps({'path':relative,'bytes':length,'sha256':h.hexdigest()}),flush=True)
    with path.open('rb') as stream:
        remaining=length
        while remaining:
            data=stream.read(min(65536,remaining));assert data
            sys.stdout.buffer.write(b'MUSIC_DATA:'+base64.b64encode(data)+b'\\n');sys.stdout.buffer.flush();remaining-=len(data)
    print('MUSIC_FILE_END',flush=True)
'''


def main():
    assert Path(sys.prefix).resolve()==PREFIX.resolve(),'Use the matching MatPool skill Conda environment'
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('probe','watch'));args=parser.parse_args()
    lease=json.loads((OPS/'rental_state.json').read_text());assert lease['phase']=='rented' and lease['node_id']==2128889
    DEST.mkdir(parents=True,exist_ok=True)
    ledger_path=DEST/'VERIFIED_FILES.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'node_id':2128889,'members':[]}
    if args.mode=='probe':
        code=SENDER+"\nsend(Path('/mnt/ai-music-rl-20261003/runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt'),'route_probe.bin',limit=16*1024*1024)\nprint('MUSIC_TRANSFER_DONE:'+json.dumps({'files_sent':1}),flush=True)\n"
        destination=DEST/'route_test'
    else:
        exclusion={m['path']:m['sha256'] for m in ledger['members']}
        code=SENDER+"\nknown="+repr(exclusion)+'''\nroot=Path('/root/music-offline-grpo1000-v2-20261031-2128889')
sent=0;started=time.monotonic()
while True:
    for folder in ('offline-smoke','offline-main'):
        for path in sorted((root/'runs'/folder/'checkpoints').glob('*.pt')):
            relative=str(path.relative_to(root))
            if relative in known:continue
            send(path,relative);known[relative]=True;sent+=1
    if Path('/mnt/ai-music-offline-grpo1000-v2-20261031-2128889/PIPELINE_EXIT.json').exists():break
    if time.monotonic()-started>48*3600:raise TimeoutError('Checkpoint backup watch timebox ended')
    print('MUSIC_HEARTBEAT:'+json.dumps({'closed_checkpoints_sent':sent}),flush=True);time.sleep(15)
print('MUSIC_TRANSFER_DONE:'+json.dumps({'files_sent':sent}),flush=True)
'''
        destination=DEST
    env={k:v for k,v in os.environ.items() if not k.startswith('MATPOOL_') and k!='MATPILOT_CONFIRMED_MATPOOL_ACTION'}
    started=time.monotonic()
    with (DEST/(args.mode+'_transport_stderr.log')).open('ab') as error_log:
        child=subprocess.Popen(cli_command(code),stdout=subprocess.PIPE,stderr=error_log,env=env)
        def record(member):
            if args.mode=='watch':
                old={m['path']:m for m in ledger['members']};old[member['path']]=member;ledger['members']=list(old.values())
                temp=ledger_path.with_suffix('.writing');temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(ledger_path)
            print(json.dumps({'status':'local_artifact_SHA256_verified',**member}),flush=True)
        try:members=receive(child.stdout,destination,probe=args.mode=='probe',on_file=record)
        except BaseException:
            child.terminate();child.wait(timeout=30);raise
        code=child.wait(timeout=30)
    if code:raise RuntimeError('Official CLI transfer did not exit successfully')
    if args.mode=='probe':
        assert len(members)==1 and members[0]['bytes']==16*1024*1024
        receipt={'node_id':2128889,'utc':datetime.now(timezone.utc).isoformat(),'binary_sha256_verified':True,
            'probe_bytes':members[0]['bytes'],'probe_sha256':members[0]['sha256'],'source_slice':'first16MiB_only_not_a_complete_checkpoint',
            'elapsed_seconds':time.monotonic()-started,'local_checkpoint_destination':str(DEST),'proxy_or_credential_copy_used':False}
        (DEST/'ROUTE_PROBE_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
        encoded=base64.b64encode(json.dumps(receipt).encode()).decode()
        mark="import base64,json;from pathlib import Path;p=Path("+repr(WORK+'/MAC_BACKUP_ROUTE_READY.json')+");v=base64.b64decode("+repr(encoded)+");p.open('xb').write(v+bytes([10]));assert json.loads(p.read_text())['binary_sha256_verified'] is True;print('MAC_BACKUP_ROUTE_VERIFIED')"
        result=subprocess.run(cli_command(mark),env=env,capture_output=True,text=True,timeout=120)
        if result.returncode or 'MAC_BACKUP_ROUTE_VERIFIED' not in result.stdout:raise RuntimeError('Binary probe passed but persistent route receipt was not acknowledged')
        (destination/'route_probe.bin').unlink() # Verified disposable slice, not an RL checkpoint.
        print(json.dumps({'status':'actual_binary_backup_route_verified','probe_bytes':receipt['probe_bytes'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
    else:print(json.dumps({'status':'checkpoint_watch_completed','new_files':len(members),'all_newly_transferred_files_verified':True}),flush=True)


if __name__=='__main__':main()
