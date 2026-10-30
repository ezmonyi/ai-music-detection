"""Owned-node terminal delivery and release. Never rents or launches science.

The user explicitly authorized unattended results download and release, including
protected failure shutdown. This supervisor cannot release before authenticated
archive/checkpoint/local-comparison delivery. All MatPool calls use its official
CLI/cache; no connection credential is extracted or copied.
"""
from datetime import datetime, timezone
import base64
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from checkpoint_backup import cli_command, receive, sha, SENDER

OPS=Path(__file__).resolve().parent
PREFIX=Path('/Users/yi/miniforge3/envs/matpool-machine-provider')
PACKET=Path('/Users/yi/Documents/projects/music/rl_results/offline_grpo1000_v2_20261031_2128889')
BACKUP=PACKET/'checkpoint_backup'
DOWNLOAD=Path('/Users/yi/Downloads/acestep-offline-grpo1000-v2-20261031-2128889')
ROOT='/root/music-offline-grpo1000-v2-20261031-2128889'
WORK='/mnt/ai-music-offline-grpo1000-v2-20261031-2128889'
ENV={k:v for k,v in os.environ.items() if not k.startswith('MATPOOL_') and k!='MATPILOT_CONFIRMED_MATPOOL_ACTION'}


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.writing');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def status(phase,**extra):
    write(OPS/'CLOSEOUT_SUPERVISOR_STATE.json',{'utc':utc(),'node_id':2128889,'phase':phase,**extra})
    print(json.dumps({'utc':utc(),'phase':phase,**extra}),flush=True)


def remote_value(code,marker='OWNED_RESULT:',timeout=180):
    result=subprocess.run(cli_command(code),env=ENV,capture_output=True,text=True,timeout=timeout)
    lines=[line for line in result.stdout.splitlines() if line.startswith(marker)]
    if result.returncode or len(lines)!=1:
        # This command contains only task-owned source, never a password or
        # provider connection response. Preserve diagnosis privately.
        path=PACKET/'closeout_transport_errors';path.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (path/(stamp+'.log')).write_text(result.stdout[-16000:]+result.stderr[-16000:])
        raise RuntimeError('Owned-node command failed; task-local error log preserved')
    return json.loads(lines[0].removeprefix(marker))


def terminal_state():
    return remote_value("""
from pathlib import Path
import json
w=Path(WORK)
v={'node_id':2128889}
for n in ('PIPELINE_EXIT.json','pipeline_state.json'):
 p=w/n
 if p.exists():v[n]=json.loads(p.read_text())
print('OWNED_RESULT:'+json.dumps(v))
""".replace('WORK',repr(WORK)))


def transfer(entries,destination,label):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    verified=[];missing=[]
    for entry in entries:
        path=Path(entry['path'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe local result path')
        source=Path(entry['source_path'])
        allowed=[Path(ROOT)/p for p in ('runs','final_test','delivery')]+[Path(WORK)/p for p in ('closeout','delivery')]
        if not any(base in source.parents for base in allowed):raise ValueError('Result outside task-owned output scope')
        target=destination/path
        if target.exists():
            if target.stat().st_size!=entry['bytes'] or sha(target)!=entry['sha256']:
                raise RuntimeError('Differing existing local result must be preserved')
            verified.append({**entry,'local_path':str(target)})
        else:missing.append(entry)
    for offset in range(0,len(missing),8):
        batch=missing[offset:offset+8]
        # A previous interrupted transfer is never mistaken for a complete
        # artifact or silently overwritten. Retain it under an explicit name.
        for entry in batch:
            partial=(destination/entry['path']).with_name(Path(entry['path']).name+'.partial')
            if partial.exists():partial.rename(partial.with_name(partial.name+'.interrupted-'+str(time.time_ns())))
        expected={e['path']:e for e in batch}
        code=SENDER+'\nentries='+repr(batch)+"""
for entry in entries:
 path=Path(entry['source_path']);assert not path.is_symlink()
 assert path.stat().st_size==entry['bytes']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'] if path.stat().st_size<1024*1024 else True
 send(path,entry['path'])
print('MUSIC_TRANSFER_DONE:'+json.dumps({'files_sent':len(entries)}),flush=True)
"""
        stderr=PACKET/'delivery_transport_stderr.log'
        with stderr.open('ab') as log:
            child=subprocess.Popen(cli_command(code),env=ENV,stdout=subprocess.PIPE,stderr=log)
            def on_file(receipt):
                verified.append({**expected[receipt['path']],**receipt})
                status('result_file_downloaded_sha256_verified',label=label,path=receipt['path'],bytes=receipt['bytes'])
            try:receive(child.stdout,destination,on_file=on_file,expected_files=expected)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
            if child.wait(timeout=30):raise RuntimeError('Official CLI did not confirm result transfer completion')
    if len(verified)!=len(entries):raise RuntimeError('Local result manifest is incomplete')
    for entry in verified:
        path=Path(entry['local_path'])
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    write(PACKET/(label+'_VERIFICATION.json'),{'node_id':2128889,'utc':utc(),'all_member_sha256_verified':True,'members':verified})
    return verified


def provider_proof():
    result=subprocess.run([str(PREFIX/'bin/node'),str(OPS/'inspect_closeout_safe.mjs')],env=ENV,capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError('Official closeout identity/billing/snapshot read failed')
    value=json.loads((OPS/'CLOSEOUT_PROVIDER_LATEST.json').read_text())
    assert value['node_id']==2128889 and value['display_id']=='ZW703Y'
    return value


def acknowledge_delivery(delivery):
    if (PACKET/'REMOTE_DELIVERY_ACK.json').exists():return
    raw=json.dumps(delivery,indent=2)+'\n'
    code='''
import base64,hashlib,json
from pathlib import Path
p=Path(WORK+'/closeout/MAC_COMPLETE_DELIVERY.json');data=base64.b64decode(DATA)
if p.exists():assert p.read_bytes()==data,'Preserve differing delivery evidence'
else:p.open('xb').write(data)
assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
print('OWNED_RESULT:'+json.dumps({'remote_delivery_receipt_sha256_verified':True}))
'''.replace('WORK',repr(WORK)).replace('DATA',repr(base64.b64encode(raw.encode()).decode())).replace('SHA',repr(hashlib.sha256(raw.encode()).hexdigest()))
    value=remote_value(code);assert value['remote_delivery_receipt_sha256_verified'] is True
    write(PACKET/'REMOTE_DELIVERY_ACK.json',{'utc':utc(),'node_id':2128889,'delivery_receipt_sha256':hashlib.sha256(raw.encode()).hexdigest(),**value})


def release_after_delivery(delivery):
    for name in delivery['file_verification_receipts']:
        receipt=json.loads((PACKET/name).read_text());assert receipt['all_member_sha256_verified'] is True
        for entry in receipt['members']:
            path=Path(entry['local_path']);assert path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    acknowledge_delivery(delivery)
    proof=provider_proof()
    assert proof['unchanged_runtime_snapshot_completed'],'Original unchanged environment snapshot must remain protected'
    assert delivery['all_unique_outputs_locally_verified'] is True
    if delivery['scientific_success']:
        assert delivery['requested_pair_wavs_verified']==100 and delivery['actual_additional_optimizer_updates']==1000
    write(PACKET/'PROVIDER_BEFORE_RELEASE.json',proof)
    if not proof['node_released']:
        preview=subprocess.run([str(PREFIX/'bin/matcli'),'node','release','--id','2128889'],env=ENV,capture_output=True,text=True,timeout=90)
        if preview.returncode or '2128889' not in preview.stdout:raise RuntimeError('Official release preview did not match this node')
        (PACKET/'RELEASE_PREVIEW.txt').write_text(preview.stdout)
        intent=OPS/'RELEASE_INTENT.json'
        if not intent.exists():
            write(intent,{'node_id':2128889,'display_id':'ZW703Y','utc':utc(),
                'authorized_by':'latest explicit user unattended completion/download/release and failure-stop instruction',
                'delivery_receipt_sha256':sha(PACKET/'MAC_COMPLETE_DELIVERY.json'),'release_preview_checked':True})
        status('all_outputs_protected_submitting_user_authorized_release')
        try:
            result=subprocess.run([str(PREFIX/'bin/matcli'),'node','release','--id','2128889','--confirm-release'],
                env=ENV,capture_output=True,text=True,timeout=120)
            write(PACKET/'RELEASE_SUBMISSION.json',{'utc':utc(),'node_id':2128889,'exit_code':result.returncode,
                'official_cli_output':result.stdout.strip(),'stderr':result.stderr.strip()})
        except subprocess.TimeoutExpired:
            write(PACKET/'RELEASE_SUBMISSION.json',{'utc':utc(),'node_id':2128889,'outcome':'uncertain_timeout_read_only_confirmation_required'})
    previous=None
    for _ in range(40):
        proof=provider_proof()
        if proof['compute_billing_stopped_confirmed'] and previous is not None:
            if previous['node_released'] and previous['bills']==proof['bills']:
                cli_bill=subprocess.run([str(PREFIX/'bin/matcli'),'node','bill','--request-id','ZW703Y'],env=ENV,capture_output=True,text=True,timeout=90)
                if cli_bill.returncode:raise RuntimeError('Official final node bill CLI failed')
                (PACKET/'OFFICIAL_NODE_BILL.txt').write_text(cli_bill.stdout)
                value={'utc':utc(),'node_id':2128889,'display_id':'ZW703Y','node_released':True,
                    'compute_billing_stopped_confirmed':True,'two_read_only_bill_samples_stable':True,
                    'provider_receipt':proof,'delivery':delivery,'scientific_success':delivery['scientific_success'],
                    'actual_additional_optimizer_updates':delivery['actual_additional_optimizer_updates'],
                    'no_new_rental_or_recharge_or_expansion':True}
                write(PACKET/'FINAL_CLOSEOUT.json',value);write(OPS/'FINAL_CLOSEOUT.json',value)
                for name in ('EXPERIMENT_STATE.json','RUNBOOK.json'):
                    path=OPS/name;old=json.loads(path.read_text());old.update(phase='protected_closed_node_released',current_phase='protected_closed_node_released',
                        formal_optimizer_updates=delivery['actual_additional_optimizer_updates'],compute_billing_stopped_confirmed=True,
                        closeout_receipt=str(PACKET/'FINAL_CLOSEOUT.json'),download_directory=delivery.get('download_directory'),last_checked_utc=utc())
                    write(path,old)
                status('protected_closeout_complete_node_released',scientific_success=delivery['scientific_success'],
                    actual_additional_optimizer_updates=delivery['actual_additional_optimizer_updates']);return value
        previous=proof;time.sleep(15)
    raise RuntimeError('Release was requested but stable released/billing proof remains unconfirmed; do not create or blindly resubmit')


def main():
    assert Path(sys.prefix).resolve()==PREFIX.resolve(),'Use the MatPool provider skill Conda environment'
    rental=json.loads((OPS/'rental_state.json').read_text());assert rental['phase']=='rented' and rental['node_id']==2128889
    PACKET.mkdir(parents=True,exist_ok=True)
    lock=(OPS/'.closeout-supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OPS/'FINAL_CLOSEOUT.json').exists():
        if json.loads((OPS/'FINAL_CLOSEOUT.json').read_text()).get('compute_billing_stopped_confirmed'):return
    delivery_path=PACKET/'MAC_COMPLETE_DELIVERY.json'
    if delivery_path.exists():
        release_after_delivery(json.loads(delivery_path.read_text()));return
    while True:
        value=terminal_state()
        if 'PIPELINE_EXIT.json' in value:break
        status('waiting_existing_pipeline_terminal',pipeline=value.get('pipeline_state.json'))
        time.sleep(60)
    terminal=value['PIPELINE_EXIT.json'];assert terminal['node_id']==2128889
    deployment=json.loads((OPS/'CLOSEOUT_DEPLOY_RESULT.json').read_text())
    source=deployment['cloud_protection_script']
    assert source.startswith(WORK+'/closeout_support-') and source.endswith('/protect_outputs.py')
    status('pipeline_ended_protecting_unique_outputs',actual_updates=terminal.get('formal_optimizer_updates',0),terminal=terminal)
    code='''
import hashlib,json,runpy
from pathlib import Path
p=Path(SOURCE);assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED
runpy.run_path(str(p),run_name='__main__')
'''.replace('SOURCE',repr(source)).replace('EXPECTED',repr(deployment['protection_script_sha256']))
    bundle=remote_value(code,marker='PROTECTION_JSON:',timeout=7200)
    assert bundle['node_id']==2128889 and bundle['archive_members_sha256_verified'] is True
    write(PACKET/'REMOTE_DELIVERY_MANIFEST.json',bundle)
    # Allow the existing watcher to finish any in-flight checkpoint first.
    # A transport failure is recoverable without restarting science.
    last_progress=time.monotonic();last_count=-1
    while True:
        watch=json.loads((OPS/'CHECKPOINT_BACKUP_WATCH.json').read_text())
        result=subprocess.run(['ps','-p',str(watch['pid']),'-o','command='],capture_output=True,text=True)
        alive=result.returncode==0 and ('conda' in result.stdout or 'checkpoint_backup.py' in result.stdout or 'miniforge3/bin/python' in result.stdout)
        count=len(list(BACKUP.glob('runs/*/checkpoints/*.pt')))
        if count!=last_count:last_count=count;last_progress=time.monotonic()
        if not alive:break
        if time.monotonic()-last_progress>1200:raise RuntimeError('Checkpoint watcher has made no complete-file progress for 20 minutes; preserve and diagnose')
        status('waiting_checkpoint_watcher_flush',completed_local_checkpoints=count);time.sleep(15)
    checkpoints=transfer(bundle['checkpoints'],BACKUP,'ALL_CHECKPOINTS')
    archives=transfer(bundle['archives']+bundle['receipts'],PACKET/'protected_outputs','OUTPUT_ARCHIVES')
    requested=[]
    if bundle['scientific_success']:
        requested=transfer(bundle['requested_downloads'],DOWNLOAD,'FIXED_50_TEST_PAIRS')
        wavs=[m for m in requested if m['path'].endswith('.wav')]
        assert len(wavs)==100
        (DOWNLOAD/'README.md').write_text('# Fixed held-out official / offline-RL comparison\n\n'
            '50 prompts, matched seeds and sampling settings. `*_base.wav` is the official open SFT model; '
            '`*_candidate.wav` is the new offline continuation after 1,000 additional real optimizer updates. '
            '`pairs.jsonl` contains frozen-metric results and admission diagnostics; invalid samples are retained. '
            'The prior online initialization is preserved separately in the complete archive as `*_initial.wav`.\n')
        write(DOWNLOAD/'FILES_SHA256.json',{'node_id':2128889,'members':requested,'all_member_sha256_verified':True})
    delivery={'utc':utc(),'node_id':2128889,'scientific_success':bundle['scientific_success'],
        'actual_additional_optimizer_updates':bundle['actual_additional_optimizer_updates'],
        'all_unique_outputs_locally_verified':True,'archive_members_verified_in_cloud':True,
        'all_checkpoint_sha256_verified':True,'checkpoint_count':len(checkpoints),'archive_and_receipt_files':len(archives),
        'latest_checkpoint_persistent_copy':bundle['latest_persistent_checkpoint'],
        'requested_pair_wavs_verified':sum(e['path'].endswith('.wav') for e in requested),
        'download_directory':str(DOWNLOAD) if requested else None,'local_packet':str(PACKET),
        'cloud_persistent_root':WORK,'input_132GB_downloaded':False,
        'file_verification_receipts':['ALL_CHECKPOINTS_VERIFICATION.json','OUTPUT_ARCHIVES_VERIFICATION.json']+(['FIXED_50_TEST_PAIRS_VERIFICATION.json'] if requested else [])}
    write(delivery_path,delivery)
    release_after_delivery(delivery)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('supervisor_failed_preserved_requires_diagnosis',error_type=type(error).__name__,error=str(error))
        raise
