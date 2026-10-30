"""Protect terminal outputs only; never trains, changes protocol, or releases nodes.

Runs on the single current cloud node after PIPELINE_EXIT exists. Original /mnt
inputs and system-cache inputs are excluded. Archives are lossless, split below
the authenticated transfer limit, and re-read member by member before delivery.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import tarfile
import threading
import time

ROOT=Path('/root/music-offline-grpo1000-v2-20261031-2128889')
WORK=Path('/mnt/ai-music-offline-grpo1000-v2-20261031-2128889')
NODE=2128889


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=Path(str(path)+'.writing')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


class PartWriter:
    def __init__(self,base,limit=1024**3):
        self.base=Path(base);self.limit=limit;self.paths=[];self.stream=None;self.size=0
    def write(self,data):
        length=len(data);view=memoryview(data)
        while view:
            if self.stream is None:
                path=Path(str(self.base)+f'.part{len(self.paths):03d}')
                self.stream=path.open('xb');self.paths.append(path);self.size=0
            block=view[:self.limit-self.size];self.stream.write(block);self.size+=len(block);view=view[len(block):]
            if self.size==self.limit:self._close()
        return length
    def _close(self):
        if self.stream:
            self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.stream=None
    def flush(self):
        if self.stream:self.stream.flush()
    def close(self):self._close()


class PartReader:
    def __init__(self,paths):self.paths=iter(paths);self.stream=None
    def read(self,n=-1):
        if n<0:raise ValueError('Only bounded archive reads are allowed')
        chunks=[];remaining=n
        while remaining:
            if self.stream is None:
                path=next(self.paths,None)
                if path is None:break
                self.stream=Path(path).open('rb')
            block=self.stream.read(remaining)
            if block:chunks.append(block);remaining-=len(block)
            else:self.stream.close();self.stream=None
        return b''.join(chunks)
    def close(self):
        if self.stream:self.stream.close();self.stream=None


def verify_archive(paths,members):
    expected={m['path']:m for m in members};seen=set();reader=PartReader(paths)
    try:
        with tarfile.open(fileobj=reader,mode='r|gz') as archive:
            for info in archive:
                if not info.isfile() or info.name not in expected or info.name in seen:
                    raise ValueError('Unexpected, duplicate, or non-regular archived member')
                entry=expected[info.name];assert info.size==entry['bytes']
                stream=archive.extractfile(info);h=hashlib.sha256()
                for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
                assert h.hexdigest()==entry['sha256'],'Archived member changed'
                seen.add(info.name)
        assert seen==set(expected),'Archive omitted an output'
    finally:reader.close()


def archive_members(members,base,limit=1024**3):
    writer=PartWriter(base,limit)
    try:
        with tarfile.open(fileobj=writer,mode='w|gz',compresslevel=3) as archive:
            for entry in members:
                path=Path(entry['source_path'])
                assert not path.is_symlink() and path.stat().st_size==entry['bytes']
                archive.add(path,arcname=entry['path'],recursive=False)
    finally:writer.close()
    verify_archive(writer.paths,members)
    return writer.paths


def inventory(folder,prefix,exclude=()):
    entries=[]
    if not folder.exists():return entries
    for path in sorted(folder.rglob('*')):
        relative=path.relative_to(folder)
        if relative.parts[0] in exclude:continue
        if path.is_symlink():raise ValueError('No output symlinks are followed')
        if path.is_file():entries.append({'path':str(Path(prefix)/relative),'source_path':str(path),
            'bytes':path.stat().st_size,'sha256':sha(path)})
    return entries


def stop_owned_tensorboard():
    p=WORK/'MONITOR_PROCESSES.json'
    if not p.exists():return {'tensorboard_was_started':False}
    record=json.loads(p.read_text());pid=record['tensorboard_pid'];cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists():
        command=cmd.read_bytes()
        if b'tensorboard.main' not in command or str(ROOT/'runs').encode() not in command:
            raise RuntimeError('Saved monitor PID belongs to a different process')
        assert os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGTERM)
        for _ in range(30):
            if not cmd.exists():break
            time.sleep(.2)
        else:raise RuntimeError('Owned TensorBoard is still active')
    return {'tensorboard_was_started':True,'pid':pid,'stopped':True}


def main():
    assert ROOT.is_dir() and WORK.is_dir()
    terminal=json.loads((WORK/'PIPELINE_EXIT.json').read_text());assert terminal['node_id']==NODE
    lock=(WORK/'.protection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    directory=WORK/'closeout';directory.mkdir(exist_ok=True)
    final=directory/'DELIVERY_MANIFEST.json'
    if final.exists():
        value=json.loads(final.read_text())
        for entry in value['archives']:
            path=Path(entry['source_path']);assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        print('PROTECTION_JSON:'+json.dumps(value),flush=True);return
    done=threading.Event()
    def heartbeat():
        while not done.wait(15):print('PROTECTION_HEARTBEAT:working',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        monitor=stop_owned_tensorboard();write(directory/'CLOUD_MONITORS_STOPPED.json',monitor)
        # PIPELINE_EXIT is written in the controller's finally block after its
        # scientific child and GPU log writer have stopped. Verify the saved
        # scientific stage PID has not remained active before archiving.
        active=json.loads((WORK/'ACTIVE_STAGE_PID.json').read_text())
        command=Path(f"/proc/{active['child_pid']}/cmdline")
        if command.exists() and b'music_detector.rl.' in command.read_bytes():
            raise RuntimeError('A scientific child remains alive; do not freeze or release')
        output_entries=inventory(ROOT/'runs','runs')+inventory(ROOT/'final_test','final_test')
        checkpoints=[m for m in output_entries if Path(m['path']).suffix=='.pt' and '/checkpoints/' in m['path']]
        latest=None
        if checkpoints:
            final_candidates=[m for m in checkpoints if m['path']=='runs/offline-main/checkpoints/final.pt']
            latest=max(final_candidates or checkpoints,key=lambda e:Path(e['source_path']).stat().st_mtime_ns)
            dest=WORK/'protected_checkpoint'/Path(latest['path']).name
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                if shutil.disk_usage(WORK).free<latest['bytes']+2*2**30:
                    raise RuntimeError('Insufficient /mnt room for the last complete checkpoint')
                temp=dest.with_suffix('.pt.partial');shutil.copyfile(latest['source_path'],temp)
                assert sha(temp)==latest['sha256'];temp.replace(dest)
            assert dest.stat().st_size==latest['bytes'] and sha(dest)==latest['sha256']
            latest={**latest,'persistent_copy':str(dest),'persistent_copy_sha256_verified':True}
        complete=terminal.get('formal_optimizer_updates')==1000 and terminal.get('test_prompt_triplets')==50
        requested=[]
        if complete:
            summary=json.loads((ROOT/'final_test/summary.json').read_text())
            assert summary['count']==50 and summary['offline_optimizer_updates']==1000
            test=Path('/mnt/ai-music-rl-20261003/data/musiccaps_a6000_curated_500_20261003/test.jsonl')
            assert sha(test)=='d9090e509eacca21ff64decc945fe31a197dd97d0edc47011b1e694cc52a1406'
            rows=[json.loads(line) for line in test.read_text().splitlines() if line.strip()]
            pairs=[json.loads(line) for line in (ROOT/'final_test/pairs.jsonl').read_text().splitlines() if line.strip()]
            assert len(pairs)==50 and {r['prompt_id'] for r in rows}=={r['prompt_id'] for r in pairs}
            def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
            expected={digest(r['prompt_id'])[:16]+'_'+suffix+'.wav' for r in rows for suffix in ('base','candidate')}
            for entry in output_entries:
                name=Path(entry['path']).name
                if entry['path'].startswith('final_test/') and name in expected:
                    requested.append({**entry,'path':name})
            assert len(requested)==100 and {e['path'] for e in requested}==expected
            for name in ('pairs.jsonl','summary.json'):
                path=ROOT/'final_test'/name
                requested.append({'path':name,'source_path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
            copy=directory/'FIXED_TEST_PROMPTS.jsonl';shutil.copyfile(test,copy)
            requested.append({'path':'FIXED_TEST_PROMPTS.jsonl','source_path':str(copy),'bytes':copy.stat().st_size,'sha256':sha(copy)})
        report=("# ACE-Step offline Flow-GRPO v2 terminal result\n\n"
            f"Node: {NODE} (RTX 3090). Additional real optimizer updates: {terminal.get('formal_optimizer_updates',0)} / 1000.\n\n"
            f"Pipeline terminal record: `{json.dumps(terminal,sort_keys=True)}`\n\n"
            "Independent v2 numerical audit passed on 2,976 new transitions; maximum absolute logprob error 0.000102996826171875. "
            "User-reviewed receiver tolerance was fixed at 0.0003; native/repeat controls stayed at 0.000001. "
            "This is a receiver-specific allowance, not a universal cross-GPU guarantee. Strict defaults and original core were preserved.\n\n"
            +("The fixed held-out 50-prompt comparison contains official SFT base, previous online initialization, and new offline candidate. "
              "Only base/candidate form the requested 50 new-RL comparison pairs. Same prompt, seed, length, sampler and VAE were used.\n\n" if complete else
              "The requested 1,000-update experiment did not complete. No earlier online checkpoint or old audio is being substituted for a new final RL result. "
              "Partial checkpoints and any partial evaluation outputs are preserved exactly as produced.\n\n")+
            "All immutable source training TARs remain on /mnt. Their 132 GB duplicate is excluded from local delivery. "
            "Complete checkpoint history is separately downloaded with per-file SHA-256. Lossless output archives are re-read per member; "
            "available /mnt space is used without expansion. Existing completed environment snapshot 227273 remains the unchanged runtime recovery source.\n")
        (WORK/'TERMINAL_REPORT.md').write_text(report)
        work_entries=inventory(WORK,'work',exclude=('delivery','closeout','protected_checkpoint','.pipeline.lock','.preparation.lock','.protection.lock'))
        members=work_entries+output_entries
        write(directory/'OUTPUT_MEMBERS_SHA256.json',{'node_id':NODE,'members':members,'original_input_cache_excluded':True})
        attempt=ROOT/'delivery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');attempt.mkdir(parents=True)
        archives=[]
        groups=[('work',work_entries)]
        for run in ('offline-smoke','offline-main'):
            group=[m for m in output_entries if m['path'].startswith('runs/'+run+'/') and m not in checkpoints]
            if group:groups.append((run,group))
        loose=[m for m in output_entries if m['path'].startswith('runs/') and m not in checkpoints
            and not any(m['path'].startswith('runs/'+run+'/') for run in ('offline-smoke','offline-main'))]
        if loose:groups.append(('run-boundaries',loose))
        group=[m for m in output_entries if m['path'].startswith('final_test/')]
        if group:groups.append(('final-test',group))
        covered=[m['path'] for _,group in groups for m in group]+[m['path'] for m in checkpoints]
        assert len(covered)==len(set(covered)) and set(covered)=={m['path'] for m in members},'Unique output backup coverage mismatch'
        persistent=WORK/'delivery';persistent.mkdir(exist_ok=True)
        for name,group in groups:
            if not group:continue
            paths=archive_members(group,attempt/(name+'.tar.gz'))
            protected_paths=[]
            for path in paths:
                size=path.stat().st_size;digest_value=sha(path);saved=None
                if shutil.disk_usage(WORK).free>size+2*2**30:
                    saved=persistent/path.name
                    if saved.exists():
                        if saved.stat().st_size!=size or sha(saved)!=digest_value:
                            saved=persistent/(attempt.name+'-'+path.name)
                    if not saved.exists():
                        temp=Path(str(saved)+'.partial');shutil.copyfile(path,temp)
                        assert temp.stat().st_size==size and sha(temp)==digest_value;temp.replace(saved)
                    assert sha(saved)==digest_value;protected_paths.append(saved)
                else:protected_paths.append(path)
                archives.append({'path':'archives/'+path.name,'source_path':str(saved or path),'bytes':size,'sha256':digest_value,
                    'persistent_copy':str(saved) if saved else None,'archive_group':name,'members':len(group)})
                print('PROTECTION_PROGRESS:'+json.dumps({'archive_part':path.name,'bytes':size,'persistent_copy':bool(saved)}),flush=True)
            verify_archive(protected_paths,group)
        receipts=[]
        for path in sorted(directory.glob('*')):
            if path.is_file() and path.name!='DELIVERY_MANIFEST.json':
                receipts.append({'path':'receipts/'+path.name,'source_path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
        value={'node_id':NODE,'utc':datetime.now(timezone.utc).isoformat(),'pipeline_exit':terminal,
            'scientific_success':complete,'actual_additional_optimizer_updates':terminal.get('formal_optimizer_updates',0),
            'all_output_members':members,'checkpoints':checkpoints,'latest_persistent_checkpoint':latest,
            'archives':archives,'receipts':receipts,'requested_downloads':requested,
            'archive_members_sha256_verified':True,'unique_outputs_archived_or_checkpoint_manifested':True,
            'source_training_data_downloaded':False,'runtime_dependencies_changed':False,
            'new_runtime_snapshot_required':False,'existing_completed_runtime_snapshot_id':227273}
        write(final,value)
        print('PROTECTION_JSON:'+json.dumps(value),flush=True)
    finally:done.set();thread.join(timeout=1)


if __name__=='__main__':main()
