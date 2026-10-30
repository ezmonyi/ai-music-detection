"""One network read per immutable TAR, bounded local staging, full hash checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

from .numerical_calibration import write_once
from .offline_archive import hashes
from .portable_delivery import _read_archive, inventory


def stage_archive(source, target, expected):
    """Authenticate bytes before extraction; never edit a source or partial file."""
    source, target = Path(source), Path(target)
    if target.exists():
        if hashes(target) != expected:
            raise ValueError('Existing local archive has a different identity')
        return expected
    partial = target.with_name(target.name + '.partial')
    if partial.exists():
        raise FileExistsError('Interrupted staging is preserved; inspect before resuming')
    sha = hashlib.sha256(); md5 = hashlib.md5(usedforsecurity=False); size = 0
    with source.open('rb') as stream, partial.open('xb') as out:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            out.write(block); sha.update(block);md5.update(block);size += len(block)
    actual = {'bytes': size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}
    if actual != expected:
        raise ValueError('Source archive failed full byte/size/SHA256/MD5 verification')
    partial.rename(target)
    return actual


def restore(source, output, receipts, *, expected_index_sha256):
    source, output, receipts = map(Path, (source, output, receipts))
    index_path = source / 'EXPORT_INDEX.json'
    if hashes(index_path)['sha256'] != expected_index_sha256:
        raise ValueError('Original source export index changed')
    index = json.loads(index_path.read_text())
    if index.get('complete') is not True or index['exported_groups'] != 1000 or len(index['archives']) != 42:
        raise ValueError('Expected the complete original 1000-group export')
    if output.exists() or receipts.exists():
        raise FileExistsError('Fresh restore only; do not duplicate active or overwrite prior work')
    payload_bytes = sum(r['archive']['bytes'] for r in index['archives'].values())
    peak_archive = max(r['archive']['bytes'] for r in index['archives'].values())
    if shutil.disk_usage(output.parent).free < payload_bytes + peak_archive + 20*2**30:
        raise RuntimeError('Insufficient measured system disk for full data plus bounded staging and reserve')
    output.mkdir();receipts.mkdir(parents=True)
    lock = (receipts / '.restore.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    staging = output.parent / (output.name + '.archive-staging');staging.mkdir()
    started = time.monotonic(); completed = []

    def state(phase, **extra):
        value = {'utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'archives_completed':len(completed),
            'archives_target':42,'payload_bytes_verified':sum(v['archive']['bytes'] for v in completed),
            'elapsed_seconds':time.monotonic()-started,'optimizer_updates':0,**extra}
        temp=receipts/'state.writing';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(receipts/'state.json')
        print(json.dumps(value),flush=True)

    try:
        write_once(receipts/'SOURCE_INDEX_IDENTITY.json',{'sha256':expected_index_sha256,'payload_bytes':payload_bytes,
            'source':str(source),'output':str(output),'source_archives_modified':False})
        names = ['metadata.tar', *sorted(n for n in index['archives'] if n.startswith('groups_')), 'completion.tar']
        for name in names:
            entry=index['archives'][name]
            state('staging_verified_archive',archive=name)
            local=staging/name
            stage_archive(source/name,local,entry['archive'])
            state('extracting_locally_with_member_sha256',archive=name)
            _read_archive(local,entry['files'],output=output)
            receipt={'name':name,'archive':entry['archive'],'member_count':len(entry['files']),
                'all_member_sha256_verified':True,'source_modified':False}
            write_once(receipts/(name+'.VERIFIED.json'),receipt)
            completed.append(receipt)
            # Only this derived local TAR is removed, after full archive and
            # member verification. The persistent source TAR is untouched.
            local.unlink()
            state('archive_verified_and_extracted',archive=name)
        state('verifying_complete_restored_inventory')
        result=inventory(output,require_complete=True)
        if result['manifest_sha256']!=index['manifest_sha256'] or result['closed_groups']!=1000:
            raise ValueError('Complete restored collection identity mismatch')
        write_once(receipts/'SUMMARY.json',{'status':'all_42_source_archives_and_restored_members_verified',
            'archives':completed,'groups':1000,'trajectories':4000,'candidate_base_wavs':8000,
            'payload_bytes':payload_bytes,'manifest_sha256':result['manifest_sha256'],
            'elapsed_seconds':time.monotonic()-started,'optimizer_updates':0,'source_archives_modified':False})
        state('complete',groups=1000,trajectories=4000)
    except Exception as error:
        write_once(receipts/'FAILURE.json',{'error':str(error),'completed_archives':len(completed),'optimizer_updates':0})
        state('failed_preserved',error=str(error))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','receipts','expected-index-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();restore(a.source,a.output,a.receipts,expected_index_sha256=a.expected_index_sha256)


if __name__=='__main__':main()
