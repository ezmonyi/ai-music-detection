#!/usr/bin/env python3
"""Verify every copied raw MP3 against a passed local physical-byte audit."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def verify(root,audit_path):
    audit=json.loads(audit_path.read_text());ah=sha(audit_path)
    if audit['status']!='passed' or audit['selected']!=500 or not audit['all_original_files_rehashed'] or audit['classifier_admission_authorized']:
        raise ValueError('No passed non-admitting local audit')
    rows=audit['rows'];ids=[r['id'] for r in rows]
    if len(ids)!=500 or len(set(ids))!=500 or not all(i.isdigit() for i in ids):raise ValueError('Invalid IDs')
    if sha(root/'contract.json')!=audit['contract_sha256'] or sha(root/'summary.json')!=audit['summary_sha256']:raise ValueError('Copied contract/summary changed')
    if {p.stem for p in (root/'raw/mureka_v9').glob('*.mp3')}!=set(ids):raise ValueError('Raw inventory mismatch')
    total=0
    for row in rows:
        p=root/'raw/mureka_v9'/(row['id']+'.mp3')
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Copied raw changed: '+row['id'])
        total+=p.stat().st_size
    s=json.loads((root/'summary.json').read_text())
    if {p.name for p in (root/'items').glob('*.json')}!=set(s['receipts_sha256']):raise ValueError('Receipt inventory mismatch')
    for name,digest in s['receipts_sha256'].items():
        if Path(name).name!=name or sha(root/'items'/name)!=digest:raise ValueError('Copied receipt changed')
    if sha(audit_path)!=ah or total!=audit['bytes']:raise ValueError('Audit/total changed')
    return dict(status='passed',rows=len(rows),bytes=total,local_audit_sha256=ah,
        remote_root=str(root),all_raw_files_rehashed=True,all_item_receipts_rehashed=True,
        decoded_again=False,classifier_fitted=False,code_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--local-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.root,a.local_audit)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
