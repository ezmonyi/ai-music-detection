"""Recover first-party source snapshots into a publication staging directory."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'deliverables/github_staging_20260912'
SUFFIXES={'.py','.sh','.r','.R','.m','.tex','.bib','.yaml','.yml','.toml','.md'}
DENY={'venv','.venv','site-packages','node_modules','.git','__pycache__','repos','checkpoints','models'}
SECRET=re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP)? ?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{25,}\b|\bgh[pousr]_[A-Za-z0-9]{25,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b')


def main():
    OUT.mkdir(exist_ok=True)
    records=[]
    excluded=[]
    def save(relative,data,source):
        if SECRET.search(data):
            excluded.append({'path':str(relative),'reason':'credential_pattern_quarantined','source':source})
            return
        target=OUT/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert target.read_bytes()==data
        else:
            target.write_bytes(data)
        records.append(dict(path=str(relative),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source=source))
    for archive in sorted((ROOT/'deliverables/local_results_archive_20260911').glob('*.tar.gz')):
        with tarfile.open(archive,'r:gz') as stream:
            for member in stream:
                path=PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts:
                    continue
                if any(part in DENY for part in path.parts) or path.suffix not in SUFFIXES or member.size>5_000_000:
                    continue
                # Explicit source/report-text only; no arbitrary metadata or binary artifacts.
                if not ({'code','scripts','latex'} & set(path.parts)):
                    continue
                data=stream.extractfile(member).read()
                save(Path('historical_snapshots')/archive.name.removesuffix('.tar.gz')/Path(*path.parts),data,archive.name)
    for path in sorted((ROOT/'artifacts').rglob('*')):
        if not path.is_file() or path.is_symlink() or path.suffix not in SUFFIXES or path.stat().st_size>5_000_000:
            continue
        if any(part in DENY for part in path.parts):
            continue
        save(Path('current')/path.relative_to(ROOT/'artifacts'),path.read_bytes(),'local_current')
    manifest=dict(status='staged_not_published_not_complete_secret_audit',files=records,excluded=excluded,
                  credential_scan='private-key and common HF/GitHub literal token patterns; not a full semantic review',
                  historical_git_history_reconstructed=False)
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(records),'quarantined':len(excluded),'output':str(OUT)}))


if __name__=='__main__':
    main()
