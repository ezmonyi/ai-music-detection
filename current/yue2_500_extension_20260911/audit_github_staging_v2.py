"""Hash publication sources and reject common literal credential material."""
import hashlib
import json
from pathlib import Path
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--manifest-name', default='PUBLICATION_MANIFEST_20260912_v2.json')
args = parser.parse_args()
assert Path(args.manifest_name).name == args.manifest_name

ROOT=Path(__file__).resolve().parents[2]/'deliverables/github_staging_20260912'
PATTERN=re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----|\bhf_[A-Za-z0-9]{25,}\b|\bgh[pousr]_[A-Za-z0-9]{25,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b|\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9_-]{32,}\b')
files=[]
for folder in ('current','historical_snapshots'):
    for path in sorted((ROOT/folder).rglob('*')):
        if not path.is_file():continue
        assert not path.is_symlink(),str(path)
        raw=path.read_bytes()
        assert not PATTERN.search(raw),'Potential credential in '+str(path.relative_to(ROOT))
        files.append({'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
manifest={'status':'source_hashes_and_literal_credential_pattern_check_passed',
          'not_a_full_security_audit':True,'remote_repository':'https://github.com/ezmonyi/ai-music-detection',
          'files':files,'prior_archive_provenance':'SOURCE_MANIFEST.json'}
with (ROOT/args.manifest_name).open('x') as stream:json.dump(manifest,stream,indent=2)
print(json.dumps({'files':len(files),'credential_pattern_matches':0}))
