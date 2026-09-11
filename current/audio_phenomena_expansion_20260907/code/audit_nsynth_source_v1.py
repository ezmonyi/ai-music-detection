"""Independent byte, metadata and decoded-PCM check of the acquired source."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import wave

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    source=ROOT/'external_validation/nsynth_test_source_v1'
    marker=json.loads((source/'COMMIT.json').read_text());products=marker['products']
    require(marker['status']=='committed','Uncommitted')
    require(set(products)=={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p!=source/'COMMIT.json'},'inventory')
    for name,r in products.items():
        p=source/name
        require(p.resolve()==p and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],name)
    require(sha(source/'nsynth-test.jsonwav.tar.gz')=='0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c','archive')
    original=json.loads((source/'nsynth-test/examples.json').read_text())
    records=[json.loads(s) for s in (source/'manifest.jsonl').read_text().splitlines()]
    require(len(records)==len(original)==len({r['id'] for r in records})==4096,'count')
    instrument=defaultdict(list); duplicate=Counter()
    for r in records:
        require(r['metadata']==original[r['id']] and r['ai_human_label'] is None and r['role']=='external_measurement_control_only','metadata/role')
        p=source/'nsynth-test/audio'/(r['id']+'.wav')
        require(str(p)==r['path'] and sha(p)==r['file_sha256'],'audio binding')
        with wave.open(str(p),'rb') as w:
            require((w.getnchannels(),w.getframerate(),w.getsampwidth(),w.getnframes())==(1,16000,2,64000),'shape')
            raw=w.readframes(64001)
        require(len(raw)==128000 and hashlib.sha256(raw).hexdigest()==r['pcm_sha256'],'PCM')
        m=r['metadata']; instrument[m['instrument_str']].append(r)
        require(m['note_str']==r['id'] and r['id']==f"{m['instrument_str']}-{m['pitch']:03d}-{m['velocity']:03d}",'identity fields')
        duplicate[r['pcm_sha256']]+=1
    for name,rs in instrument.items():
        require(len({(r['metadata']['instrument'],r['metadata']['instrument_source_str'],r['metadata']['instrument_family_str']) for r in rs})==1,'instrument consistency')
        require(len({r['pcm_sha256'] for r in rs})>=2,'fewer than2 distinct PCM per instrument')
    receipt=dict(passed=True,rows=4096,instruments=len(instrument),products_hashed=len(products),
        source_note_counts=dict(Counter(r['metadata']['instrument_source_str'] for r in records)),
        source_instrument_counts=dict(Counter(rs[0]['metadata']['instrument_source_str'] for rs in instrument.values())),
        unique_pcm_hashes=len(duplicate),exact_duplicate_pcm_groups=sum(n>1 for n in duplicate.values()),
        maximum_duplicate_multiplicity=max(duplicate.values()),
        source_commit_sha256=sha(source/'COMMIT.json'),source_manifest_sha256=sha(source/'manifest.jsonl'),
        audit_code_sha256=sha(Path(__file__)),all_wavs_decoded=True,ai_human_labels_assigned=False,
        scope='Independent stored-source bytes/metadata/PCM replay; not an amplitude-modulation gate or AI detection result')
    with (ROOT/'audit/nsynth_source_parent_acceptance_v1.json').open('x') as f:
        json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
