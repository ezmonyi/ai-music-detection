"""Acquire the official NSynth test release for measurement controls, not AI labels."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile
import urllib.request
import wave

PAGE='https://magenta.tensorflow.org/datasets/nsynth'
OFFICIAL='http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz'
DOWNLOAD='https://storage.googleapis.com/download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz'
BYTES=349501546
MD5='5e6f8719bf7e16ad0a00d518b78af77d'
GENERATION='1491603911045000'


def require(ok,message):
    if not ok: raise ValueError(message)


def checksum(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def save(path,data):
    with path.open('x') as f: json.dump(data,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def acquire(root):
    root.mkdir(exist_ok=False)
    # Save the exact source/terms document; no access forms or credentials.
    with urllib.request.urlopen(PAGE,timeout=45) as response:
        page=response.read()
    require(OFFICIAL.encode() in page and b'https://creativecommons.org/licenses/by/4.0/' in page,
        'Official page no longer supplies the registered release and license')
    with (root/'official_dataset_page.html').open('xb') as f: f.write(page)
    archive=root/'nsynth-test.jsonwav.tar.gz'
    with urllib.request.urlopen(DOWNLOAD,timeout=60) as response:
        headers=dict(response.headers.items())
        require(int(response.headers['Content-Length'])==BYTES and response.headers['x-goog-generation']==GENERATION,
                'Remote version/length differs')
        require(response.headers['ETag'].strip('"')==MD5,'Remote MD5/ETag differs')
        size=0
        with archive.open('xb') as f:
            for block in iter(lambda:response.read(1048576),b''):
                size+=len(block); require(size<=BYTES,'Overlong response'); f.write(block)
    require(size==BYTES and checksum(archive,'md5')==MD5,'Archive length/MD5 mismatch')
    print('archive downloaded and MD5 verified',flush=True)
    # Force gzip EOF/CRC, including content after the tar end marker.
    with gzip.open(archive,'rb') as f:
        expanded=0
        for block in iter(lambda:f.read(1048576),b''):
            expanded+=len(block); require(expanded<2_000_000_000,'Unexpected expanded size')
    members={}; total=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            p=PurePosixPath(member.name)
            require(not p.is_absolute() and '..' not in p.parts and p.parts[0]=='nsynth-test','Unsafe archive path')
            require(member.name not in members,'Duplicate archive member')
            members[member.name]=dict(size=member.size,type='directory' if member.isdir() else 'file')
            require(member.isdir() or member.isfile(),'Links/devices not permitted')
            if member.isdir(): continue
            require((len(p.parts)==3 and p.parts[1]=='audio' and p.suffix=='.wav')
                    or member.name=='nsynth-test/examples.json','Unexpected release member')
            require(0<member.size<20_000_000,'Member too large/empty'); total+=member.size
            target=root.joinpath(*p.parts); target.parent.mkdir(parents=True,exist_ok=True)
            stream=tar.extractfile(member); require(stream is not None,'Member unavailable')
            with target.open('xb') as f:
                remaining=member.size
                while remaining:
                    block=stream.read(min(1048576,remaining)); require(block,'Truncated member')
                    f.write(block); remaining-=len(block)
    metadata=json.loads((root/'nsynth-test/examples.json').read_text())
    audio=root/'nsynth-test/audio'; require(len(metadata)==4096,'Metadata count')
    require({p.stem for p in audio.iterdir()}==set(metadata),'Audio/metadata join')
    records=[]
    for identity,m in sorted(metadata.items()):
        path=audio/(identity+'.wav')
        with wave.open(str(path),'rb') as w:
            require(w.getnchannels()==1 and w.getframerate()==16000 and w.getsampwidth()==2
                    and w.getnframes()==64000 and w.getcomptype()=='NONE','Unexpected audio shape')
            pcm=w.readframes(64001); require(len(pcm)==128000 and not w.readframes(1),'Decoded duration/EOF')
        require(m['note_str']==identity and m['sample_rate']==16000,'Metadata/sample identity')
        require(m['instrument_source_str'] in ['acoustic','electronic','synthetic'],'Source encoding')
        records.append(dict(id=identity,path=str(path),file_sha256=checksum(path),pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            metadata=m,role='external_measurement_control_only',ai_human_label=None,
            native_sample_rate=16000,native_channels=1,frames=64000,duration_s=4))
    require(len({r['metadata']['note'] for r in records})==4096,'Duplicate note integer ID')
    with (root/'manifest.jsonl').open('x') as f:
        for r in records: f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    receipt=dict(status='accepted_measurement_source_only',created_utc=datetime.now(timezone.utc).isoformat(),
        official_page=PAGE,official_link=OFFICIAL,download_url=DOWNLOAD,google_generation=GENERATION,
        release_bytes=BYTES,release_md5=MD5,release_sha256=checksum(archive),http_headers=headers,
        license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
        attribution='Google Inc.; Engel et al. (2017), Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders, arXiv:1704.01279',
        notes=4096,instruments=len({r['metadata']['instrument_str'] for r in records}),
        source_counts=dict(Counter(r['metadata']['instrument_source_str'] for r in records)),
        instrument_family_counts=dict(Counter(r['metadata']['instrument_family_str'] for r in records)),
        expanded_gzip_bytes=expanded,regular_member_bytes=total,archive_members=members,
        decoded_all=True,gzip_eof_crc_checked=True,ai_human_labels_assigned=False,features_extracted=False,
        limitations='Sample-library-rendered notes, not performed songs. synthetic means instrument production method, not generative-AI authorship. No feature gate or class-accuracy claim.',
        code_sha256=checksum(Path(__file__)),official_page_sha256=checksum(root/'official_dataset_page.html'))
    save(root/'acquisition_receipt.json',receipt)
    products={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=checksum(p))
              for p in sorted(root.rglob('*')) if p.is_file()}
    save(root/'COMMIT.json',dict(status='committed',products=products))
    print(json.dumps({k:receipt[k] for k in ['status','notes','instruments','source_counts','release_sha256']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();acquire(a.output)
