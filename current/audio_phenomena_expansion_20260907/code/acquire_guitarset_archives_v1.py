"""Download only official GuitarSet mic/annotation archives; not decode acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

EXPECTED={
    'audio_mono-mic.zip':(656927981,'275966d6610ac34999b58426beb119c3'),
    'annotation.zip':(39132574,'b39b78e63d3446f2e54ddb7a54df9b10'),
}


def digest(path,algorithm='sha256'):
    with path.open('rb') as stream: return hashlib.file_digest(stream,algorithm).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',required=True,type=Path)
    parser.add_argument('--metadata-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if digest(args.metadata)!=args.metadata_sha256: raise ValueError('metadata SHA mismatch')
    metadata=json.loads(args.metadata.read_text())
    if (metadata['doi']!='10.5281/zenodo.3371780' or
        metadata['metadata']['license']['id']!='cc-by-4.0' or
        metadata['metadata']['access_right']!='open' or
        metadata['metadata']['version']!='1.1.0'):
        raise ValueError('unexpected upstream provenance/license')
    entries={f['key']:f for f in metadata['files']}
    for name,(size,md5) in EXPECTED.items():
        f=entries[name]
        if f['size']!=size or f['checksum']!='md5:'+md5 or f['links']['self']!=f'https://zenodo.org/api/records/3371780/files/{name}/content':
            raise ValueError('unexpected archive identity')
    root=args.output.resolve()
    root.mkdir(exist_ok=False)
    shutil.copyfile(args.metadata,root/'upstream_metadata.json')
    acquired=[]
    try:
        for name,(size,md5) in EXPECTED.items():
            partial=root/(name+'.part')
            command=['curl','--fail','--location','--silent','--show-error','--retry','3',
                '--retry-delay','10','--connect-timeout','20','--max-time','1800',
                '--dump-header',str(root/(name+'.headers.txt')),
                '--output',str(partial),entries[name]['links']['self']]
            start=time.monotonic()
            run=subprocess.run(command,capture_output=True,text=True,timeout=7400)
            (root/(name+'.command.json')).write_text(json.dumps({'command':command,'returncode':run.returncode,
                'stdout':run.stdout,'stderr':run.stderr,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
            if run.returncode: raise RuntimeError('archive download failed: '+name)
            if partial.stat().st_size!=size or digest(partial,'md5')!=md5:
                raise ValueError('official archive size/MD5 mismatch: '+name)
            partial.rename(root/name)
            acquired.append({'name':name,'bytes':size,'md5':md5,'sha256':digest(root/name)})
            print(json.dumps({'event':'archive_verified','archive':acquired[-1]}),flush=True)
        if digest(args.metadata)!=args.metadata_sha256: raise ValueError('metadata changed')
        receipt={'status':'archives_verified_not_decoded','metadata_sha256':args.metadata_sha256,
            'doi':metadata['doi'],'license':'CC-BY-4.0','archives':acquired,
            'code_sha256':digest(Path(__file__)),'zip_crc_checked':False,'audio_decoded':False,
            'bc_measured':False,'classifier_fits':0,'external_gate_passed':False}
        (root/'acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        products={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in root.iterdir()}
        (root/'COMMIT.json').write_text(json.dumps({'status':'committed','kind':'archive_acquisition_only',
            'products':products},indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        (root/'EXECUTION_FAILURE.json').write_text(json.dumps({'error_type':type(exc).__name__,
            'error':str(exc),'verified_archives':acquired,'commit_published':False},indent=2)+'\n')
        raise


if __name__=='__main__': main()
