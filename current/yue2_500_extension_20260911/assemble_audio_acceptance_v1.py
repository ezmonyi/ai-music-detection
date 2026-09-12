"""Collect completed independent upload receipts without claiming whole-corpus coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SPECS={
 'hf_aime_originals_v1':(5000,100,'audio/aime_generated_originals_v1/'),
 'hf_open_model_audio_v1':(2000,100,'audio/project_open_models_v1/'),
 'hf_suno_originals_v1':(500,5,'audio/historical_suno_originals_v1/'),
 'hf_mtt_clips_v1':(500,1,'audio/magnatagatune_selected_v1/'),
 'hf_early_fma_originals_v1':(162,1,'audio/early_fma_originals_v1/'),
 'hf_early_suno_originals_v1':(100,1,'audio/early_suno_originals_v1/'),
}


def main(source,out):
    summary=[];copies=[]
    for name,(count,batches,prefix) in SPECS.items():
        folder=source/name;accept=json.loads((folder/'INDEPENDENT_ACCEPTANCE.json').read_text())
        terminal=json.loads((folder/'COMMIT.json').read_text())
        assert accept['final_acceptance'] is True
        assert accept.get('checked_files',accept.get('verified_audio_files'))==count
        paths=sorted(folder.glob('batch_*.json'));assert len(paths)==batches
        files=[]
        for i,path in enumerate(paths):
            assert path.name==f'batch_{i:03d}.json'
            r=json.loads(path.read_text());assert r['sha256_verified'] is True
            files.extend(r['files'])
        assert len(files)==len(set(files))
        audio=[p for p in files if Path(p).suffix in ('.mp3','.wav','.flac')]
        assert len(audio)==count and all(p.startswith(prefix) for p in audio)
        summary.append(dict(publication=name,audio_file_entries=count,batches=batches,
            revision=accept['revision'],prefix=prefix,terminal_status=terminal['status'],
            independent_acceptance=True))
        copies.extend((p,Path(name)/p.name) for p in paths+[folder/'COMMIT.json',folder/'INDEPENDENT_ACCEPTANCE.json'])
    out.mkdir(exist_ok=False)
    products={}
    for p,relative in copies:
        q=out/relative;q.parent.mkdir(exist_ok=True);shutil.copyfile(p,q)
        products[str(relative)]=dict(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())
    (out/'SUMMARY.json').write_text(json.dumps(dict(publications=summary,audio_file_entries=8262,
        receipt_products=products,scope='Six completed queued publications; not all project sources or distinct songs.',
        whole_project_complete=False),indent=2))
    assert sum(r['audio_file_entries'] for r in summary)==8262
    print('Collected six acceptances and 208 batch receipts for 8262 audio file entries')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.source,a.out)
