"""Append current source metadata without changing frozen experimental labels."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path


def main(audit, metadata, out):
    rows=json.loads(audit.read_text())
    selected=[r for r in rows if r['source_collection']=='suno_unknown']
    source={r['id']:r for r in csv.DictReader(metadata.open())}
    assert len(selected)==250
    output=[]
    for r in selected:
        uuid=Path(r['source_relative_path']).stem
        m=source[uuid]
        assert m['file_name']=='audio/'+uuid+'.mp3'
        output.append(dict(id=r['id'],source_uuid=uuid,historical_model_name=r['model_name'],
            recovered_model_name=m['model_name'],recovered_major_version=m['major_model_version'],
            original_sha256=r['actual_original_sha256'],source_relative_path=m['file_name'],
            source_dataset='Kukedlc/suno-ai-music-dataset',
            metadata_revision='bdff424e70c10ef62dca13ba43659ad9e7e1fcdb'))
    assert all(r['historical_model_name']=='unknown' for r in output)
    out.mkdir(exist_ok=False)
    with (out/'model_provenance.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(output[0]),lineterminator='\n');w.writeheader();w.writerows(output)
    (out/'README.md').write_text('''# Recovered source model metadata for 250 historical Suno IDs

All 250 historical suno_unknown source UUIDs match the pinned public metadata.
The source model_name field contains six labels; the original experiment stored
unknown. Both values are preserved side by side. This is a post-experiment
provenance supplement, not a retroactive relabelling, refit, new model holdout,
or independent new evaluation. No claim that card-level V5.5 prose describes
every row is made. Historical Suno source/group membership remains unchanged.

Source: https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset
Metadata revision: bdff424e70c10ef62dca13ba43659ad9e7e1fcdb.
This exporter checks UUID identity; remote audio byte checks are separate.
Prompt text, lyrics, images and source signed URLs are not included here.
''')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'COMMIT.json').write_text(json.dumps(dict(rows=250,
        recovered_model_counts=dict(Counter(r['recovered_model_name'] for r in output)),
        inputs=dict(audit_sha256=sha(audit),metadata_sha256=sha(metadata)),
        products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()},
        historical_experiments_changed=False,audio_uploaded=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('audit','metadata','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();main(a.audit,a.metadata,a.out)
