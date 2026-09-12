"""Reconcile selected AIME generated recordings without admitting human licenses."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(source, historical, out):
    rows=[json.loads(line) for line in source.open()]
    selected=[r for r in rows if r['label']==1]
    assert len(rows)==5500 and len(selected)==5000
    assert len({r['id'] for r in selected})==5000
    inputs={r['id']:r for r in csv.DictReader(historical.open())}
    counts=collections.Counter(r['model'] for r in selected)
    assert len(counts)==10 and set(counts.values())=={500}
    fields=['id','model','hf_dataset','hf_main_revision','hf_parquet_revision',
            'hf_original_id','source_parquet','source_row_number','raw_sha256',
            'original_sample_rate','original_channels','original_duration_s',
            'standardized_sha256','standardized_relpath','crop_start_s',
            'historical_role','historical_group_id']
    output=[]
    for r in selected:
        h=inputs[r['id']]
        assert h['label']=='1' and h['raw_sha256']==r['raw_sha256']
        record={k:r[k] for k in fields if k in r}
        record.update(historical_role=h['role'],historical_group_id=h['group_id'])
        output.append(record)
    out.mkdir(exist_ok=False)
    with (out/'selected_generated.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(output)
    (out/'README.md').write_text('''# Selected AIME generated-audio catalogue

5,000 selected generated recordings: 500 each from AudioLDM 2 Large, AudioLDM 2
Music, MusicGen Large/Medium/Small, Mustango, Riffusion, Stable Audio v1/v2 and
Udio. Exact IDs, original-byte hashes and AI labels were reconciled with the
historical 10s input manifest. The 500 MTG-labelled human records are excluded
from this publication plan because their separate per-track licenses must be
resolved. Other AIME records not selected for this experiment are not added.

This is a provenance catalogue, not evidence of a new raw-byte audit or completed
upload. Parquet revision/shard/row and original ID identify the preserved source.
Raw and standardized hashes refer to different files. Current historical roles
and group IDs are retained; the original external-test role is not silently
treated as a new untouched evaluation after later development reuse.

The AIME publisher's README explicitly licenses its generated audio CC BY 4.0:
https://huggingface.co/datasets/disco-eth/AIME/blob/main/README.md . This is the
publisher's declaration, not independent verification of each service's rights.
Description tags have separate CC BY-NC-SA 4.0 terms and are omitted here.
MTG per-track licenses are not replaced by the generated-audio license.
Retain AIME attribution and its paper, Benchmarking Music Generation Models and
Metrics via Human Preference Studies:
https://ieeexplore.ieee.org/abstract/document/10887745 .
''')
    (out/'COMMIT.json').write_text(json.dumps(dict(status='selected_provenance_join_not_audio_upload',
        rows=5000,model_counts=dict(counts),source_manifest_sha256=sha(source),
        historical_metadata_sha256=sha(historical),
        products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()},
        original_bytes_freshly_verified=False,audio_uploaded=False),indent=2))
    print(json.dumps(dict(rows=len(output),model_counts=dict(counts))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.source,a.historical,a.out)
