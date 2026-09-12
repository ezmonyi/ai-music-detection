"""Preserve early input membership; compare locators without claiming content identity."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tarfile

SPECS=[('frequency_ablation_20260813','manifest.jsonl'),
       ('direct_band_500x500_20260815','manifest.jsonl'),
       ('stem_spectral_pilot_20260813','selected_manifest.jsonl')]


def run(archives,out):
    records=[];bindings={}
    for experiment,filename in SPECS:
        archive=archives/(experiment+'.tar.gz')
        member=f'artifacts/{experiment}/{filename}'
        with tarfile.open(archive,'r:gz') as stream:
            raw=stream.extractfile(member).read()
        bindings[experiment]=dict(archive=archive.name,member=member,
            member_sha256=hashlib.sha256(raw).hexdigest())
        rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len({r['id'] for r in rows})==len(rows)
        records.extend(dict(experiment=experiment,legacy_id=r['id'],label=r['label'],
            source=r['source'],relative_source_path=r['path'],
            duration_s=r['duration'],sample_rate_hz=r['sample_rate'],
            crop_start_s=r.get('crop_start'),historical_split=r.get('split',''),
            original_audio_hash=None,original_audio_publication_verified=False) for r in rows)
    reference={r['relative_source_path'] for r in records if r['experiment']=='direct_band_500x500_20260815'}
    summary=[]
    for experiment,_ in SPECS:
        subset=[r for r in records if r['experiment']==experiment]
        summary.append(dict(experiment=experiment,rows=len(subset),
            exact_paths_in_direct_band=sum(r['relative_source_path'] in reference for r in subset),
            source_counts=dict(Counter(r['source'] for r in subset))))
    assert [r['rows'] for r in summary]==[400,1000,20]
    out.mkdir(exist_ok=False)
    with (out/'inputs.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    (out/'SUMMARY.json').write_text(json.dumps(dict(experiments=summary,source_bindings=bindings,
        limitation='Relative path and legacy ID comparison is not content deduplication or raw-byte verification.',
        whole_project_complete=False),indent=2))
    (out/'COMMIT.json').write_text(json.dumps(dict(rows=len(records),products={p.name:dict(
        bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir()},
        status='early_input_membership_preserved_not_audio_verified'),indent=2))
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.archives,args.out)
