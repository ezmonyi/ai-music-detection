"""Join historical measurements to input roles and verified original uploads."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

WORK=Path('/Users/yi/Documents/code/music')
REPORT=Path('/Users/yi/Documents/report/music_ai_detection_20260912')
OUT=WORK/'deliverables/github_staging_20260912/datasets/historical_measurement_coverage_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    inputs={};published={}
    def bound(p):inputs[str(p)]=sha(p);return read(p)
    fma_path=REPORT/'datasets/fma_original_publication_plan_v1.json'
    assert sha(fma_path)=='2aae3374be02122ce65874a64bb5bb8996403965a23613c22d01c02f15350ccc'
    fma=bound(fma_path);receipt=bound(REPORT/'current_results/fma_originals_publication_v1.json')
    assert receipt['all_remote_hashes_verified'] and receipt['audio_files']==392
    for row in fma['rows']:
        if row['status']=='explicit_version_original_candidate':
            published[row['id']]=(receipt['prefix']+'audio/'+row['id']+'.mp3',receipt['revision'])
    mroot=WORK/'deliverables/hf_maestro_originals_v1'
    mc=bound(mroot/'COMMIT.json');mr=bound(mroot/'manifest.json')
    assert sha(mroot/'manifest.json')==mc['manifest_sha256'] and len(mr)==300
    evidence={}
    for index in range(30):
        batch=bound(mroot/f'batch_{index:03d}.json');assert batch['sha256_verified']
        for path in batch['files']:assert path not in evidence;evidence[path]=batch['revision']
    assert set(evidence)=={r['path'] for r in mr}
    for r in mr:published[r['id']]=(r['path'],evidence[r['path']])
    assert len(published)==692
    OUT.mkdir(exist_ok=False);counts=Counter();ids=set();published_ids=set()
    fields=['id','view','source_group','role','feature_status','s16_feature_status','s8_feature_status',
            'd_feature_status','r_feature_status','p_feature_status','published_original_path','published_original_revision']
    for view in ['10s','30s']:
        folder=REPORT/'current_results/historical_four_family_inventory_v1'/('four_family_'+view)
        summary=bound(folder/'merge_summary.json')
        fp=folder/Path(summary['feature_csv_path']).name
        mp=REPORT/'datasets/historical_manifest_sources_v1'/('metadata_'+view+'.csv')
        assert sha(fp)==summary['feature_csv_sha256'] and sha(mp)==summary['metadata_sha256']
        inputs[str(fp)]=sha(fp);inputs[str(mp)]=sha(mp)
        metadata={r['id']:r for r in csv.DictReader(mp.open())}
        features=list(csv.DictReader(fp.open()));assert len(features)==len(metadata)==summary['rows']
        assert {r['item_id'] for r in features}==set(metadata)
        with (OUT/('measurement_status_'+view+'.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for feature in features:
                uid=feature['item_id'];meta=metadata[uid];ids.add(uid)
                pub=published.get(uid,('',''))
                if pub[0]:published_ids.add(uid)
                row={k:feature[k] for k in fields if k.endswith('feature_status')}
                row.update(id=uid,view=view,source_group=meta['source_group'],role=meta['role'],
                           published_original_path=pub[0],published_original_revision=pub[1])
                writer.writerow(row)
                counts[(view,meta['source_group'],feature['feature_status'],bool(pub[0]))]+=1
    with (OUT/'source_status_counts.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['view','source_group','feature_status','has_published_original','view_rows'])
        writer.writerows([*key,value] for key,value in sorted(counts.items()))
    notice='''# Historical measurement coverage

The historical feature CSVs and their bound input metadata were hash-verified
before joining by exact ID. Complete/serialization-failure and per-family
availability states are retained. This is measurement coverage, NOT proof that
every row entered a fitted classifier. Input roles are preserved without
admitting provisional/pilot/stress material into the current classifier.

Published-original links are based on verified FMA and MAESTRO publication
receipts. A link to an original does NOT mean the 10s/30s view or separated stems
were uploaded. FMA originals mean unchanged FMA medium excerpts, not full songs.
Blank links mean no publication was established by this join, not proof that no
other release exists. No license or public redistribution permission is inferred.
The inventory covers the historical 26-source population, not all external
phenomenon controls or later YuE2/Mureka/Saraga additions.
'''
    with (OUT/'README.md').open('x') as f:f.write(notice)
    with (OUT/'COMMIT.json').open('x') as f:
        json.dump(dict(status='verified_historical_measurement_join',distinct_ids=len(ids),
            ids_with_verified_original_publication=len(published_ids),
            source_bindings={Path(p).name+'#'+v[:12]:v for p,v in inputs.items()},
            products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.name!='COMMIT.json'},
            whole_dataset_complete=False),f,indent=2)
    print(json.dumps(dict(distinct_ids=len(ids),original_publication_matched=len(published_ids))))


if __name__=='__main__':main()
