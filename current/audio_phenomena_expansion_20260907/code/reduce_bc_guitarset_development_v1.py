"""Apply the reviewed scalar reducer to accepted development metadata only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from bicoherence_scalar_v1 import reduce_metadata, compare_conditions

COMMIT_SHA='cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c'
AUDIT_SHA='61a926a92c764d6852608f9829395382e6593e56544d1434ec0f28af509ffc93'
REDUCER_SHA='b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85'
CONDITIONS=['baseline','common_gain','polarity','closed_minus6db','independent_minus6db','closed_0db','independent_0db']
CROP={'start_sample':0,'stop_sample_exclusive':128000,'source_resampled_samples':128000}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def mean(values):return math.fsum(values)/len(values) if values else None


def run(root,audit_path,output):
    root,audit_path,output=map(Path,(root,audit_path,output))
    require(not output.exists() and output.parent.is_dir(),'exclusive existing-parent output required')
    require(sha(root/'COMMIT.json')==COMMIT_SHA and sha(audit_path)==AUDIT_SHA,'accepted source/audit binding')
    commit=json.loads((root/'COMMIT.json').read_text());audit=json.loads(audit_path.read_text())
    require(audit['passed'] is True and audit['result_commit_sha256']==COMMIT_SHA,'actual accepted replay')
    here=Path(__file__);reducer=here.with_name('bicoherence_scalar_v1.py')
    require(sha(reducer)==REDUCER_SHA,'reviewed scalar reducer binding')
    bindings={str(p):sha(p) for p in [here,reducer,here.with_name('test_bicoherence_scalar_v1.py'),root/'COMMIT.json',audit_path]}
    def saved(rel):
        path=root/rel;raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        require(digest==commit['products'][rel]['sha256'] and len(raw)==commit['products'][rel]['bytes'],'bound raw metadata:'+rel)
        bindings[str(path)]=digest
        return json.loads(raw)
    summary=saved('summary.json');rows=summary['per_recording']
    require(len(rows)==90 and len({r['item_id'] for r in rows})==90,'development count')
    records=[]
    for row in rows:
        require(row['split_role']=='development','reserved/unused data forbidden')
        ident=row['item_id'];conditions={c:saved(ident+'/'+c+'.json') for c in CONDITIONS}
        reduced={c:reduce_metadata(m,crop=CROP) for c,m in conditions.items()}
        comparisons={}
        for level in ['minus6db','0db']:
            comp=compare_conditions(conditions['closed_'+level],conditions['independent_'+level],first_crop=CROP,second_crop=CROP)
            reference=next(r for r in summary['levels'][level]['per_recording'] if r['item_id']==ident)
            require(comp['paired_pool_count']==reference['covered_pools'],'paired coverage differs from accepted summary')
            require(math.isclose(comp['paired_pool_mean_difference'],reference['difference'],rel_tol=2e-12,abs_tol=2e-14),'paired difference differs from accepted summary')
            comparisons[level]={k:v for k,v in comp.items() if k not in ['first','second']}
        records.append({'id':ident,'score_id':row['score_id'],'player_id':row['player_id'],
                        'performance':row['performance'],'conditions':reduced,'comparisons':comparisons})
    levels={}
    for level in ['minus6db','0db']:
        grouped=defaultdict(list);differences=[];disagreements=[]
        for r in records:
            c=r['comparisons'][level];value=c['operational_median_difference']
            if value is not None:grouped[r['score_id']].append(value);differences.append(value)
            if value is not None and c['paired_pool_mean_difference'] is not None:
                disagreements.append(abs(value-c['paired_pool_mean_difference']))
        levels[level]={'equal_score_operational_difference':mean([mean(v) for v in grouped.values()]),
                       'covered_scores':len(grouped),'score_denominator':15,'available_recordings':len(differences),
                       'recording_denominator':90,'positive_recordings':sum(x>0 for x in differences),
                       'negative_recordings':sum(x<0 for x in differences),'zero_recordings':sum(x==0 for x in differences),
                       'maximum_operational_vs_paired_difference':max(disagreements) if disagreements else None}
    require(all(sha(path)==digest for path,digest in bindings.items()),'reducer inputs/code changed')
    result={'status':'development_scalar_reduction_complete_not_admission','metadata_conditions_read':630,
            'levels':levels,'records':records,'bindings':bindings,'classifier_fits':0,
            'reserved_data_read':False,'external_gate_passed':False,
            'crop_coordinate_scope':'each physical8s condition WAV is the standardized source; original recording crop provenance remains in accepted producer records',
            'interpretation':'operational endpoint verification on existing exposed development metadata; not independent sensitivity replication or AI/Human utility'}
    temp=output.with_name('.'+output.name+'.tmp')
    with temp.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,output)
    finally:temp.unlink()
    return {'status':result['status'],'levels':levels,'output_sha256':sha(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','audit','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.audit,a.output),indent=2))
