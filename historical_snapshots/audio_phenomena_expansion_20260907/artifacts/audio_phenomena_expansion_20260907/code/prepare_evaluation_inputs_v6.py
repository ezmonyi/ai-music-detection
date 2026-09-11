#!/usr/bin/env python3
"""Prepare matched-cohort SC_L6 inputs without fitting; preserve old tokens."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import prepare_evaluation_inputs_v5 as P5

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/equal60_exploratory_package_v5_v1'
GATE=ROOT/'results/stereo_lowband_gate_v2/gate_result.json'
GATE_AUDIT=ROOT/'audit/stereo_lowband_gate_parent_replay_v2.json'
KEYS=[f'SC_{lo}_{hi}hz_{metric}_median' for lo,hi in [(80,500),(500,2000),(2000,6000)]
      for metric in ['abs_iid_db','side_energy_fraction']]
EXTRA_LEDGER=['sc_input_file_sha256','sc_measurement_sha256','sc_native_channels',
              'sc_extraction_root','sc_gate_sha256']


def sha(p): return P5.sha(p)
def require(condition, message='v6 lineage validation failed'):
    if not condition:
        raise ValueError(message)


def checked_runtime():
    # Older pinned helpers use assertions: never run this boundary with -O.
    require(sys.flags.optimize == 0, 'Optimized Python is not authorized')


def read(p): return json.loads(Path(p).read_text())
def write(path,value):
    with path.open('x') as f:
        f.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')


def rows(path):
    with path.open(newline='') as f:
        reader=csv.DictReader(f); values=list(reader)
        require(values and all(None not in r and None not in r.values() for r in values))
        return reader.fieldnames,values


def reconstruct(extraction,freeze):
    checked_runtime()
    extraction,freeze=Path(extraction),Path(freeze)
    require(extraction.is_absolute() and extraction.resolve()==extraction and not extraction.is_symlink())
    require(freeze.is_absolute() and freeze.resolve()==freeze and not freeze.is_symlink())
    oldproof=P5.validate_package(OLD,False)
    require(sha(OLD/'COMMIT.json')=='7cc03aa840c910df919e531153b61daba864e80794abc91ebf29d3859f70947a')
    gate,audit=read(GATE),read(GATE_AUDIT)
    require(gate['measurement_gate_passed'] and gate['candidate_features']==KEYS)
    require(audit['passed'] and audit['gate_receipt_sha256']==sha(GATE))
    marker_path=extraction/'COMMIT.json'
    require(marker_path.is_file() and marker_path.resolve()==marker_path and not marker_path.is_symlink())
    ecommit=read(extraction/'COMMIT.json'); efreeze=read(freeze); esummary=read(extraction/'summary.json')
    require(ecommit['freeze_sha256']==sha(freeze)==esummary['freeze_sha256'])
    require(sha(freeze)=='06dc798c0551d7f9b49c67e226e2633609b4e434fec06ad16ef28aa08340f979')
    require(efreeze['candidate_features']==KEYS==esummary['candidate_features'])
    require(esummary['rows']==2174 and esummary['quality_did_not_filter_rows'] is True)
    require(efreeze['status']=='frozen_for_measurement_only' and esummary['classifier_fitted'] is False)
    require(set(ecommit['products'])=={str(p.relative_to(extraction)) for p in extraction.rglob('*')
        if p.is_file() and p!=extraction/'COMMIT.json'})
    for name,record in ecommit['products'].items():
        p=extraction/name
        require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.is_relative_to(extraction))
        require(p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],str(p))
    for p,digest in efreeze['bindings'].items(): require(sha(p)==digest,str(p))
    require(esummary['selected_jsonl_sha256']==sha(extraction/'selected_features.jsonl'))
    measured=[json.loads(s) for s in (extraction/'selected_features.jsonl').read_text().splitlines()]
    require(len(measured)==len({r['id'] for r in measured})==2174)
    mcols,metadata=rows(OLD/'metadata_60s.csv'); fcols,features=rows(OLD/'features_60s.csv')
    lcols,ledger=rows(OLD/'origin_ledger.csv')
    require(mcols==P5.META and fcols==['id',*P5.DESCRIPTORS] and lcols==P5.LEDGER)
    native_path=Path(oldproof['contract']['source_spec']['old_package'])/'evidence/native.csv'
    _,native=rows(native_path)
    protected=[r for r in native if r['role'] in ['locked','pilot']]
    require(len(protected)==488)
    protected_ids={r['id'] for r in protected}; protected_groups={r['group_id'] for r in protected}
    native_rows={r['metadata']['id']:r for r in efreeze['selected']}
    require(len(native_rows)==2174)
    wanted=set(native_rows)
    meta=[r for r in metadata if r['id'] in wanted]
    feature=[dict(r) for r in features if r['id'] in wanted]
    origin=[dict(r) for r in ledger if r['id'] in wanted]
    ids=[r['id'] for r in meta]
    require(ids==[r['id'] for r in measured]==[r['id'] for r in feature]==[r['id'] for r in origin])
    require(len(meta)==2174 and Counter(r['label'] for r in meta)=={'0':1278,'1':896})
    require(not set(ids)&protected_ids and not {r['group_id'] for r in meta}&protected_groups)
    require(all(r['source_group']!='human_urmp' for r in meta))
    require({r['id'] for r in metadata}-wanted=={r['metadata']['id'] for r in efreeze['excluded']})
    require(set(ecommit['products'])=={'selected_features.jsonl','summary.json'} |
        {f'items/{identity}/{name}' for identity in ids for name in ['frames.npz','measurement.json']})
    for m,old,source,measurement in zip(meta,feature,origin,measured):
        original=native_rows[m['id']]
        require(original['metadata']==m and original['native_channels']==2)
        require(all(measurement[k]==m[k] for k in ['id','source_group','group_id','label']))
        require(measurement['native_channels']==2)
        require(measurement['input_file_sha256']==original['standardized_file_sha256'])
        require(measurement['input_path']==original['standardized_path'])
        item=extraction/'items'/m['id']/'measurement.json'; proof=read(item)
        require(all(proof[k]==v for k,v in measurement.items()), 'Per-recording provenance differs: '+m['id'])
        require(set(measurement['candidate_features'])==set(KEYS))
        require(measurement['candidate_missing']==[k for k in KEYS if measurement['candidate_features'][k] is None])
        for key in KEYS:
            v=measurement['candidate_features'][key]
            require(v is None or (not isinstance(v,bool) and isinstance(v,(float,int)) and math.isfinite(v)))
            old[key]='' if v is None else format(v,'.17g')
        source.update(sc_input_file_sha256=measurement['input_file_sha256'],
            sc_measurement_sha256=sha(item),sc_native_channels='2',
            sc_extraction_root=str(extraction),sc_gate_sha256=sha(GATE))
    # Verify exact token preservation before writing, not numerical closeness.
    lookup={r['id']:r for r in features}
    require(all(all(r[c]==lookup[r['id']][c] for c in fcols) for r in feature))
    paths=[Path(__file__),Path(P5.__file__),ROOT/'EXPLORATORY_V6_INPUT_PACKAGE_PROTOCOL_EN.md',
        OLD/'COMMIT.json',OLD/'preparation_audit.json',OLD/'metadata_60s.csv',OLD/'features_60s.csv',
        OLD/'origin_ledger.csv',OLD/'family_config.json',native_path,freeze,extraction/'COMMIT.json',
        extraction/'selected_features.jsonl',extraction/'summary.json',GATE,GATE_AUDIT]
    bindings={str(p):sha(p) for p in paths}
    bindings.update(efreeze['bindings'])
    contract=dict(schema_version=6,lineage_validated=True,original_54_values_preserved=True,
        native_stereo_only=True,historical_locked_pilot_ids_and_groups_excluded=True,
        input_files_sha256=bindings,old_package_commit_sha256=sha(OLD/'COMMIT.json'),
        extraction_commit_sha256=sha(extraction/'COMMIT.json'),native_mono_excluded=33,
        extraction_root=str(extraction),extraction_freeze_path=str(freeze),
        historical_protected_id_count=488,rows=2174,
        descriptor_columns=P5.DESCRIPTORS+KEYS,
        external_gate=dict(receipt_path=str(GATE),receipt_sha256=sha(GATE),
            freeze_path=str(ROOT/'preregistration/stereo_lowband_gate_frozen_v2.json'),
            raw_commit_path=str(ROOT/'results/stereo_lowband_controls_v2/COMMIT.json')),
        independent_gate_review=dict(receipt_path=str(GATE_AUDIT),receipt_sha256=sha(GATE_AUDIT)),
        source_counts=dict(Counter(r['source_group'] for r in meta)),
        feature_tokens_sha256=P5.digest(feature),metadata_sha256=P5.digest(meta),origin_ledger_sha256=P5.digest(origin),
        scope='same-cohort exploratory development, not independent source confirmation')
    require(esummary['source_counts']==contract['source_counts'])
    require(esummary['candidate_missing_counts']=={k:sum(r['candidate_features'][k] is None for r in measured) for k in KEYS})
    return meta,feature,origin,contract


def validate_package(package,synthetic=False):
    checked_runtime()
    require(synthetic is False, 'Real lineage validator cannot validate synthetic fixtures')
    package=Path(package)
    require(package.is_absolute() and package.resolve()==package and not package.is_symlink())
    marker_path=package/'COMMIT.json'
    require(marker_path.is_file() and marker_path.resolve()==marker_path and not marker_path.is_symlink())
    proof=read(package/'preparation_audit.json'); marker=read(package/'COMMIT.json')
    products={'metadata_60s.csv','features_60s.csv','origin_ledger.csv','family_config.json','preparation_audit.json'}
    require(marker['status']=='committed' and set(marker['files'])==products)
    require({p.name for p in package.iterdir()}==products|{'COMMIT.json'})
    for n,r in marker['files'].items():
        p=package/n
        require(p.resolve()==p and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'])
    require(proof['status']=='prepared_not_authorized_for_fitting' and proof['schema_version']==6)
    require(proof['synthetic_test_only'] is False and proof['fitting_authorized'] is False and proof['scoring_authorized'] is False)
    contract=proof['contract']
    meta,feature,origin,replayed=reconstruct(contract['extraction_root'],contract['extraction_freeze_path'])
    require(contract==replayed and proof['contract_sha256']==P5.digest(replayed), 'Reconstructed lineage contract differs')
    require(proof['rows']==len(meta)==2174)
    require(rows(package/'metadata_60s.csv')==(P5.META,meta), 'Metadata replay differs')
    require(rows(package/'features_60s.csv')==(['id',*P5.DESCRIPTORS,*KEYS],feature), 'Original/new feature token replay differs')
    require(rows(package/'origin_ledger.csv')==(P5.LEDGER+EXTRA_LEDGER,origin), 'Origin ledger replay differs')
    require(read(package/'family_config.json')==dict(schema_version=6,families={**P5.COLUMNS,'SC':KEYS}))
    require(proof['files_sha256']=={n:sha(package/n) for n in products-{'preparation_audit.json'}})
    return proof


def prepare(extraction,freeze,output):
    checked_runtime()
    require(not output.exists())
    meta,feature,origin,contract=reconstruct(extraction,freeze)
    bindings=contract['input_files_sha256']
    mcols,fcols,lcols=P5.META,['id',*P5.DESCRIPTORS],P5.LEDGER
    output.mkdir()
    for name,records,fields in [('metadata_60s.csv',meta,mcols),('features_60s.csv',feature,fcols+KEYS),
        ('origin_ledger.csv',origin,lcols+EXTRA_LEDGER)]:
        P5.write_csv(output/name,records,fields)
    write(output/'family_config.json',dict(schema_version=6,families={**P5.COLUMNS,'SC':KEYS}))
    products=['metadata_60s.csv','features_60s.csv','origin_ledger.csv','family_config.json']
    proof=dict(schema_version=6,status='prepared_not_authorized_for_fitting',rows=2174,
        synthetic_test_only=False,fitting_authorized=False,scoring_authorized=False,
        contract=contract,contract_sha256=P5.digest(contract),files_sha256={n:sha(output/n) for n in products})
    write(output/'preparation_audit.json',proof)
    for p,digest in bindings.items(): require(sha(p)==digest,str(p))
    # Read back every token and metadata field before publication.
    require(rows(output/'metadata_60s.csv')[1]==meta and rows(output/'features_60s.csv')[1]==feature)
    require(rows(output/'origin_ledger.csv')[1]==origin)
    write(output/'COMMIT.json',dict(status='committed',publication='exclusive directory; COMMIT last',
        files={n:dict(sha256=sha(output/n),bytes=(output/n).stat().st_size)
               for n in products+['preparation_audit.json']}))
    print(json.dumps(dict(status='prepared_not_authorized_for_fitting',rows=2174,
        descriptors=60,commit_sha256=sha(output/'COMMIT.json'),output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--extraction',type=Path,required=True)
    parser.add_argument('--extraction-freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.extraction,args.extraction_freeze,args.output)
