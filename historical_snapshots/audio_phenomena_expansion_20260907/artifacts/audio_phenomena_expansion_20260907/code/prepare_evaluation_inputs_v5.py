#!/usr/bin/env python3
"""Derived2207 exploratory input package only; never authorizes or fits models."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import prepare_evaluation_inputs_v4 as P4
import present_saraga_mureka_frozen_transfer_v1 as U

require,digest,sha,read_json=U.require,U.digest,U.sha,U.read_json
ROOT=Path(__file__).resolve().parent.parent
RD=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
PROTOCOL=ROOT/'EXPLORATORY_V5_INPUT_PACKAGE_PROTOCOL_EN.md'
COLUMNS=P4.COLUMNS
DESCRIPTORS=sum(COLUMNS.values(),[])
META=['id','label','source_group','group_id','role','duration_view','native_sample_rate_hz']
LEDGER=['id','label','source_group','group_id','original_role','original_flags_json','original_metadata_json',
        'origin_package','origin_kind','origin_row_sha256','consumed_external','exploratory_reuse_status']
PRODUCTS={'metadata_60s.csv','features_60s.csv','family_config.json','origin_ledger.csv'}
AUDIT_HASHES={'old':'9b93b558cff42426218abf8ee79ba19833b2a57918319e8855c54aa03bc5a53c',
    'mureka':'47b567d8e126cbcbefa51794b6f2b0af0b462091d719bce1a8fa1fc4ea45da97',
    'saraga':'f28dbfe0de177dfe29263a28f43b09f5c19c9511e424cf2c41ad6e706b826f02'}
COUNTS={'Suno':396,'MTG-Jamendo':481,'human_maestro_v3':300,'human_medleydb':156,
        'human_moisesdb':238,'human_urmp':33,'Mureka_v9':500,'human_saraga_hindustani_v1':103}
NATIVE_RATES={'8000','11025','12000','16000','22050','24000','32000','44100','48000','88200','96000','176400','192000'}


class Bindings(U.Bindings):
    def file(self,path,expected=None):
        # Bound inputs recur across proof layers; independently recheck later.
        key=str(Path(path))
        if key in self.files:
            require(expected is None or self.files[key]==expected,'Conflicting input proof: '+key)
            return self.files[key]
        return super().file(path,expected)
    def recheck(self):
        for path,value in list(self.files.items()): U.Bindings.file(self,path,value)
    def json(self,path,expected=None): self.file(path,expected);return read_json(path)


def rows(path,b,fields=None,key='id'):
    b.file(path)
    with Path(path).open(newline='') as file:
        reader=csv.DictReader(file)
        require(reader.fieldnames and len(set(reader.fieldnames))==len(reader.fieldnames),'Invalid CSV header')
        require(fields is None or reader.fieldnames==fields,'CSV schema changed: '+str(path))
        result=list(reader)
    require(result and all(None not in row and None not in row.values() for row in result),'Malformed/empty CSV')
    ids=[r[key] for r in result]
    require(len(ids)==len(set(ids)) and all(i.strip() for i in ids),'Duplicate/blank identity')
    return result


def bound_by_basename(path,mapping,b):
    """Mirror paths may differ; exactly one original artifact must match its bytes."""
    value=b.file(path)
    matches=[p for p,h in mapping.items() if Path(p).name==Path(path).name and h==value]
    require(len(matches)==1,'Artifact lacks unique accepted binding: '+str(path))
    return matches[0]


def old_source(spec,b):
    package=Path(spec['old_package']);result=Path(spec['old_results'])
    audit=b.json(spec['old_audit'],AUDIT_HASHES['old'])
    require(audit['status']=='passed' and audit['synthetic_test_only'] is False and audit['rows']==1604
            and audit['metrics_independently_recomputed'] is True and audit['model_fitting_performed'] is False,'Unaccepted old numerical audit')
    b.file(Path(__file__).with_name('audit_equal60_v4_results.py').resolve(),audit['auditor_sha256'])
    manifest=b.json(result/'run_manifest.json',audit['result_manifest_sha256'])
    proof=P4.validate_package(package,False)
    b.file(package/'preparation_audit.json',manifest['contract']['preparation_audit_sha256'])
    require(manifest['contract']['package_files_sha256']==proof['files_sha256'],'Old audited package differs')
    for name,value in proof['files_sha256'].items(): b.file(package/name,value)
    metadata=rows(package/'metadata_60s.csv',b)
    feature=rows(package/'features_60s.csv',b,['id',*DESCRIPTORS])
    native={r['id']:r for r in rows(package/'evidence/native.csv',b)}
    require(len(metadata)==1604 and sum(r['role']=='locked' for r in native.values())==438
            and sum(r['role']=='pilot' for r in native.values())==50,'Original historical/pilot exclusion inventory changed')
    rates={r['id']:native[r['id']]['native_sample_rate_hz'] for r in metadata}
    flags={r['id']:{k:v for k,v in native[r['id']].items()
        if k in ('acquisition_role','evaluation_allowed','classifier_admission_authorized')} for r in metadata}
    excluded={r['id']:r['group_id'] for r in native.values() if r['role'] in ('locked','pilot')}
    return dict(kind='old',metadata=metadata,features=feature,rates=rates,flags=flags,
                origin=str(package),excluded_id_groups=excluded)


def external_source(spec,kind,b):
    root=Path(spec[kind+'_results']);old4=Path(spec[kind+'_old4_dir']);fhm=Path(spec[kind+'_fhm_dir'])
    audit=b.json(spec[kind+'_audit'],AUDIT_HASHES[kind]);checked=audit['checked_files_sha256']
    n={'mureka':500,'saraga':103}[kind];U.verify_audit(audit,kind,False,n)
    b.file(Path(__file__).with_name(U.AUDIT_NAMES[kind]).resolve())
    bound_by_basename(Path(__file__).with_name(U.AUDIT_NAMES[kind]).resolve(),checked,b)
    manifest=b.json(root/'publication_manifest.json')
    bound_by_basename(root/'publication_manifest.json',checked,b)
    receipt=b.json(root/'scoring_receipt.json');c=receipt['contract']
    require(receipt['status']=='frozen' and receipt['authorized_stage']==c['authorized_stage']==U.STAGES[kind]
            and receipt['contract_sha256']==digest(c)==audit['contract_sha256'] and c['rows']==n
            and c['synthetic_test_only'] is False and c['refitting'] is False and c['model_selection'] is False
            and c['threshold_tuning'] is False,'External accepted scoring contract changed')
    require(manifest['status']=='scored' and manifest['synthetic_test_only'] is False
            and manifest['contract_sha256']==digest(c) and set(manifest['files'])==U.FILES[kind]
            and manifest['unique_new_ids']==n and manifest['prediction_rows']==3175*n,'External publication scope changed')
    original_root=Path(bound_by_basename(root/'publication_manifest.json',checked,b)).parent
    require(all(checked.get(str(original_root/name))==record['sha256'] for name,record in manifest['files'].items()),
            'Unverified external publication graph')
    require(audit['frozen_receipt_sha256'] in checked.values(),'Original frozen receipt binding missing')
    # Do not reread prediction streams: numerical audits already accepted them.
    for name in ('scoring_receipt.json','identity_roles.csv','model_index.csv'):
        b.file(root/name,manifest['files'][name]['sha256'])
    if kind=='saraga':
        marker=b.json(root/'COMMIT.json',audit['publication_commit_sha256'])
        require(marker['status']=='committed' and marker['files']==dict(manifest['files'],**{
            'publication_manifest.json':{'sha256':b.file(root/'publication_manifest.json'),'bytes':(root/'publication_manifest.json').stat().st_size}}),'Saraga publication COMMIT changed')
    else: require(b.file(root/'publication_manifest.json')==audit['publication_manifest_sha256'],'Mureka publication changed')
    original=c['input_files_sha256']
    bound_by_basename(Path(spec['old_results'])/'run_manifest.json',original,b)
    bound_by_basename(Path(spec['old_audit']),original,b)
    old_csv=old4/'features/expanded_features_60s.csv';fhm_csv=fhm/'features/features.csv'
    for path in (old_csv,fhm_csv):
        original_path=bound_by_basename(path,original,b)
        require(checked.get(original_path)==b.file(path),'Numerical feature CSV not audited')
    strict=b.json(old4/'strict_inference_audit.json');extraction=b.json(old4/'extraction_receipt.json')
    acceptance=b.json(fhm/'acceptance.json')
    for path in (old4/'strict_inference_audit.json',old4/'extraction_receipt.json',old4/'measurement_roles.json',fhm/'acceptance.json',fhm/'launch_contract.json',fhm/'features/contract.json'):
        bound_by_basename(path,original,b)
    require(strict['status']==extraction['status']=='passed' and strict['rows']==extraction['rows']==acceptance['rows']==n
            and strict['classifier_fitted'] is False and strict['scores_generated'] is False
            and strict['all_input_and_stem_samples_decoded'] is True
            and extraction['strict_audit_sha256']==b.file(old4/'strict_inference_audit.json'), 'Incomplete accepted old4 measurement')
    require(acceptance['status']==('passed' if kind=='mureka' else 'passed_measurement_not_classifier_admission')
            and acceptance['classifier_fitted'] is False and acceptance['scores_generated'] is False
            and acceptance['all_descriptor_values_checked'] is True
            and acceptance['launch_contract_sha256']==b.file(fhm/'launch_contract.json'), 'Incomplete accepted native FHM measurement')
    require(any(Path(p).name==old_csv.name and record['sha256']==b.file(old_csv) for p,record in extraction['features'].items())
            and any(Path(p).name=='features.csv' and h==b.file(fhm_csv) for p,h in acceptance['outputs_sha256'].items()),'Measurement-to-CSV binding mismatch')
    identity=rows(root/'identity_roles.csv',b,U.META[kind]);old=rows(old_csv,b,key='item_id');new=rows(fhm_csv,b)
    ids=[r['id'] for r in identity]
    require(len(ids)==n and ids==[r['item_id'] for r in old]==[r['id'] for r in new]
            and digest(identity)==c['identity_sha256'],'Accepted external cohort/order mismatch')
    features=[];flags={};rates={}
    for m,o,f in zip(identity,old,new):
        require(m['label']==o['label']==f['label'] and m['source_group']==o['source_id']==f['source_group']
                and m['group_id']==o['group_id']==f['group_id'] and m['role']==f['role']
                and o['status']=='complete' and f['extraction_status']=='ok','External numerical identity mismatch')
        require(o['native_sample_rate_hz']==f['native_sample_rate_hz'],'External native rate mismatch')
        rates[m['id']]=f['native_sample_rate_hz']
        flags[m['id']]={k:v for k,v in f.items() if k in ('acquisition_role','evaluation_allowed','classifier_admission_authorized')}
        features.append({'id':m['id'],**{col:(o if family in P4.OLD_COLUMNS else f)[col] for family,cols in COLUMNS.items() for col in cols}})
    return dict(kind=kind,metadata=identity,features=features,rates=rates,flags=flags,origin=str(root))


def synthetic_sources(directory,b):
    marker=b.json(directory/'synthetic_fixture.json')
    require(marker=={'synthetic_test_only':True,'purpose':'derived_v5_package_fixture_no_real_admission'},'Explicit synthetic fixture marker required')
    sources=[]
    for kind in ('old','mureka','saraga'):
        metadata=rows(directory/(kind+'_metadata.csv'),b)
        feature=rows(directory/(kind+'_features.csv'),b,['id',*DESCRIPTORS])
        sources.append(dict(kind=kind,metadata=metadata,features=feature,
            rates={r['id']:r['native_sample_rate_hz'] for r in metadata},flags={r['id']:{} for r in metadata},origin=str(directory/kind)))
    return sources


def validate_real_counts(source_counts,label_counts,saraga_component_counts):
    require(source_counts==COUNTS and sum(source_counts.values())==2207,'Exact2207 source composition required')
    require(label_counts=={'0':1311,'1':896},'Exact2207 class composition required')
    require(sorted(saraga_component_counts,reverse=True)==[72,14,10,5,2],'Frozen Saraga components changed')


def combine(sources,synthetic=False):
    require([s['kind'] for s in sources]==['old','mureka','saraga'],'Exact three origins required')
    metadata=[];features=[];ledger=[]
    for source in sources:
        kind=source['kind'];ids=[r['id'] for r in source['metadata']]
        require(ids==[r['id'] for r in source['features']] and len(ids)==len(set(ids)),'Origin duplicate/order mismatch')
        for original,values in zip(source['metadata'],source['features']):
            iid=original['id'];label=original['label'];name=original['source_group'];role=original['role']
            require(list(values)==['id',*DESCRIPTORS],'Exact54 predictor schema required')
            expected_role={'old':'development','mureka':'external_generator_unscored','saraga':'external_human_unscored'}[kind]
            require(role==expected_role,'Original historical/pilot/external role leakage')
            require(label in ('0','1') and original['group_id'].strip(),'Invalid label/group')
            require((kind!='mureka' or (label=='1' and name=='Mureka_v9')) and
                    (kind!='saraga' or (label=='0' and name=='human_saraga_hindustani_v1')) and
                    (kind!='old' or name not in ('Mureka_v9','human_saraga_hindustani_v1')),'Source/class/origin leakage')
            require(label==('1' if name in ('Suno','Mureka_v9') else '0'),'Source label changed')
            require(not synthetic or iid.startswith('synthetic_v5_'),'Synthetic fixture contains real identity')
            require(synthetic or not iid.startswith('synthetic_'),'Real package contains synthetic identity')
            rate=source['rates'][iid];require(rate in NATIVE_RATES,'Invalid preserved native rate')
            require(original.get('duration_view','60s')=='60s','Non60s context refused')
            flags={k:v for k,v in original.items() if k in ('acquisition_role','evaluation_allowed','classifier_admission_authorized')}
            for key,value in source['flags'][iid].items():
                require(key not in flags or flags[key]==value,'Original metadata/measurement flags disagree');flags[key]=value
            if kind!='old':
                require(flags.get('classifier_admission_authorized','False')=='False' and flags.get('evaluation_allowed','False')=='False',
                        'Previously admitted external source flags changed')
            for col in DESCRIPTORS:
                token=values[col].strip()
                require(token.lower() in ('','nan','na','null') or math.isfinite(float(token)),'Infinity/non-numeric predictor refused')
            metadata.append(dict(id=iid,label=label,source_group=name,group_id=original['group_id'],role='development',duration_view='60s',native_sample_rate_hz=rate))
            features.append(dict(values))  # preserve exact token including all missing encodings
            ledger.append(dict(id=iid,label=label,source_group=name,group_id=original['group_id'],original_role=role,
                original_flags_json=json.dumps(flags,sort_keys=True,separators=(',',':')),
                original_metadata_json=json.dumps(original,sort_keys=True,separators=(',',':')),
                origin_package=source['origin'],origin_kind=kind,origin_row_sha256=digest({'metadata':original,'features':values,'flags':flags,'native_sample_rate_hz':rate}),
                consumed_external=str(kind!='old'),exploratory_reuse_status='consumed_external_for_exploratory_development' if kind!='old' else 'continued_original_development'))
    ids=[r['id'] for r in metadata]
    require(len(ids)==len(set(ids)),'Cross-origin duplicate identity')
    excluded=sources[0].get('excluded_id_groups',{})
    require(not set(ids).intersection(excluded)
            and not {r['group_id'] for r in metadata}.intersection(excluded.values()),
            'Combined cohort overlaps original locked/pilot IDs or groups')
    associations={}
    for row in metadata:
        key=row['group_id'];identity=(row['label'],row['source_group'])
        require(key not in associations or associations[key]==identity,'Global group crosses labels or sources; resolve before packaging')
        associations[key]=identity
    if synthetic:require(0<len(ids)<=16,'Synthetic16-row bound')
    else:
        validate_real_counts(Counter(r['source_group'] for r in metadata),Counter(r['label'] for r in metadata),
            list(Counter(r['group_id'] for r in metadata if r['source_group']=='human_saraga_hindustani_v1').values()))
    # Stable global ID sort after strict per-origin order checks, shared by all products.
    order=sorted(range(len(ids)),key=lambda i:ids[i])
    return tuple([[records[i] for i in order] for records in (metadata,features,ledger)])


def assemble(spec,synthetic=False):
    b=Bindings()
    if synthetic:
        require(set(spec)=={'synthetic_fixture'},'Synthetic fixture may not add real inputs')
        sources=synthetic_sources(Path(spec['synthetic_fixture']),b)
    else:
        require('synthetic_fixture' not in spec,'Synthetic bypass refused')
        require(set(spec)==set(defaults(ROOT)),'Exact accepted real source paths required')
        sources=[old_source(spec,b),external_source(spec,'mureka',b),external_source(spec,'saraga',b)]
    meta,features,ledger=combine(sources,synthetic)
    for p in (Path(__file__).resolve(),Path(P4.__file__).resolve(),Path(U.__file__).resolve(),
              Path(__file__).with_name('verify_extract_equal60.py').resolve(),PROTOCOL):b.file(p)
    contract=dict(schema_version=5,stage='derived_exploratory_package_only',synthetic_test_only=synthetic,rows=len(meta),
        source_spec=spec,input_files_sha256=dict(sorted(b.files.items())),metadata_sha256=digest(meta),features_sha256=digest(features),origin_ledger_sha256=digest(ledger),
        source_counts=dict(sorted(Counter(r['source_group'] for r in meta).items())),label_counts=dict(sorted(Counter(r['label'] for r in meta).items())),
        descriptor_columns=DESCRIPTORS,families=COLUMNS,family_config_sha256=digest(P4.build_config()),
        consumed_external_sources=['Mureka_v9','human_saraga_hindustani_v1'],historical_locked_and_pilot_included=False,
        fitting_authorized=False,threshold_selection_authorized=False,interpretation='exploratory development after external-source consumption; not prospective independent confirmation',
        admission_scope='accepted compact numerical/provenance proofs only; no raw media re-admission')
    b.recheck()
    return contract,meta,features,ledger,b


def write_csv(path,records,fields):
    with path.open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(records)


def validate_package(path,synthetic=False):
    root=Path(path);b=Bindings();marker=b.json(root/'COMMIT.json');proof=b.json(root/'preparation_audit.json')
    require(marker['status']=='committed' and marker['publication']=='exclusive hardlinks, COMMIT last'
            and set(marker['files'])==PRODUCTS|{'preparation_audit.json'}
            and {p.name for p in root.iterdir()}==PRODUCTS|{'preparation_audit.json','COMMIT.json'},'Incomplete/orphan package')
    for name,record in marker['files'].items():
        b.file(root/name,record['sha256']);require((root/name).stat().st_size==record['bytes'],'Package size changed')
    require(proof['schema_version']==5 and proof['status']=='prepared_not_authorized_for_fitting'
            and proof['synthetic_test_only'] is synthetic and proof['scoring_authorized'] is False
            and proof['fitting_authorized'] is False and set(proof['files_sha256'])==PRODUCTS,
            'Invalid/nonprospective preparation proof')
    require(proof['files_sha256']=={n:r['sha256'] for n,r in marker['files'].items() if n!='preparation_audit.json'},'Proof/COMMIT product mismatch')
    contract,meta,features,ledger,inputs=assemble(proof['contract']['source_spec'],synthetic)
    require(proof['contract']==contract and proof['contract_sha256']==digest(contract) and proof['rows']==len(meta),'Current package contract differs')
    require(rows(root/'metadata_60s.csv',b,META)==meta and rows(root/'features_60s.csv',b,['id',*DESCRIPTORS])==features
            and rows(root/'origin_ledger.csv',b,LEDGER)==ledger,'Derived values/roles/order differ from accepted origins')
    require(b.json(root/'family_config.json')==P4.build_config(),'Frozen54-descriptor seven-family configuration changed')
    inputs.recheck();b.recheck()
    return proof


def prepare(spec,output,synthetic=False):
    output=Path(output);require(not output.exists() and not output.is_symlink(),'Existing output refused')
    contract,meta,features,ledger,b=assemble(spec,synthetic)
    output.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix=output.name+'.tmp.',dir=output.parent))
    try:
        write_csv(temp/'metadata_60s.csv',meta,META);write_csv(temp/'features_60s.csv',features,['id',*DESCRIPTORS]);write_csv(temp/'origin_ledger.csv',ledger,LEDGER)
        U.write_json(temp/'family_config.json',P4.build_config())
        proof=dict(schema_version=5,status='prepared_not_authorized_for_fitting',synthetic_test_only=synthetic,rows=len(meta),
            scoring_authorized=False,fitting_authorized=False,contract=contract,contract_sha256=digest(contract),files_sha256={n:sha(temp/n) for n in sorted(PRODUCTS)})
        U.write_json(temp/'preparation_audit.json',proof);b.recheck();U.publish(temp,output)
    finally:shutil.rmtree(temp)
    return validate_package(output,synthetic)


def defaults(root):
    spec={'old_package':str(root/'results/equal60_package_v4_with_recovery_v1'),'old_results':str(root/'results/equal60_results_v4_with_recovery_v1'),
          'old_audit':str(root/'audit/equal60_v4_with_recovery_results_audit_v1.json')}
    for kind,tag in [('mureka','mureka60'),('saraga','saraga103')]:
        spec.update({kind+'_results':str(root/f'results/{tag}_frozen_v4_transfer_results_v1'),
            kind+'_audit':str(root/f'audit/{tag}_frozen_v4_transfer_numerical_audit_v1.json'),
            kind+'_fhm_dir':str(root/('results/measurements_mureka60_fhm_v2' if kind=='mureka' else 'results/measurements_saraga103_fhm_v1')),
            kind+'_old4_dir':str(RD/('mureka60_inputs_v2/measurements' if kind=='mureka' else 'saraga_external103_old4_v1/measurements'))})
    return spec


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=lambda p:Path(p).absolute(),default=ROOT)
    parser.add_argument('--output-dir',type=lambda p:Path(p).absolute(),required=True)
    parser.add_argument('--synthetic-test-only',action='store_true');parser.add_argument('--synthetic-fixture',type=lambda p:Path(p).absolute())
    for name in defaults(ROOT):parser.add_argument('--'+name.replace('_','-'),type=lambda p:Path(p).absolute())
    a=parser.parse_args(argv)
    if a.synthetic_test_only:
        require(a.synthetic_fixture is not None,'Explicit synthetic fixture required')
        require(all(getattr(a,n) is None for n in defaults(ROOT)),'Synthetic/real inputs mixed')
        spec={'synthetic_fixture':str(a.synthetic_fixture)}
    else:
        require(a.synthetic_fixture is None,'Synthetic fixture refused in real mode')
        spec={k:str(getattr(a,k) or v) for k,v in defaults(a.root).items()}
    proof=prepare(spec,a.output_dir,a.synthetic_test_only)
    print(json.dumps({k:proof[k] for k in ('status','rows','contract_sha256','fitting_authorized')}))


if __name__=='__main__':main()
