#!/usr/bin/env python3
"""Presentation only: exactly matched audited AI and Human transfer summaries.

No classifier calls, model loading, raw-media admission, fitting, threshold
selection or pooled two-class endpoints. Writes only a new presentation folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile

INDEX=['combination','quantity','fold_index','fold_uid','model_sha256','candidate_key','train_id_set_sha256','test_id_set_sha256']
COMBINATIONS=['+'.join(c) for n in range(1,8) for c in itertools.combinations('SDRPFHM',n)]
CAPS=['25','50','100','200','all']
BASELINE='S+D+R+P'
CONTRASTS=[BASELINE+'+F',BASELINE+'+H',BASELINE+'+M',BASELINE+'+F+H+M']
OVERVIEW=list('SDRPFHM')+[BASELINE,*CONTRASTS]
METRICS=['mureka_ai_sensitivity','mureka_false_negative_rate','saraga_specificity','saraga_false_positive_rate',
         'saraga_equal_component_specificity','saraga_equal_component_false_positive_rate']
STAGES={'mureka':'mureka500_frozen_v4_transfer_scoring_only','saraga':'saraga103_frozen_v4_transfer_scoring_only'}
AUDIT_NAMES={'mureka':'audit_mureka60_transfer_v1.py','saraga':'audit_saraga103_transfer_v1.py'}
META={'mureka':['id','label','source_group','group_id','role','acquisition_role'],
      'saraga':['id','label','source_group','group_id','role','evaluation_allowed','classifier_admission_authorized']}
SUMMARY={'mureka':INDEX+['synthetic_test_only','rows','unique_ids','unique_groups','tp','fn','threshold','ai_sensitivity','false_negative_rate'],
         'saraga':INDEX+['synthetic_test_only','rows','unique_ids','unique_groups','fp','tn','threshold','false_positive_rate','specificity',
                         'equal_component_false_positive_rate','equal_component_specificity']}
COMPONENT=INDEX+['synthetic_test_only','group_id','rows','fp','tn','false_positive_rate','specificity']
FILES={'mureka':{'identity_roles.csv','model_index.csv','per_model_sensitivity.csv','predictions.csv','scoring_receipt.json'},
       'saraga':{'identity_roles.csv','model_index.csv','per_model_specificity.csv','per_component_specificity.csv','predictions.csv',
                 'scoring_receipt.json','all635_cells.csv','predefined_overview.csv','output_accounting.json'}}
PROTOCOL=Path(__file__).resolve().parent.parent/'SARAGA_MUREKA_TRANSFER_PRESENTATION_PROTOCOL_EN.md'
MUREKA_ACCEPTED_AUDIT='47b567d8e126cbcbefa51794b6f2b0af0b462091d719bce1a8fa1fc4ea45da97'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'Duplicate JSON key'); value[key]=item
        return value
    def nonfinite(token): raise ValueError('Nonfinite JSON: '+token)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=nonfinite)


def read_csv(path,fields):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f); require(reader.fieldnames==fields,'CSV schema changed: '+str(path)); rows=list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows),'Empty/malformed CSV')
    return rows


class Bindings:
    def __init__(self): self.files={}
    def file(self,path,expected=None):
        path=Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve()==path,'Noncanonical regular input: '+str(path))
        value=sha(path)
        require(expected is None or value==expected,'Input hash mismatch: '+str(path))
        require(str(path) not in self.files or self.files[str(path)]==value,'Input changed: '+str(path))
        self.files[str(path)]=value; return value
    def recheck(self):
        for path,value in list(self.files.items()): self.file(path,value)


def record_files(root,records,b):
    require(records,'Empty publication')
    for name,record in records.items():
        require(Path(name).name==name and name not in ('.','..') and set(record)=={'sha256','bytes'}
                and type(record['bytes']) is int and record['bytes']>=0,'Unsafe publication record')
        b.file(root/name,record['sha256']); require((root/name).stat().st_size==record['bytes'],'Publication size changed')


def verify_audit(audit,kind,synthetic,n):
    require(audit.get('status')=='passed' and audit.get('synthetic_test_only') is synthetic
            and audit.get('model_fitting_performed') is False and audit.get('rows')==n
            and audit.get('model_instances')==3175 and audit.get('prediction_rows')==3175*n,
            'Passed matching numerical audit required')
    if kind=='mureka':
        require(audit.get('all_ids_in_every_model') is True
                and audit.get('all_decisions_match_published_and_independent_scores') is True
                and audit.get('sensitivity_summary_rows')==3175
                and audit.get('new_source_transform_learning_performed') is False,'Incomplete Mureka numerical audit')
    else:
        require(all(audit.get(k) is True for k in ('all_decisions_exact','all_models_all_ids_checked','all_integer_counts_exact',
                'equal_component_and_per_recording_metrics_separately_checked','all635_mean_min_max_checked'))
                and audit.get('model_summary_rows')==3175 and audit.get('aggregate_cells')==635
                and audit.get('model_selection_performed') is False and audit.get('threshold_tuning_performed') is False,
                'Incomplete Saraga numerical audit')


def index_key(row): return tuple(row[k] for k in ('combination','quantity','fold_index'))


def validate_index(index):
    require(len(index)==3175 and len({index_key(r) for r in index})==3175
            and {index_key(r) for r in index}==set(itertools.product(COMBINATIONS,CAPS,map(str,range(5)))), 'Exact127x5x5 index required')
    folds={}
    for row in index:
        require(all(row[k].strip() for k in INDEX) and all(re.fullmatch('[0-9a-f]{64}',row[k]) for k in ('model_sha256','train_id_set_sha256','test_id_set_sha256')), 'Invalid index identity/hash')
        key=(row['quantity'],row['fold_index']); identity=tuple(row[k] for k in ('fold_uid','train_id_set_sha256','test_id_set_sha256'))
        require(key not in folds or folds[key]==identity,'Training/test/fold identity varies across combinations')
        folds[key]=identity
    require(len({v[0] for v in folds.values()})==25,'Fold UIDs reused across cap/fold')


def integer(value):
    require(isinstance(value,str) and re.fullmatch(r'0|[1-9][0-9]*',value),'Nonnegative integer required')
    return int(value)


def rate(value,expected=None):
    value=float(value)
    require(math.isfinite(value) and 0<=value<=1,'Invalid rate')
    require(expected is None or abs(value-expected)<=1e-15,'Rate/count mismatch')
    return value


def load_source(root,audit_path,audit_sha,kind,b,synthetic=False):
    b.file(audit_path,audit_sha)
    if not synthetic and kind=='mureka': require(audit_sha==MUREKA_ACCEPTED_AUDIT,'Only accepted Mureka audit permitted')
    audit=read_json(audit_path)
    b.file(root/'publication_manifest.json')
    manifest=read_json(root/'publication_manifest.json'); receipt=read_json(root/'scoring_receipt.json'); c=receipt['contract']
    n=c['rows']
    require(type(n) is int and ((synthetic and 0<n<=16) or (not synthetic and n=={'mureka':500,'saraga':103}[kind])), 'Real/synthetic cohort count boundary')
    verify_audit(audit,kind,synthetic,n)
    require(receipt['status']=='frozen' and receipt['authorized_stage']==c['authorized_stage']==STAGES[kind]
            and receipt['contract_sha256']==digest(c)==audit['contract_sha256']
            and c['synthetic_test_only'] is synthetic and c['threshold']==.5 and c['model_instances']==3175
            and c['refitting'] is False and c['model_selection'] is False and c['threshold_tuning'] is False,
            'Scoring freeze/scope changed')
    require(manifest['status']=='scored' and manifest['synthetic_test_only'] is synthetic
            and manifest['contract_sha256']==digest(c) and manifest['classifier_fitted'] is False
            and manifest['original_v4_development_admission'] is False and manifest['unique_new_ids']==n
            and manifest['model_instances']==3175 and manifest['prediction_rows']==3175*n,'Publication counts/contract changed')
    require(set(manifest['files'])==FILES[kind],'Scored publication file inventory changed')
    record_files(root,manifest['files'],b)
    checked=audit['checked_files_sha256']
    # The scorer reserializes the frozen JSON when publishing scoring_receipt;
    # its byte hash can differ from the original frozen receipt. The passed
    # numerical audit binds both files and checks their parsed equality.
    require(audit['frozen_receipt_sha256'] in checked.values(),'Numerical audit lacks its original frozen receipt binding')
    manifests=[p for p,h in checked.items() if Path(p).name=='publication_manifest.json' and h==b.files[str(root/'publication_manifest.json')]]
    require(len(manifests)==1,'Audit does not bind unique current publication manifest')
    audited_root=Path(manifests[0]).parent
    for name,record in manifest['files'].items():
        require(checked.get(str(audited_root/name))==record['sha256'],'Numerical audit did not check published file: '+name)
    if kind=='saraga':
        b.file(root/'COMMIT.json',audit['publication_commit_sha256']); marker=read_json(root/'COMMIT.json')
        require(marker['status']=='committed' and marker['publication']=='exclusive hardlinks, COMMIT last'
                and marker['files']==dict(manifest['files'],**{'publication_manifest.json':{'sha256':b.files[str(root/'publication_manifest.json')],
                    'bytes':(root/'publication_manifest.json').stat().st_size}}),'Saraga COMMIT mismatch')
        require(set(p.name for p in root.iterdir())==FILES[kind]|{'publication_manifest.json','COMMIT.json'},'Extra/orphan Saraga publication files')
    else:
        require(audit['publication_manifest_sha256']==b.files[str(root/'publication_manifest.json')]
                and set(p.name for p in root.iterdir())==FILES[kind]|{'publication_manifest.json'},'Mureka publication mismatch')
    code=Path(__file__).with_name(AUDIT_NAMES[kind]).resolve()
    code_hash=b.file(code)
    require(any(Path(p).name==code.name and h==code_hash for p,h in checked.items()),'Numerical auditor code differs from passed audit')
    index=read_csv(root/'model_index.csv',INDEX); validate_index(index)
    require(digest(index)==c['model_index_sha256'],'Frozen model index digest changed')
    meta=read_csv(root/'identity_roles.csv',META[kind]); require(len(meta)==n and len({r['id'] for r in meta})==n,'Cohort identity count mismatch')
    require(digest(meta)==c['identity_sha256'],'Frozen cohort identity digest changed')
    expected={'mureka':('1','Mureka_v9','external_generator_unscored'),
              'saraga':('0','human_saraga_hindustani_v1','external_human_unscored')}[kind]
    require(all((r['label'],r['source_group'],r['role'])==expected for r in meta),'Source/class/measurement role changed')
    require(all(r['id'].startswith('synthetic_'+kind+'_v4_') for r in meta) if synthetic
            else all(not r['id'].startswith('synthetic_') for r in meta),'Synthetic/real identity mixing')
    groups=dict(sorted(Counter(r['group_id'] for r in meta).items()))
    if kind=='saraga':
        require(groups==c['component_sizes']==audit['component_sizes'] and (synthetic or sorted(groups.values(),reverse=True)==[72,14,10,5,2]),'Frozen Human component assignments changed')
        require(all(r['evaluation_allowed']==r['classifier_admission_authorized']=='False' for r in meta),'Human measurement admission flag changed')
    else: require(all(r['acquisition_role']=='reserved_unscored' for r in meta),'Mureka acquisition role changed')
    rows=read_csv(root/('per_model_sensitivity.csv' if kind=='mureka' else 'per_model_specificity.csv'),SUMMARY[kind])
    require(len(rows)==3175,'Missing model summaries')
    by_key={}
    for identity,row in zip(index,rows):
        require(all(row[k]==identity[k] for k in INDEX) and row['synthetic_test_only']==str(synthetic)
                and row['threshold']=='0.5' and integer(row['rows'])==integer(row['unique_ids'])==n
                and integer(row['unique_groups'])==len(groups),'Model summary identity/denominator changed')
        first,second=('tp','fn') if kind=='mureka' else ('fp','tn')
        f,t=integer(row[first]),integer(row[second]); require(f+t==n,'Integer counts do not sum to denominator')
        rate(row['ai_sensitivity' if kind=='mureka' else 'false_positive_rate'],f/n)
        rate(row['false_negative_rate' if kind=='mureka' else 'specificity'],t/n)
        by_key[index_key(row)]=row
    components=[]
    if kind=='saraga':
        components=read_csv(root/'per_component_specificity.csv',COMPONENT)
        require(len(components)==3175*len(groups)==audit['component_summary_rows'],'Missing component summaries')
        cursor=0
        for identity,row in zip(index,rows):
            fpr=[]; fp_total=0
            for group,count in groups.items():
                item=components[cursor];cursor+=1
                require(all(item[k]==identity[k] for k in INDEX) and item['group_id']==group and integer(item['rows'])==count
                        and item['synthetic_test_only']==str(synthetic),'Component identity/denominator changed')
                fp,tn=integer(item['fp']),integer(item['tn']);require(fp+tn==count,'Component integer count mismatch')
                fpr.append(rate(item['false_positive_rate'],fp/count));rate(item['specificity'],tn/count);fp_total+=fp
            require(fp_total==int(row['fp']),'Component/recording count disagreement')
            rate(row['equal_component_false_positive_rate'],statistics.mean(fpr))
            rate(row['equal_component_specificity'],statistics.mean(1-x for x in fpr))
    return dict(index=index,rows=by_key,components=components,groups=groups,count=n,contract_sha256=digest(c),audit_sha256=audit_sha,
                original_frozen_receipt_sha256=audit['frozen_receipt_sha256'],
                published_scoring_receipt_sha256=b.files[str(root/'scoring_receipt.json')])


def matching_rows(mureka,saraga):
    require(mureka['index']==saraga['index'],'Sources do not share exact cap/fold/model/candidate/train/test identities')
    output=[]
    for identity in mureka['index']:
        key=index_key(identity); ai,human=mureka['rows'][key],saraga['rows'][key]
        row=dict(identity,mureka_rows=mureka['count'],saraga_rows=saraga['count'],
                 mureka_tp=int(ai['tp']),mureka_fn=int(ai['fn']),saraga_fp=int(human['fp']),saraga_tn=int(human['tn']))
        row.update(dict(zip(METRICS,[float(ai['ai_sensitivity']),float(ai['false_negative_rate']),float(human['specificity']),
            float(human['false_positive_rate']),float(human['equal_component_specificity']),float(human['equal_component_false_positive_rate'])])))
        output.append(row)
    return output


def aggregate(rows,group_fields,metrics):
    grouped=defaultdict(list)
    for row in rows: grouped[tuple(row[k] for k in group_fields)].append(row)
    output=[]
    for key,part in grouped.items():
        require(len(part)==5,'Every presentation cell must contain five matched models')
        value=dict(zip(group_fields,key));value['stored_fold_models']=5
        for metric in metrics:
            values=[float(r[metric]) for r in part]
            value.update({metric+'_mean':statistics.mean(values),metric+'_min':min(values),metric+'_max':max(values)})
        output.append(value)
    return output


def build_tables(mureka,saraga):
    matched=matching_rows(mureka,saraga)
    cells=aggregate(matched,['combination','quantity'],METRICS)
    order={c:i for i,c in enumerate(COMBINATIONS)}
    cells.sort(key=lambda r:(order[r['combination']],CAPS.index(r['quantity'])))
    for row in cells: row.update(mureka_rows_per_model=mureka['count'],saraga_rows_per_model=saraga['count'])
    overview=[r for combo in OVERVIEW for r in cells if r['combination']==combo]
    by_key={index_key(r):r for r in matched};deltas=[]
    for candidate in CONTRASTS:
        for cap in CAPS:
            for fold in map(str,range(5)):
                base,new=by_key[(BASELINE,cap,fold)],by_key[(candidate,cap,fold)]
                require(all(base[k]==new[k] for k in ('fold_uid','train_id_set_sha256','test_id_set_sha256')),'Contrast folds/training/test rows differ')
                row=dict(baseline=BASELINE,candidate=candidate,quantity=cap,fold_index=fold,
                    **{k:base[k] for k in ('fold_uid','train_id_set_sha256','test_id_set_sha256')},
                    baseline_model_sha256=base['model_sha256'],candidate_model_sha256=new['model_sha256'],
                    baseline_candidate_key=base['candidate_key'],candidate_candidate_key=new['candidate_key'])
                row.update({metric+'_delta_pp':100*(new[metric]-base[metric]) for metric in METRICS});deltas.append(row)
    delta_cells=aggregate(deltas,['baseline','candidate','quantity'],[m+'_delta_pp' for m in METRICS])
    component_cells=aggregate(saraga['components'],['combination','quantity','group_id'],['false_positive_rate','specificity'])
    for row in component_cells: row['component_rows_per_model']=saraga['groups'][row['group_id']]
    require(len(cells)==635 and len(overview)==60 and len(deltas)==100 and len(delta_cells)==20,'Incomplete prospective presentation grid')
    return {'matched_per_model.csv':matched,'all635_side_by_side.csv':cells,'fixed60_overview.csv':overview,
            'matched100_deltas_pp.csv':deltas,'fixed20_delta_cells_pp.csv':delta_cells,
            'saraga_component_cells.csv':component_cells}


def write_json(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def markdown(tables,mureka,saraga,synthetic):
    lines=['# Matched external transfer presentation','',
        '**Synthetic test fixture only; not real transfer results.**' if synthetic else 'Audited Mureka AI sensitivity and Saraga Human specificity on exactly matching original saved models.',
        '',f'Mureka: {mureka["count"]} AI recordings per model. Saraga: {saraga["count"]} Human recordings per model.',
        'Human connected component sizes: '+', '.join(map(str,sorted(saraga['groups'].values(),reverse=True)))+'.',
        'Five-model minima and maxima are descriptive ranges, not confidence intervals. Repeated model applications use the same recordings.',
        'Equal-component Human rates weight each connected component equally; primary Human rates weight recordings equally.',
        'The two sources are displayed separately. Source/class confounding and historical Saraga exposure remain limitations.',
        '', '| Families | Cap | AI sensitivity % | Human specificity % | Equal-component specificity % |',
        '|---|---|---|---|---|']
    for row in tables['fixed60_overview.csv']:
        def show(metric): return f'{100*row[metric+"_mean"]:.2f} [{100*row[metric+"_min"]:.2f}, {100*row[metric+"_max"]:.2f}]'
        lines.append(f'| {row["combination"]} | {row["quantity"]} | {show("mureka_ai_sensitivity")} | {show("saraga_specificity")} | {show("saraga_equal_component_specificity")} |')
    lines += ['', 'All635 cells and all component cells are supplied as CSV. FNR/FPR complements are included explicitly in CSV.',
        'Every delta is candidate minus S/D/R/P, in percentage points, paired within the original cap/fold. Positive sensitivity/specificity deltas mean higher correct-class rates; positive FNR/FPR deltas mean higher error rates.',
        '', '| Increment | Cap | AI sensitivity delta pp | Human specificity delta pp | Equal-component specificity delta pp |',
        '|---|---|---|---|---|']
    for row in tables['fixed20_delta_cells_pp.csv']:
        def show(metric): return f'{row[metric+"_delta_pp_mean"]:+.2f} [{row[metric+"_delta_pp_min"]:+.2f}, {row[metric+"_delta_pp_max"]:+.2f}]'
        lines.append(f'| {row["candidate"].removeprefix(BASELINE+"+")} | {row["quantity"]} | {show("mureka_ai_sensitivity")} | {show("saraga_specificity")} | {show("saraga_equal_component_specificity")} |')
    return '\n'.join(lines)+'\n'


def publish(temporary,destination):
    destination.mkdir()
    for path in sorted(temporary.iterdir()):
        with path.open('rb') as file: os.fsync(file.fileno())
        os.link(path,destination/path.name)
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(destination.iterdir())}
    write_json(temporary/'COMMIT.json',dict(status='committed',files=files,publication='exclusive hardlinks, COMMIT last'))
    with (temporary/'COMMIT.json').open('rb') as file: os.fsync(file.fileno())
    os.link(temporary/'COMMIT.json',destination/'COMMIT.json')
    record_files(destination,files,Bindings())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mureka-results','saraga-results','mureka-audit','saraga-audit','output-dir'):
        parser.add_argument('--'+name,type=lambda p:Path(p).absolute(),required=True)
    for name in ('mureka-audit-sha256','saraga-audit-sha256'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--synthetic-test-only',action='store_true');parser.add_argument('--synthetic-fixture-root',type=lambda p:Path(p).absolute())
    args=parser.parse_args(argv);b=Bindings()
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(),'Existing output refused')
    if args.synthetic_test_only:
        require(args.synthetic_fixture_root is not None,'Explicit synthetic fixture required')
        marker=args.synthetic_fixture_root/'presentation_fixture.json';b.file(marker)
        require(read_json(marker)=={'synthetic_test_only':True,'purpose':'handwritten_presentation_fixture_no_inference_or_scores'},'Invalid synthetic fixture marker')
        require(args.mureka_results.parent==args.saraga_results.parent==args.synthetic_fixture_root,'Synthetic source location mismatch')
    else: require(args.synthetic_fixture_root is None,'Synthetic fixture refused in real mode')
    sources={kind:load_source(getattr(args,kind+'_results'),getattr(args,kind+'_audit'),getattr(args,kind+'_audit_sha256'),kind,b,args.synthetic_test_only) for kind in ('mureka','saraga')}
    tables=build_tables(sources['mureka'],sources['saraga'])
    b.file(Path(__file__).resolve());b.file(PROTOCOL)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=args.output_dir.name+'.tmp.',dir=args.output_dir.parent))
    try:
        for name,records in tables.items():
            with (temporary/name).open('x',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        with (temporary/'MATCHED_TRANSFER_TABLES_EN.md').open('x') as file:
            file.write(markdown(tables,sources['mureka'],sources['saraga'],args.synthetic_test_only))
        b.recheck()
        write_json(temporary/'presentation_receipt.json',dict(status='presentation_only',synthetic_test_only=args.synthetic_test_only,
            classifier_fitted=False,classifier_scores_generated=False,raw_media_admission_repeated=False,
            input_files_sha256=dict(sorted(b.files.items())),row_counts={k:len(v) for k,v in tables.items()},
            source_contract_sha256={k:v['contract_sha256'] for k,v in sources.items()},
            source_audit_sha256={k:v['audit_sha256'] for k,v in sources.items()},
            source_original_frozen_receipt_sha256={k:v['original_frozen_receipt_sha256'] for k,v in sources.items()},
            source_published_scoring_receipt_sha256={k:v['published_scoring_receipt_sha256'] for k,v in sources.items()},
            exact_model_identity_match=True,source_pooling_performed=False,threshold=.5,
            delta_definition='100 * (candidate rate - S/D/R/P rate), within original cap/fold',
            ranges='min/max of five stored models or five matched differences; not confidence intervals',
            saraga_component_sizes=sources['saraga']['groups'],
            forbidden_endpoints=['balanced_accuracy','roc_auc','source_transfer_J','rank','winner']))
        publish(temporary,args.output_dir)
    finally: shutil.rmtree(temporary)


if __name__=='__main__': main()
