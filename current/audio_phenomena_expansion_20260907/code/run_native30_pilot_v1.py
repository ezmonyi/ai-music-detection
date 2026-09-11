"""Eight-source CPU preprocessing pilot; never admit a cohort or fit a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf
import standardize_native30_v1 as dsp

REPORT_SHA='a3dcf87f703b7709b01d963e1a1c2e96f9a223d159180d1b793021337a3b160a'
DSP_SHA='2e05a4d0d2e7789888aa41a53bece46e8de97fad34ef34f21354d165e77b55be'
PREFIX='Native30-DSP-pilot-20260907|'
SOURCES={'ACE-Step','HeartMuLa','Suno','Udio','FMA','MTG-Jamendo','human_medleydb','human_moisesdb'}


def write_json(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()


def selection(report):
    candidates=report['candidates']
    dsp.require(len(candidates)==1746 and len({r['id'] for r in candidates})==1746,'candidate identity/count')
    groups={}
    for r in candidates:
        if r['native_stereo_metadata_eligible']:
            dsp.require(r['native_evidence']['channels']==2,'native stereo flag mismatch')
            key=hashlib.sha256((PREFIX+r['id']).encode()).hexdigest()
            if r['source_group'] not in groups or key<groups[r['source_group']][0]:
                groups[r['source_group']]=(key,r)
    dsp.require(set(groups)==SOURCES,'pilot source set changed')
    return [{'selection_hash':key,**r} for source,(key,r) in sorted(groups.items())]


def run(report_path,copy_dir,output):
    report_path,copy_dir,output=map(lambda p:Path(p).absolute(),(report_path,copy_dir,output))
    dsp.require(dsp.digest(report_path)==REPORT_SHA,'trusted source report SHA mismatch')
    code=Path(__file__).absolute();module=code.with_name('standardize_native30_v1.py')
    dsp.require(dsp.digest(module)==DSP_SHA,'reviewed DSP changed')
    bound={str(p):dsp.digest(p) for p in [report_path,code,module,code.with_name('test_standardize_native30_v1.py')]}
    rows=selection(json.loads(report_path.read_text()))
    dsp.require(not output.exists() and output.parent.is_dir(),'new output directory required')
    for r in rows:
        ident=r['id'];dsp.require(re.fullmatch(r'[A-Za-z0-9_.-]+',ident) is not None,'unsafe item id')
        native=r['native_evidence'];p=Path(native['path'])
        # Only the two explicitly local native originals receive execution copies.
        if p.parts[:2]==('/','Users'):
            dsp.require(r['source_group'] in {'FMA','Suno'},'unexpected local-origin mapping')
            p=copy_dir/(ident+p.suffix)
        dsp.require(p.is_file() and dsp.digest(p)==native['sha256'],'native execution copy/source differs')
        r['execution_native_path']=str(p)
    output.mkdir();(output/'audio').mkdir();(output/'items').mkdir()
    write_json(output/'contract.json',{
        'status':'frozen_before_pilot_preprocessing','scope':'eight-source native30 DSP pilot only',
        'selection_rule':PREFIX+' hash minimum per eligible source', 'rows':rows,
        'configuration':dsp.CONFIG,'bindings':bound,'approved_region':None,
        'excludes_interval_only_sources':['Mureka','Saraga'],
        'classifier_admitted':False,'feature_extraction_authorized':False})
    receipts=[];failures=[]
    for r in rows:
        try:
            native=r['native_evidence']
            y,a=dsp.standardize(r['execution_native_path'],native['sha256'],native['sample_rate_hz'],native['channels'])
            wav=output/'audio'/(r['id']+'.wav');tmp=wav.with_name('.'+wav.name+'.tmp')
            with tmp.open('xb') as f:
                sf.write(f,y,dsp.RATE,format='WAV',subtype='FLOAT');f.flush();os.fsync(f.fileno())
            back,rate=sf.read(tmp,dtype='float32',always_2d=True)
            dsp.require(rate==dsp.RATE and back.shape==y.shape and back.tobytes()==y.tobytes(),'WAV float32 byte roundtrip')
            os.link(tmp,wav);tmp.unlink()
            receipt={'id':r['id'],'source_group':r['source_group'],'label':r['label'],
                     'component_id':r['component_id'],'source_receipt':r['receipt_binding'],
                     'source_receipt_row':r['receipt_row'],'original_native_path':native['path'],
                     'standardized_path':str(wav),'file_sha256':dsp.digest(wav),
                     'file_bytes':wav.stat().st_size,'waveform_bit_exact_roundtrip':True,'audit':a}
            write_json(output/'items'/(r['id']+'.json'),receipt);receipts.append(receipt)
            print(json.dumps({'completed':len(receipts),'id':r['id'],'actual_native_frames':a['sequential_decode']['actual_frames']}),flush=True)
        except Exception as exc:
            failure={'id':r['id'],'exception':type(exc).__name__,'message':str(exc)}
            failures.append(failure);write_json(output/'items'/(r['id']+'.failure.json'),failure)
            print(json.dumps({'failure':failure}),flush=True)
    dsp.require(all(dsp.digest(p)==sha for p,sha in bound.items()),'pilot bound inputs/code changed')
    summary={'status':'completed_DSP_pilot' if not failures else 'failed_DSP_pilot',
             'planned':8,'completed':len(receipts),'failures':failures,
             'records':receipts,'classifier_fits':0,'cohort_admitted':False,
             'limitations':['not full3869 cohort','not acoustic-stereo or decoder-agreement certification','not AI/Human feature performance']}
    write_json(output/'summary.json',summary)
    if failures:
        raise RuntimeError('pilot failures retained; no COMMIT and no padding/replacement')
    products={str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':dsp.digest(p)}
              for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'COMMIT.json',{'status':'committed_DSP_pilot_only','products':products})
    return {'status':summary['status'],'completed':len(receipts),'commit_sha256':dsp.digest(output/'COMMIT.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['report','copy-dir','output']:p.add_argument('--'+arg,required=True)
    a=p.parse_args();print(json.dumps(run(a.report,a.copy_dir,a.output)))
