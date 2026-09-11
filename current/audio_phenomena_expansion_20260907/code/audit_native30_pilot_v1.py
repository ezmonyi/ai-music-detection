"""Independent eight-source pilot replay; no producer imports or audio writes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def audit(root,report):
    root,report=Path(root),Path(report)
    bindings={str(p):sha(p) for p in [Path(__file__),root/'COMMIT.json',report]}
    require(bindings[str(root/'COMMIT.json')]=='a008a59ec808aaf09a33e1147e72250ec4a35240c8c5a087ce2b6f622cad5498','trusted pilot COMMIT')
    require(bindings[str(report)]=='a3dcf87f703b7709b01d963e1a1c2e96f9a223d159180d1b793021337a3b160a','trusted source report')
    commit=json.loads((root/'COMMIT.json').read_text())
    require(commit['status']=='committed_DSP_pilot_only','commit scope')
    for rel,r in commit['products'].items():
        p=root/rel;require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'product binding:'+rel)
    report_rows=json.loads(report.read_text())['candidates']
    chosen={}
    for source in sorted({r['source_group'] for r in report_rows if r['native_stereo_metadata_eligible']}):
        group=[r for r in report_rows if r['source_group']==source and r['native_stereo_metadata_eligible']]
        r=min(group,key=lambda x:hashlib.sha256(('Native30-DSP-pilot-20260907|'+x['id']).encode()).hexdigest())
        chosen[r['id']]=r
    summary=json.loads((root/'summary.json').read_text());contract=json.loads((root/'contract.json').read_text())
    require(len(chosen)==8 and len(summary['records'])==8 and not summary['failures'],'complete eight-source pilot')
    require({r['id'] for r in summary['records']}==set(chosen)=={r['id'] for r in contract['rows']},'selection identity')
    for p,digest in contract['bindings'].items():require(sha(p)==digest,'bound producer input/code changed')
    results=[]
    for row in summary['records']:
        ident=row['id'];origin=chosen[ident]['native_evidence'];p=Path(row['audit']['source_path'])
        require(sha(p)==origin['sha256'],'native source digest')
        # Small pilot inputs only: concatenate an independently sized block stream.
        parts=[]
        with sf.SoundFile(p) as f:
            rate,ch=f.samplerate,f.channels;header_frames=f.frames
            require(rate==origin['sample_rate_hz'] and ch==origin['channels']==2,'native metadata')
            while True:
                x=f.read(32771,dtype='float64',always_2d=True)
                require(np.isfinite(x).all(),'nonfinite source')
                if len(x)==0:break
                parts.append(x)
        full=np.concatenate(parts,axis=0);n=len(full);count=30*rate;start=(n-count)//2
        require(start>=0,'short source')
        selected=full[start:start+count];expected=selected
        if rate!=44100:
            d=math.gcd(rate,44100);expected=scipy.signal.resample_poly(selected,44100//d,rate//d,axis=0,window=('kaiser',5.0),padtype='constant')
        expected=np.asarray(expected,dtype='<f4',order='C')
        a=row['audit'];c=a['coordinates']
        require(c=={'region_start_frame':0,'region_frames':n,'crop_start_frame':start,'crop_frames':count,'crop_end_frame_exclusive':start+count},'native crop coordinates')
        require(a['sequential_decode']['actual_frames']==n and a['header_minus_actual_frames']==header_frames-n,'decode counts')
        require(hashlib.sha256(np.asarray(full,dtype='<f8',order='C').tobytes()).hexdigest()==a['sequential_decode']['float64_pcm_sha256'],'whole decoded hash')
        require(hashlib.sha256(np.asarray(selected,dtype='<f8',order='C').tobytes()).hexdigest()==a['native_crop_float64_sha256'],'crop hash')
        wav=Path(row['standardized_path']);info=sf.info(wav);actual,sr=sf.read(wav,dtype='float32',always_2d=True)
        require((info.frames,sr,info.channels,info.format,info.subtype)==(1323000,44100,2,'WAV','FLOAT'),'output format')
        require(actual.tobytes()==expected.tobytes(),'independent reference float32 samples differ')
        require(sha(wav)==row['file_sha256'] and sha(p)==origin['sha256'],'end source/output binding')
        require(hashlib.sha256(actual.tobytes()).hexdigest()==a['output_waveform_float32_sha256'],'output PCM hash')
        require(row['source_group']==chosen[ident]['source_group'] and row['label']==chosen[ident]['label'] and row['component_id']==chosen[ident]['component_id'],'source/label/component lineage')
        results.append({'id':ident,'source_group':row['source_group'],'actual_native_frames':n,'native_rate_hz':rate,'crop_start_native_frame':start,'output_frames':1323000,'float32_bit_exact':True,'header_minus_actual_frames':header_frames-n})
    require(all(sha(p)==d for p,d in bindings.items()),'audit inputs changed')
    return {'status':'passed_independent_pilot_DSP_replay','bindings':bindings,'records':results,'product_count':len(commit['products']),'output_samples_checked':8*1323000*2,'shared_libraries':['libsndfile','NumPy','SciPy resample_poly'],'producer_imported':False,'cohort_admitted':False,'classifier_fits':0,'scope':'eight selected pilot recordings only; no independent decoder claim; not all3869'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();print(json.dumps(audit(a.root,a.report),indent=2))
