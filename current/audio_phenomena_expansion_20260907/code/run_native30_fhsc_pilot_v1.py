"""Eight accepted pilot WAVs: frozen F/H/SC integration, no classifier.

The archived stereo FLOAT WAV is unchanged. F/H use their established separate
mono/DC-subtracted16k analysis view; SC uses the stereo44.1k view. M is retained
only as an expected short-context missingness diagnostic, never a predictor.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

PINS={'standardize_native30_v1.py':'2e05a4d0d2e7789888aa41a53bece46e8de97fad34ef34f21354d165e77b55be',
      'audio_inputs.py':'3c411df6a94ab74e61dd27293e7fcc35f919ff55494cb55dfe18dafa34a095d2',
      'phase_features.py':'f1598ecb513ae67431571e586e436840f80114ac2b6d04fce3a46e8eeb0a5ac5',
      'musical_features.py':'72d54ceb5266e2ff2f0d1ce5bdbcfa7e8c93611784a03e40d3255672020f8083',
      'stereo_candidate_v1.py':'132bd5fe225258b72b913238bd456e3b18279b6b5a664b2191b0292aede9481c',
      'materialize_native30_new1695_v1.py':'165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a'}
PILOT_SHA='a008a59ec808aaf09a33e1147e72250ec4a35240c8c5a087ce2b6f622cad5498'
AUDIT_SHA='957c3936eb375c351e7e885a6af4cef76d4c49e058f1daffc4e6cf09c5a9aebb'
SC_KEYS=[f'SC_{lo}_{hi}hz_{name}_median' for lo,hi in [(80,500),(500,2000),(2000,6000)]
         for name in ['abs_iid_db','side_energy_fraction']]


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float):
        if math.isinf(value):raise ValueError('infinite measurement')
        if math.isnan(value):return None
    return value


def selected_sc(sc):
    values={k:sc['features'][k] for k in SC_KEYS}
    missing=[k for k,v in values.items() if not np.isfinite(v)]
    return {'features':values,'selected_six_status':'complete' if not missing else ('missing' if len(missing)==6 else 'partial'),
            'selected_six_finite_count':6-len(missing),'selected_six_missing':missing,
            'full_grid_diagnostic_status':sc['status'],'frame_count':sc['frame_count'],'bands':sc['bands']}


def measure(path):
    import audio_inputs as inputs
    import phase_features as phase
    import musical_features as musical
    import stereo_candidate_v1 as stereo
    y,loading=inputs.load_exact({'audio_path':str(path),'audio_offset_s':0},duration=30,target_sr=16000)
    f=phase.extract_phase_features(y,16000); hm=musical.extract_musical_features(y,16000)
    x,sr=sf.read(path,dtype='float64',always_2d=True)
    if sr!=44100 or x.shape!=(1323000,2) or not np.isfinite(x).all():raise ValueError('exact30 stereo input required')
    sc=stereo.extract_stereo_candidate(x,sr,native_channels=2)
    if hm['M_status'] not in {'missing_short_duration','missing_low_energy'}:raise ValueError('M must not become a short-context feature')
    return clean({'F':f,'H':{k:v for k,v in hm.items() if k.startswith('H_')},
                  'M_diagnostic_not_predictor':{k:v for k,v in hm.items() if k.startswith('M_')},
                  'SC':selected_sc(sc),
                  'analysis_view_audit':loading}),sc


def main():
    p=argparse.ArgumentParser()
    for key in ['pilot','pilot-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();here=Path(__file__).parent
    import materialize_native30_new1695_v1 as io
    import standardize_native30_v1 as dsp
    for name,expected in PINS.items():
        io.require(io.digest(here/name)==expected,'code pin mismatch '+name)
        io.require(Path(importlib.import_module(Path(name).stem).__file__).resolve()==(here/name).resolve(),'wrong module import')
    io.require(io.digest(a.pilot/'COMMIT.json')==PILOT_SHA and io.digest(a.pilot_audit)==AUDIT_SHA,'pilot acceptance binding')
    commit=io.read_json(a.pilot/'COMMIT.json');summary=io.read_json(a.pilot/'summary.json')
    for name,value in commit['products'].items():io.require(io.digest(a.pilot/name)==value['sha256'],'pilot product changed')
    io.require(summary['completed']==8 and summary['failures']==[] and summary['classifier_fits']==0,'pilot not complete')
    paths={r['id']:a.pilot/'audio'/(r['id']+'.wav') for r in summary['records']}
    io.require(len(paths)==8,'eight unique pilot IDs required')
    bindings={str(here/name):io.digest(here/name) for name in PINS}
    bindings.update({str(Path(__file__)):io.digest(__file__),str(here/'test_run_native30_fhsc_pilot_v1.py'):io.digest(here/'test_run_native30_fhsc_pilot_v1.py'),str(a.pilot_audit):AUDIT_SHA,str(a.pilot/'COMMIT.json'):PILOT_SHA})
    import audio_inputs,phase_features,musical_features,stereo_candidate_v1
    contract={'status':'frozen_for_eight_item_measurement_integration_only','rows':sorted(paths),'bindings':bindings,
              'runtime':io.runtime_binding(dsp),'F_H_view':audio_inputs.CONFIG,'F_config':phase_features.CONFIG,
              'SC_config':clean(stereo_candidate_v1.CONFIG),'SC_predictors':SC_KEYS,
              'H_predictors':list(musical_features.H_MEASURE_NAMES),'M_min_context_seconds':musical_features.M_MIN_DURATION_SEC,
              'no_neural_inference':True,'classifier_fits':0,'cohort_admitted':False,'source_selection_changed':False}
    a.output.mkdir(exist_ok=False);io.write_new(a.output/'contract.json',contract)
    results=[]
    for ident,path in sorted(paths.items()):
        before=io.digest(path);result,sc=measure(path);io.require(io.digest(path)==before,'input changed')
        result.update(id=ident,input_path=str(path),input_file_sha256=before)
        np.savez_compressed(a.output/(ident+'.sc_frames.npz'),**sc['per_frame'],**sc['valid_masks'])
        io.write_new(a.output/(ident+'.json'),result);results.append(result)
        print(json.dumps({'id':ident,'F_status':result['F']['F_status'],'H_status':result['H']['H_status'],'SC_selected_six_status':result['SC']['selected_six_status'],'SC_full_grid_diagnostic_status':result['SC']['full_grid_diagnostic_status'],'M_status':result['M_diagnostic_not_predictor']['M_status']}),flush=True)
    for path,expected in bindings.items():io.require(io.digest(path)==expected,'binding changed')
    for name,value in commit['products'].items():io.require(io.digest(a.pilot/name)==value['sha256'],'pilot product changed')
    counts={'F':dict(Counter(r['F']['F_status'] for r in results)),'H':dict(Counter(r['H']['H_status'] for r in results)),
            'SC_selected_six':dict(Counter(r['SC']['selected_six_status'] for r in results)),
            'SC_full_grid_diagnostic':dict(Counter(r['SC']['full_grid_diagnostic_status'] for r in results)),
            'M_diagnostic':dict(Counter(r['M_diagnostic_not_predictor']['M_status'] for r in results))}
    io.write_new(a.output/'summary.json',{'status':'eight_item_integration_complete_not_accuracy','records':results,'availability':counts,'classifier_fits':0,'cohort_admitted':False})
    products={q.name:{'bytes':q.stat().st_size,'sha256':io.digest(q)} for q in sorted(a.output.iterdir()) if q.is_file()}
    io.write_new(a.output/'COMMIT.json',{'products':products,'pilot_COMMIT_sha256':PILOT_SHA,'pilot_audit_sha256':AUDIT_SHA,'classifier_fits':0})
    print(json.dumps({'status':'completed','COMMIT_sha256':io.digest(a.output/'COMMIT.json'),'availability':counts}),flush=True)


if __name__=='__main__':main()
