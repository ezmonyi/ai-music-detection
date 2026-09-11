#!/usr/bin/env python3
"""Independent, read-only numerical audit; does not import the producer/extractor."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/stereo_music_controls_v1'
FREEZE = ROOT / 'preregistration/stereo_music_controls_frozen_v1.json'
OUT = ROOT / 'audit/stereo_music_controls_independent_audit_v1.json'
BANDS = [(80,500),(500,2000),(2000,6000),(6000,12000),(12000,20000)]
VARIANTS = ['gain_half','swap','iid_plus_3db','iid_plus_6db','iid_plus_12db','width_0.5','width_2']
CODECS = ['mp3_128k','mp3_256k','m4a_128k','m4a_256k']
METRICS = ['iid_db','signed_real_coherence','magnitude_coherence','side_energy_fraction','ipd_increment_rad']
TOL = 1e-8

def read(path):
    return json.loads(path.read_text())

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}

def finite(value):
    return value is not None and np.isfinite(value)

def delta(a,b):
    return abs(b-a) if finite(a) and finite(b) else None

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    problems=[]
    def check(ok, message):
        if not ok:
            problems.append(message)
    def same(actual, saved, message, atol=1e-14):
        if actual is None or saved is None:
            check(actual is None and saved is None,message)
        elif isinstance(actual,(bool,int,np.integer)):
            check(actual==saved,message)
        else:
            check(bool(np.isclose(actual,saved,atol=atol,rtol=1e-12)),message)

    freeze=read(FREEZE); commit=read(RESULTS/'COMMIT.json'); summary=read(RESULTS/'summary.json')
    check(digest(FREEZE)==commit['freeze_sha256'],'commit freeze hash')
    check(digest(FREEZE)==summary['freeze_sha256'],'summary freeze hash')
    for path,value in freeze['bindings'].items():
        check(digest(Path(path))==value,'freeze binding '+path)
    total_bytes=0
    for relative, expected in commit['products'].items():
        path=RESULTS/relative
        check(path.is_file(),'missing product '+relative)
        if path.is_file():
            check(path.stat().st_size==expected['bytes'],'size '+relative)
            check(digest(path)==expected['sha256'],'hash '+relative)
            total_bytes+=path.stat().st_size
    actual_files={str(p.relative_to(RESULTS)) for p in RESULTS.rglob('*') if p.is_file() and p.name!='COMMIT.json'}
    check(actual_files==set(commit['products']),'committed product inventory')

    gates=[]; ratio_rows=[]; scalar_count=0; frame_pair_count=0; alignment_rows=[]
    max_iid_error=0.; max_width_error=0.; max_invariance_error=0.
    for source in freeze['selected']:
        track_id=source['track_id']; directory=RESULTS/track_id
        check(digest(Path(source['mixture']))==source['input_sha256'],'input hash '+track_id)
        comparisons=read(directory/'comparisons.json')['comparisons']
        baseline=arrays(directory/'original_frames.npz')
        features=read(directory/'original_features.json')['features']
        for variant in VARIANTS:
            prefix=track_id+'/'+variant
            changed=arrays(directory/(variant+'_frames.npz'))
            changed_features=read(directory/(variant+'_features.json'))['features']
            receipt=comparisons[variant]
            for key,value in features.items():
                expected_delta=delta(value,changed_features[key]); scalar=receipt['scalars'][key]
                same(expected_delta,scalar['absolute_delta'],prefix+'/'+key+'/scalar_delta')
                same(bool(finite(value)!=finite(changed_features[key])),scalar['missingness_changed'],prefix+'/'+key+'/missingness')
                scalar_count+=1
            for bi,(lo,hi) in enumerate(BANDS):
                saved=receipt['band_checks'][bi]
                if variant in ['gain_half','swap']:
                    keys=[k for k in features if k.startswith(f'SC_{lo}_{hi}hz_')]
                    changes=int(sum(finite(features[k])!=finite(changed_features[k]) for k in keys))
                    differences=[delta(features[k],changed_features[k]) for k in keys]
                    differences=[v for v in differences if v is not None]
                    mx=max(differences) if differences else None
                    recalculated=dict(common_finite_features=len(differences),missingness_changes=changes,
                        max_absolute_error=mx,passed=bool(differences and mx<=TOL and changes==0))
                    max_invariance_error=max(max_invariance_error,mx or 0.)
                else:
                    if variant.startswith('iid'):
                        change=float(variant.split('_')[-1][:-2]); original=baseline['iid_db'][bi]
                        target=original+change
                        target[(abs(original)>=60)|(abs(target)>=60)]=np.nan
                        observed=changed['iid_db'][bi]
                    else:
                        width=float(variant.split('_')[-1]); q=baseline['side_energy_fraction'][bi]
                        target=width**2*q/(1-q+width**2*q); observed=changed['side_energy_fraction'][bi]
                    common=np.isfinite(target)&np.isfinite(observed)
                    errors=abs(observed[common]-target[common]); n=int(common.sum())
                    fraction=n/len(common) if len(common) else 0.
                    mx=float(errors.max()) if n else None
                    recalculated=dict(common_frames=n,common_fraction=fraction,
                        median_absolute_error=float(np.median(errors)) if n else None,
                        max_absolute_error=mx,coverage_pass=bool(n>=30 and fraction>=.5),
                        arithmetic_pass=bool(mx is not None and mx<=TOL),
                        passed=bool(n>=30 and fraction>=.5 and mx is not None and mx<=TOL))
                    if variant.startswith('iid'):
                        max_iid_error=max(max_iid_error,mx or 0.)
                    else:
                        max_width_error=max(max_width_error,mx or 0.)
                for key,value in recalculated.items():
                    same(value,saved[key],prefix+'/'+str(bi)+'/'+key)
                check(saved['band']==[lo,hi],prefix+'/band')
                check(saved['primary']==(bi<4),prefix+'/primary')
                gates.append(dict(track_id=track_id,variant=variant,band=[lo,hi],primary=bi<4,**recalculated))

        for codec in CODECS:
            receipt=comparisons[codec]; alignment=receipt['alignment']
            alignment_rows.append(dict(track_id=track_id,codec=codec,**alignment))
            if not alignment['eligible']:
                check((directory/(codec+'_full_decode.wav')).is_file(),'rejected decode persistence')
                continue
            original_features=read(directory/(codec+'_reference_features.json'))['features']
            codec_features=read(directory/(codec+'_features.json'))['features']
            targets=read(directory/(codec+'_matched_target_features.json'))
            expected_ratio_keys={k for k in original_features if k.endswith('abs_iid_db_median') or k.endswith('side_energy_fraction_median')}
            check(set(receipt['target_response_ratios'])==expected_ratio_keys,'ratio feature inventory '+track_id+'/'+codec)
            for key,value in original_features.items():
                same(delta(value,codec_features[key]),receipt['scalars'][key]['absolute_delta'],track_id+'/'+codec+'/'+key+'/scalar')
                scalar_count+=1
            for key,saved in receipt['target_response_ratios'].items():
                target='iid_plus_6db' if key.endswith('abs_iid_db_median') else 'width_2'
                denominator=delta(original_features[key],targets[target][key]); numerator=delta(original_features[key],codec_features[key])
                ratio=numerator/denominator if numerator is not None and denominator is not None and denominator>TOL else None
                same(numerator,saved['codec_delta'],'ratio numerator '+track_id+'/'+codec+'/'+key)
                same(denominator,saved['target_delta'],'ratio denominator '+track_id+'/'+codec+'/'+key)
                same(ratio,saved['ratio'],'ratio '+track_id+'/'+codec+'/'+key)
                check(saved['target']==target,'ratio target '+track_id+'/'+codec+'/'+key)
                ratio_rows.append(dict(track_id=track_id,codec=codec,feature=key,ratio=ratio))

        # Recompute all reported frame-pair differences independently from NPZ.
        for name,receipt in comparisons.items():
            if name in CODECS and not receipt['alignment']['eligible']:
                continue
            a=arrays(directory/(name+'_reference_frames.npz')) if name in CODECS else baseline
            b=arrays(directory/(name+'_frames.npz'))
            check(len(receipt['frame_differences'])==25,'frame pair inventory '+track_id+'/'+name)
            for saved in receipt['frame_differences']:
                bi=BANDS.index(tuple(saved['band'])); metric=saved['metric']; x=a[metric][bi]; y=b[metric][bi]
                common=np.isfinite(x)&np.isfinite(y); differences=y[common]-x[common]
                if metric=='ipd_increment_rad':
                    differences=np.angle(np.exp(1j*differences))
                differences=abs(differences)
                values=dict(common_frames=int(common.sum()),common_fraction=float(common.mean()),
                    original_valid_frames=int(np.isfinite(x).sum()),transformed_valid_frames=int(np.isfinite(y).sum()),
                    frame_mask_changes=int(np.count_nonzero(np.isfinite(x)!=np.isfinite(y))),
                    median_absolute_delta=float(np.median(differences)) if differences.size else None,
                    max_absolute_delta=float(differences.max()) if differences.size else None)
                for key,value in values.items():
                    same(value,saved[key],'frame pair '+track_id+'/'+name+'/'+metric+'/'+str(bi)+'/'+key)
                frame_pair_count+=1

    # Direct FFT validation for one preselected recording, independently of scipy.signal.stft.
    first=freeze['selected'][0]['track_id']; directory=RESULTS/first
    sr,wave=wavfile.read(directory/'original.wav'); check(sr==44100,'FFT source sample rate')
    nfft=4096; hop=1024; window=.5-.5*np.cos(2*np.pi*np.arange(nfft)/nfft)
    frames=np.lib.stride_tricks.sliding_window_view(wave,nfft,axis=0)[::hop]
    spectra=np.fft.rfft(frames*window[None,None,:],axis=-1)/window.sum()
    frequencies=np.fft.rfftfreq(nfft,1/sr); saved=arrays(directory/'original_frames.npz'); fft_rows=[]
    for bi,(lo,hi) in enumerate(BANDS):
        pick=(frequencies>=lo)&(frequencies<hi); left=spectra[:,0,pick]; right=spectra[:,1,pick]
        el=np.sum(abs(left)**2,axis=1); er=np.sum(abs(right)**2,axis=1); total=el+er
        cross=np.sum(left*np.conjugate(right),axis=1); energy=total>=max(1e-10,float(total.max())*1e-6)
        two=energy&(el>=np.maximum(5e-11,total*1e-8))&(er>=np.maximum(5e-11,total*1e-8))
        with np.errstate(divide='ignore',invalid='ignore'):
            iid=np.clip(10*(np.log10(el)-np.log10(er)),-60,60)
            signed=np.clip(cross.real/(np.sqrt(el)*np.sqrt(er)),-1,1)
            magnitude=np.clip(abs(cross)/(np.sqrt(el)*np.sqrt(er)),0,1)
            mid_energy=np.sum(abs((left+right)/2)**2,axis=1); side_energy=np.sum(abs((left-right)/2)**2,axis=1)
            side=np.clip(side_energy/(mid_energy+side_energy),0,1)
        for values in [iid,signed,magnitude]:
            values[~two]=np.nan
        side[~energy]=np.nan; phase_valid=two&np.isfinite(magnitude)&(magnitude>=.1)
        increment_valid=np.zeros(len(total),dtype=bool); increment_valid[1:]=phase_valid[1:]&phase_valid[:-1]
        increments=np.full(len(total),np.nan); phase=np.angle(cross)
        increments[1:][increment_valid[1:]]=np.angle(np.exp(1j*np.diff(phase)[increment_valid[1:]]))
        calculated=dict(iid_db=iid,signed_real_coherence=signed,magnitude_coherence=magnitude,
            side_energy_fraction=side,ipd_increment_rad=increments,energy_valid=energy,two_sided_valid=two,
            ipd_phase_valid=phase_valid,ipd_increment_valid=increment_valid)
        for metric,values in calculated.items():
            stored=saved[metric][bi]
            if values.dtype==bool:
                check(np.array_equal(values,stored),'FFT mask '+str(bi)+'/'+metric)
                continue
            check(np.array_equal(np.isfinite(values),np.isfinite(stored)),'FFT metric mask '+str(bi)+'/'+metric)
            common=np.isfinite(values)&np.isfinite(stored); errors=values[common]-stored[common]
            if metric=='ipd_increment_rad':
                errors=np.angle(np.exp(1j*errors))
            max_error=float(np.max(abs(errors))) if errors.size else None
            check(max_error is None or max_error<=1e-10,'FFT values '+str(bi)+'/'+metric)
            fft_rows.append(dict(band=[lo,hi],metric=metric,common_frames=int(common.sum()),max_absolute_error=max_error))

    primary=[g for g in gates if g['primary']]
    check(len(gates)==24*7*5,'all gate count')
    check(len(primary)==summary['primary_checks'],'primary count receipt')
    check(sum(g['passed'] for g in primary)==summary['passed_checks'],'passed count receipt')
    check(all(g['passed'] for g in primary)==summary['measurement_arithmetic_pass'],'overall arithmetic receipt')
    check(summary['classifier_admitted'] is False,'classifier not admitted')
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),audit_code_sha256=digest(Path(__file__)),
        freeze_sha256=digest(FREEZE),commit_sha256=digest(RESULTS/'COMMIT.json'),
        scope='Independent numerical validation of reused measurement controls; no classification admission or independence claim.',
        producer_imported=False,products_verified=len(commit['products']),product_bytes_verified=total_bytes,
        source_inputs_hashed=len(freeze['selected']),tracks=len(freeze['selected']),
        arithmetic_checks=len(gates),primary_arithmetic_checks=len(primary),primary_passed=sum(g['passed'] for g in primary),
        exploratory_passed=sum(g['passed'] for g in gates if not g['primary']),
        max_iid_target_error=max_iid_error,max_width_target_error=max_width_error,max_gain_swap_scalar_error=max_invariance_error,
        scalar_receipt_checks=scalar_count,frame_pair_receipt_checks=frame_pair_count,
        codec_ratio_checks=len(ratio_rows),codec_ratios_available=sum(r['ratio'] is not None for r in ratio_rows),
        codec_alignments=len(alignment_rows),codec_alignments_eligible=sum(a['eligible'] for a in alignment_rows),
        direct_numpy_fft=dict(track_id=first,tolerance=1e-10,checks=fft_rows),
        gate_recalculations=gates,codec_ratio_recalculations=ratio_rows,
        issues=problems,passed=not problems,python=sys.version,numpy=np.__version__)
    serialized=json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)
    with OUT.open('x') as stream:
        stream.write(serialized+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['gate_recalculations','codec_ratio_recalculations','direct_numpy_fft']},indent=2))
    if problems:
        raise SystemExit(1)

if __name__=='__main__':
    main()
