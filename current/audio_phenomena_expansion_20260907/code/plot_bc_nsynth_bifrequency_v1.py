"""Descriptive all-grid visualization of independently accepted BC development.

Not a time-frequency spectrogram, significance map, gate or feature selection.
The all-grid visualization is post hoc; the original target remains unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

RESULT_SHA = 'c28e0136dfa76f7b0658f04ceeebcb3ae69bf32ddd19f02f149019d3f4e895c3'
AUDIT_SHA = '5ac2335ade358056f9ab123189aeee2122c5b37c58959b59747c4a69e5314c59'
GRID = [(a,b,a+b) for a in range(8,193,8) for b in range(a,193,8) if a+b<=256]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def require(value, message):
    if not value:
        raise ValueError(message)

def balanced(values, instruments):
    require(len(values)==len(instruments),'group length')
    groups = {}
    for value, instrument in zip(values,instruments):
        groups.setdefault(instrument,[])
        if value is not None:
            require(math.isfinite(value),'nonfinite value')
            groups[instrument].append(value)
    means=[math.fsum(v)/len(v) for v in groups.values() if v]
    return {'value': math.fsum(means)/len(means) if means else None,
            'covered_notes':sum(v is not None for v in values),
            'covered_instruments':len(means),
            'note_denominator':len(values),'instrument_denominator':len(groups)}

def panels(summary):
    notes=summary['per_note']
    require(len(notes)==54 and len({n['instrument'] for n in notes})==27,'cohort')
    instruments=[n['instrument'] for n in notes]
    for n in notes:
        for c in n['conditions'].values():
            require(len(c['grid_squared_bicoherence'])==len(c['grid_eligibility'])==228,'grid')
            for v,e in zip(c['grid_squared_bicoherence'],c['grid_eligibility']):
                require(type(e) is bool and e==(v is not None),'missingness')
                if v is not None:
                    require(math.isfinite(v) and 0<=v<=1,'b2 bounds')
    output={}
    for view in ('baseline','minus6db','0db'):
        cells=[]
        for index,triad in enumerate(GRID):
            if view=='baseline':
                values=[n['conditions']['baseline']['grid_squared_bicoherence'][index] for n in notes]
            else:
                pairs=[(n['conditions']['closed_'+view]['grid_squared_bicoherence'][index],
                        n['conditions']['independent_'+view]['grid_squared_bicoherence'][index]) for n in notes]
                values=[a-b if a is not None and b is not None else None for a,b in pairs]
            cells.append({'bins':list(triad),'parent_hz':[triad[0]*15.625,triad[1]*15.625],
                          **balanced(values,instruments)})
        output[view]=cells
    target=GRID.index((32,48,80))
    for view in ('minus6db','0db'):
        expected=summary['levels'][view]
        require(math.isclose(output[view][target]['value'],expected['equal_instrument_mean_difference'],
                             rel_tol=2e-12,abs_tol=2e-14),'prespecified target reconciliation')
        require(output[view][target]['covered_notes']==expected['covered_notes'],'target coverage')
    return output

def run(result,audit,output):
    result,audit,output=Path(result).resolve(),Path(audit).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(result),'new outside output')
    require(sha(result/'COMMIT.json')==RESULT_SHA and sha(audit)==AUDIT_SHA,'trusted receipts')
    commit=json.loads((result/'COMMIT.json').read_text())
    accepted=json.loads(audit.read_text())
    require(accepted['passed'] and accepted['result_commit_sha256']==RESULT_SHA
            and accepted['cells_checked']==86184 and accepted['classifier_fits']==0,'scientific audit scope')
    summary_file=result/'summary.json'
    require(sha(summary_file)==commit['products']['summary.json']['sha256']
            and summary_file.stat().st_size==commit['products']['summary.json']['bytes'],'summary binding')
    bindings={str(p):sha(p) for p in (result/'COMMIT.json',summary_file,audit,Path(__file__))}
    summary=json.loads(summary_file.read_text())
    data=panels(summary)
    output.mkdir()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parents=list(range(8,193,8))
    centers=np.asarray(parents)*15.625
    edges=np.r_[centers-62.5,centers[-1]+62.5]
    figure,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    titles=['Baseline squared bicoherence','Closed − independent: −6 dB','Closed − independent: 0 dB']
    for column,(view,cells) in enumerate(data.items()):
        values=np.full((24,24),np.nan)
        coverage=np.full((24,24),np.nan)
        for cell in cells:
            a,b,_=cell['bins']; x,y=parents.index(a),parents.index(b)
            values[y,x]=np.nan if cell['value'] is None else cell['value']
            coverage[y,x]=cell['covered_notes']
        for row,array in enumerate((values,coverage)):
            ax=axes[row,column]
            cmap=plt.get_cmap('viridis' if row or view=='baseline' else 'RdBu_r').copy()
            cmap.set_bad('#e4e4e4')
            lo,hi=(0,54) if row else ((0,1) if view=='baseline' else (-1,1))
            mesh=ax.pcolormesh(edges,edges,array,cmap=cmap,vmin=lo,vmax=hi,shading='flat',rasterized=True)
            ax.plot(500,750,marker='o',markersize=8,markerfacecolor='none',markeredgecolor='black',markeredgewidth=1.2)
            ax.set(xlabel='Parent frequency f₁ (Hz)',ylabel='Parent frequency f₂ (Hz)',
                   title=titles[column] if not row else 'Eligible notes / 54'+(' (paired)' if column else ''),
                   xlim=(62.5,3062.5),ylim=(62.5,3062.5))
            ax.set_aspect('equal')
            figure.colorbar(mesh,ax=ax,shrink=.8,label=('Notes' if row else 'b²' if not column else 'Δb²'))
    figure.suptitle('BC on NSynth backgrounds: descriptive bifrequency maps',fontsize=17)
    figure.supxlabel('Top: equal weight per covered instrument. Bottom: observed notes, not independent songs.\n'
                       'Circle = fixed 500 + 750 = 1250 Hz target. Gray = unmeasured grid or missing statistic.\n'
                       'Post-hoc visualization; no significance, causal-coupling or AI/Human accuracy claim.',fontsize=11)
    figure.savefig(output/'bifrequency_maps.png',dpi=160)
    figure.savefig(output/'bifrequency_maps.svg')
    plt.close(figure)
    payload={'scope':'posthoc_descriptive_full_grid_no_selection','panels':data,
             'target_bins':[32,48,80],'aggregation':'mean within instrument then equal weight covered instruments',
             'bounds':{'baseline':[0,1],'differences':[-1,1],'coverage':[0,54]},
             'symmetry_not_duplicated':True,'null_calibrated':False,'classifier_fits':0,
             'bindings':bindings,'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    (output/'plot_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    require(all(sha(p)==v for p,v in bindings.items()),'input changed during presentation')
    products={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir())}
    (output/'COMMIT.json').write_text(json.dumps({'status':'committed','kind':'descriptive_bifrequency_presentation',
        'products':products},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'committed','output':str(output),'products':len(products),
                      'target_minus6db':data['minus6db'][GRID.index((32,48,80))],
                      'target_0db':data['0db'][GRID.index((32,48,80))]}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('result','audit','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    run(args.result,args.audit,args.output)
