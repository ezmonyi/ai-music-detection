"""Descriptive GuitarSet bifrequency maps after independently accepted replay.

The displayed grid is not a time-frequency STFT frame or a significance map.
Pool-paired contrasts precede recording and equal-score aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

RESULT_SHA = 'cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c'
GRID = [(a,b,a+b) for a in range(8,193,8) for b in range(a,193,8) if a+b<=256]
CONDITIONS = ('baseline','common_gain','polarity','closed_minus6db','independent_minus6db','closed_0db','independent_0db')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def mean(values):
    return math.fsum(values)/len(values) if values else None


def cell_summary(pool_values, scores):
    require(len(pool_values)==len(scores),'recording/score lengths')
    grouped={score:[] for score in scores}
    recording_values=[]
    covered_pools=0
    for pools,score in zip(pool_values,scores):
        require(len(pools)==2,'two pools per recording')
        values=[value for value in pools if value is not None]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in values),'finite cell values')
        value=mean(values)
        recording_values.append(value)
        covered_pools+=len(values)
        if value is not None:
            grouped[score].append(value)
    group_means=[mean(values) for values in grouped.values() if values]
    return {'value':mean(group_means),'covered_pools':covered_pools,
            'pool_denominator':2*len(scores),
            'covered_recordings':sum(v is not None for v in recording_values),
            'recording_denominator':len(scores),'covered_scores':len(group_means),
            'score_denominator':len(grouped)}


def panels(summary):
    records=summary['per_recording']
    require(len(records)==90 and len({r['item_id'] for r in records})==90,'90 unique development recordings')
    require(len({r['score_id'] for r in records})==15 and len({r['player_id'] for r in records})==3,'crossed development groups')
    for record in records:
        require(record['split_role']=='development','development only')
        require(set(record['conditions'])==set(CONDITIONS),'seven conditions')
        for condition in record['conditions'].values():
            require([p['pool_index'] for p in condition['pools']]==[0,1],'ordered two pools')
            for pool in condition['pools']:
                values,mask=pool['grid_squared_bicoherence'],pool['grid_eligibility']
                require(len(values)==len(mask)==228,'frozen grid size')
                for value,eligible in zip(values,mask):
                    require(type(eligible) is bool and eligible==(value is not None),'mask/missingness')
                    if value is not None:
                        require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'b2 bounds')
    output={}
    scores=[r['score_id'] for r in records]
    for view in ('baseline','minus6db','0db'):
        cells=[]
        for index,triad in enumerate(GRID):
            rows=[]
            for record in records:
                conditions=record['conditions']
                if view=='baseline':
                    values=[p['grid_squared_bicoherence'][index] for p in conditions['baseline']['pools']]
                else:
                    pairs=zip(conditions['closed_'+view]['pools'],conditions['independent_'+view]['pools'])
                    values=[]
                    for closed,independent in pairs:
                        a,b=closed['grid_squared_bicoherence'][index],independent['grid_squared_bicoherence'][index]
                        values.append(a-b if a is not None and b is not None else None)
                rows.append(values)
            cells.append({'bins':list(triad),'parent_hz':[triad[0]*15.625,triad[1]*15.625],**cell_summary(rows,scores)})
        output[view]=cells
    target=GRID.index((32,48,80))
    for view in ('minus6db','0db'):
        cell,reference=output[view][target],summary['levels'][view]
        expected=reference['equal_score']['equal_group_mean_difference']
        require(cell['value'] is None if expected is None else cell['value'] is not None
                and math.isclose(cell['value'],expected,rel_tol=2e-12,abs_tol=2e-14),'prespecified target value')
        require(cell['covered_recordings']==reference['covered_recordings']
                and cell['covered_pools']==reference['paired_covered_pools'],'prespecified target coverage')
    require(output['baseline'][target]['covered_pools']==summary['baseline']['target_covered_pools'],'baseline target coverage')
    require(sum(c['covered_pools'] for c in output['baseline'])==summary['baseline']['grid_eligible_cells'],'baseline grid coverage')
    return output


def run(result,audit,audit_sha256,output):
    result,audit,output=Path(result).resolve(),Path(audit).resolve(),Path(output).resolve()
    require(not output.exists() and output.parent.is_dir() and not output.is_relative_to(result),'exclusive outside output')
    require(len(audit_sha256)==64 and sha(audit)==audit_sha256,'caller-trusted audit hash')
    require(sha(result/'COMMIT.json')==RESULT_SHA,'producer COMMIT')
    commit=json.loads((result/'COMMIT.json').read_text())
    accepted=json.loads(audit.read_text())
    require(accepted['passed'] is True and accepted['result_commit_sha256']==RESULT_SHA
            and accepted['recordings']==90 and accepted['cells_checked']==287280
            and accepted['stft_complex_coefficients_checked']==1260*247*513
            and accepted['construction_and_WAV_bit_exact'] is True
            and accepted['classifier_fits']==0 and accepted['external_gate_passed'] is False,'independent replay scope')
    summary_file=result/'summary.json'
    require(sha(summary_file)==commit['products']['summary.json']['sha256']
            and summary_file.stat().st_size==commit['products']['summary.json']['bytes'],'summary binding')
    paths=(result/'COMMIT.json',summary_file,audit,Path(__file__),Path(__file__).with_name('test_plot_bc_guitarset_bifrequency_v1.py'))
    bindings={str(path):sha(path) for path in paths}
    data=panels(json.loads(summary_file.read_text()))
    output.mkdir()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parents=list(range(8,193,8)); centers=np.asarray(parents)*15.625
    edges=np.r_[centers-62.5,centers[-1]+62.5]
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    titles=['Baseline squared bicoherence','Closed - independent: -6 dB','Closed - independent: 0 dB']
    for column,(view,cells) in enumerate(data.items()):
        values=np.full((24,24),np.nan); coverage=np.full((24,24),np.nan)
        for cell in cells:
            a,b,_=cell['bins']; x,y=parents.index(a),parents.index(b)
            values[y,x]=np.nan if cell['value'] is None else cell['value']
            coverage[y,x]=cell['covered_recordings']
        for row,array in enumerate((values,coverage)):
            ax=axes[row,column]
            cmap=plt.get_cmap('viridis' if row or view=='baseline' else 'RdBu_r').with_extremes(bad='#e4e4e4')
            lo,hi=(0,90) if row else ((0,1) if view=='baseline' else (-1,1))
            mesh=ax.pcolormesh(edges,edges,array,cmap=cmap,vmin=lo,vmax=hi,shading='flat',rasterized=True)
            ax.plot(500,750,marker='o',markersize=8,markerfacecolor='none',markeredgecolor='black',markeredgewidth=1.2)
            ax.set(xlabel='Parent frequency f1 (Hz)',ylabel='Parent frequency f2 (Hz)',
                   title=titles[column] if not row else 'Covered recordings / 90'+(' (paired pools)' if column else ''),
                   xlim=(62.5,3062.5),ylim=(62.5,3062.5))
            ax.set_aspect('equal')
            fig.colorbar(mesh,ax=ax,shrink=.8,label='Recordings' if row else 'b²' if not column else 'Δb²')
    fig.suptitle('BC on GuitarSet backgrounds: descriptive bifrequency maps',fontsize=17)
    fig.supxlabel('Top: baseline pool means or paired pool differences; recording means, then equal covered score means.\n'
                  'Bottom: recordings with at least one covered pool, not independent performers.\n'
                  'Circle = fixed target. Gray = unmeasured grid or missing value. No significance or AI/Human accuracy claim.',fontsize=10)
    fig.savefig(output/'bifrequency_maps.png',dpi=160);fig.savefig(output/'bifrequency_maps.svg');plt.close(fig)
    payload={'scope':'posthoc_descriptive_all_grid_no_selection','panels':data,'target_bins':[32,48,80],
             'aggregation':'paired eligible pools then recording mean then equal covered score mean',
             'bounds':{'baseline':[0,1],'differences':[-1,1],'coverage':[0,90]},
             'not_an_STFT_frame_heatmap':True,'independent_performers':False,'null_calibrated':False,
             'classifier_fits':0,'bindings':bindings,'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    with (output/'plot_data.json').open('x') as stream:
        json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    require(all(sha(path)==digest for path,digest in bindings.items()),'presentation inputs changed')
    products={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir())}
    with (output/'COMMIT.json').open('x') as stream:
        json.dump({'status':'committed','kind':'guitarset_descriptive_bifrequency_presentation','products':products},stream,indent=2,sort_keys=True)
        stream.write('\n')
    return {'status':'committed','products':len(products),'output':str(output),'commit_sha256':sha(output/'COMMIT.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('result','audit','audit-sha256','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.result,args.audit,args.audit_sha256,args.output),sort_keys=True))
