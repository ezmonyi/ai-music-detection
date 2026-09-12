"""Verify persisted predictions and export every fixed-model sensitivity cell."""
import csv
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict

ROOT=Path('/Users/yi/Documents/report/music_ai_detection_20260912/current_results')
SOURCE=ROOT/'native60_transfer_scores_v1'
PIN='34782ccfa467d5304ec2a4af1000d390eb467f004bb591ddce1474d06a068e64'
METADATA_PIN='d1e219e9a91370b7e470bac5c0deaaa70759b15ed49316f08b3ef5a4055771f1'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(root=ROOT, output=None):
    source=root/'native60_transfer_scores_v1'
    out=output if output is not None else root/'native60_transfer_report_v1'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite report directory: {out}')
    assert sha(source/'COMMIT.json')==PIN
    commit=json.loads((source/'COMMIT.json').read_text())
    for name,entry in commit['products'].items():
        assert sha(source/name)==entry['sha256'] and (source/name).stat().st_size==entry['bytes']
    summaries=json.loads((source/'per_model_summary.json').read_text())
    assert len(summaries)==9525
    metadata_path=root/'native60_feature_package_v1/metadata.json'
    assert sha(metadata_path)==METADATA_PIN
    metadata=json.loads(metadata_path.read_text())
    roles={r['id']:r['role'] for r in metadata};assert len(roles)==276
    expected={(r['combination'],r['quantity'],r['fold_index'],r['population']):r for r in summaries}
    assert len(expected)==9525
    counts=defaultdict(lambda:[0,0]);seen=defaultdict(set)
    with gzip.open(source/'predictions.csv.gz','rt',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['combination'],row['quantity'],row['fold_index']);uid=row['id']
            assert uid not in seen[key] and roles[uid]==row['role'];seen[key].add(uid)
            predicted=int(row['predicted_ai']);assert predicted==int(float(row['raw_score'])>=.5)
            assert row['model_sha256']==expected[(*key,'all_eligible')]['model_sha256']
            for population in ['all_eligible',row['role']]:
                counts[(*key,population)][0]+=1;counts[(*key,population)][1]+=predicted
    assert len(seen)==3175 and all(ids==set(roles) for ids in seen.values())
    for key,row in expected.items():
        n,tp=counts[key]
        assert (n,tp,n-tp)==(row['rows'],row['tp'],row['fn'])
        assert tp/n==row['ai_sensitivity']
        assert all(row[k] is None for k in ['balanced_accuracy','human_specificity','roc_auc'])
    groups=defaultdict(list)
    for row in summaries:groups[(row['combination'],row['quantity'],row['population'])].append(row)
    cells=[]
    for (combo,cap,pop),rows in sorted(groups.items()):
        assert len(rows)==5 and {r['fold_index'] for r in rows}==set(map(str,range(5)))
        values=[r['ai_sensitivity'] for r in rows]
        cells.append(dict(combination=combo,quantity=cap,population=pop,recordings_per_model=rows[0]['rows'],
            models=5,mean_sensitivity=statistics.mean(values),min_sensitivity=min(values),max_sensitivity=max(values),
            M_secondary_diagnostic='M' in combo.split('+')))
    assert len(cells)==1905
    out.mkdir(exist_ok=False)
    with (out/'all_subsets_caps_populations.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    lookup={(r['combination'],r['quantity'],r['population']):r for r in cells}
    combos=['S','D','R','P','F','H','M','S+D+R+P','S+D+R+P+F+H','S+D+R+P+F+H+M']
    tex=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrr}\toprule',
         r'Feature set & Locked (53) & Development (223) & All eligible (276)\\\midrule']
    for combo in combos:
        values=[lookup[(combo,'all',pop)]['mean_sensitivity']*100 for pop in ['locked_test','development','all_eligible']]
        tex.append(combo+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    tex += [r'\bottomrule\end{tabular}',r'\caption{YuE2 Native60 mean AI sensitivity (\%) over five frozen models at the all-cap setting. These are single-class results, not balanced accuracy. M-containing rows are secondary historical diagnostics.}',r'\label{tab:native60-transfer}\end{table}']
    with (out/'native60_transfer_tables.tex').open('x') as stream:stream.write('\n'.join(tex)+'\n')
    products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()}
    with (out/'COMMIT.json').open('x') as stream:
        json.dump(dict(status='persisted_predictions_verified_and_reported',source_commit_sha256=PIN,
            prediction_rows_verified=876300,per_model_summaries_verified=9525,aggregate_cells=1905,
            products=products,classifier_fits=0),stream,indent=2)
    print('Verified 876300 predictions; exported all 1905 sensitivity cells')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=ROOT,
        help='Directory containing native60_transfer_scores_v1 and native60_feature_package_v1')
    parser.add_argument('--output',type=Path,
        help='New report directory; existing directories are never overwritten')
    args=parser.parse_args()
    main(args.input_root,args.output)
