"""Export committed expanded YuE2 results without fitting or model selection."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/yi/Documents/report/music_ai_detection_20260912/current_results')
SOURCE=ROOT/'expanded_native30_report_v1'
PIN='b7934f2c649e5d0fbada832c335d36839a1aacb3bd7d6a674db83af5372bc911'
PROTOCOLS=['generator_holdout','human_source_holdout','ordinary_group_holdout_descriptive']


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert sha(SOURCE/'COMMIT.json')==PIN
    commit=json.loads((SOURCE/'COMMIT.json').read_text())
    for name,entry in commit['products'].items():
        assert sha(SOURCE/name)==entry['sha256'] and (SOURCE/name).stat().st_size==entry['bytes']
    summary=json.loads((SOURCE/'summary.json').read_text())
    rows=summary['primary_protocol_cells']
    assert len(rows)==3825 and all(r['feature_mode']=='values_plus_missing' for r in rows)
    lookup={(r['fold_type'],r['combination'],str(r['quantity'])):r for r in rows}
    assert len(lookup)==3825
    out=ROOT/'yue2_expanded_thesis_tables_v1';out.mkdir(exist_ok=False)
    fields=['fold_type','combination','quantity','feature_mode','macro_equal_group_equal_source_BA',
        'macro_ai_recall','macro_human_recall','macro_recording_weighted_within_experiment_BA',
        'descriptive_mean_single_model_fold_auc','available_held_source_experiments',
        'intended_held_source_experiments','eligible_model_count','omitted_model_count','status']
    with (out/'all_255_subsets_five_caps_three_protocols.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    combinations=['S','D','R','P','F','H','SC','BC','S+D+R+P','S+D+R+P+BC','S+D+R+P+F+H+SC','S+D+R+P+F+H+SC+BC']
    tex=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrr}\toprule',
         r'Feature set & Generator holdout & Human-source holdout & Grouped descriptive\\\midrule']
    for combo in combinations:
        values=[100*lookup[(p,combo,'all')]['macro_equal_group_equal_source_BA'] for p in PROTOCOLS]
        tex.append(combo+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    tex += [r'\bottomrule\end{tabular}',r'\caption{Native30 group/source-balanced accuracy (\%) at the all-cap configuration. Protocols have distinct estimands; no cross-column causal comparison is implied.}',r'\label{tab:bc-allcap}\end{table}',
            r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrrrr}\toprule',r'Protocol (all eight families) & 25 & 50 & 100 & 200 & All\\\midrule']
    for p,label in zip(PROTOCOLS,['Generator holdout','Human-source holdout','Grouped descriptive']):
        values=[100*lookup[(p,combinations[-1],cap)]['macro_equal_group_equal_source_BA'] for cap in ['25','50','100','200','all']]
        tex.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    tex += [r'\bottomrule\end{tabular}',r'\caption{All-eight-family training-group-cap ablation. Caps are per-source group budgets, not total training-song counts. The complete 3,825-cell table retains all 255 subsets at all five caps.}',r'\label{tab:bc-caps}\end{table}']
    tex=[line.replace('tab:bc-', 'tab:yue2-expanded-').replace('Native30 group/source-balanced', 'Expanded Native30 group/source-balanced') for line in tex]
    with (out/'yue2_expanded_tables_generated.tex').open('x') as stream:stream.write('\n'.join(tex)+'\n')
    products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()}
    with (out/'COMMIT.json').open('x') as stream:
        json.dump(dict(status='verified_report_projection_no_selection',source_commit_sha256=PIN,
            rows=3825,products=products,classifier_fits=0),stream,indent=2)
    print('Exported all 3825 primary cells and prespecified thesis tables')


if __name__=='__main__':main()
