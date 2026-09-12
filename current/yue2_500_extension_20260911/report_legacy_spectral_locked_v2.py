"""Export the complete fixed legacy comparison; no outcome-driven selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from verify_local_package_v1 import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();receipt=verify(args.source)
    summary=json.loads((args.source/'summary.json').read_text())
    rows=summary['results'];args.output.mkdir(exist_ok=False)
    text='''# Legacy fullmix and vocal spectral comparison: YuE2

This is a within-generator historical-holdout comparison, not a new external
generator-generalization benchmark. Training contains 400 human and 400 YuE2
recordings; testing contains 100 human and 100 YuE2 recordings. Human artist
groups and YuE2 source-song groups do not cross the fixed split. Human test
recordings were used in earlier experiments; locked status does not imply
this is their first use. No representation is selected as a winner here.

The unchanged legacy ridge implementation uses targets -1/+1, alpha 10 and
decision threshold 0 (not the later Native30 pipeline's 0.5). Training-only
means/scales are reused at test time. The first adapter run stopped when an
independent BA calculation exposed its incorrect 0.5 threshold assumption;
v2 restores the original definition. This is a code correction, not tuning
the threshold on these test outcomes. Both scripts and the failure remain.

The input view is the historical first30 representation, including two
short YuE2 originals padded by the earlier standardizer. Do not equate this
with center-native30. Fullmix receives no separator correction. The two vocal
conditions reuse the frozen external MUSDB response curve. The test compares
fixed frequency bands and metrics; row-wise development CV is not reported as
group-independent evidence. No confidence intervals or statistical winner
claims are made in this report.

| Representation | Features | Balanced accuracy | ROC AUC | Human specificity | AI sensitivity |
|---|---:|---:|---:|---:|---:|
'''
    for r in rows:
        text+=f"| {r['representation']} | {r['dimension']} | {r['balanced_accuracy']:.3f} | {r['roc_auc']:.4f} | {r['human_specificity']:.3f} | {r['ai_sensitivity']:.3f} |\n"
    text+='''
## Interpretation limits and references

High scores can reflect source, production, codec, vocal-activity and excerpt
position differences, not uniquely AI authorship. The average spectrograms
are not phrase-aligned and exhibit substantial time-dependent contrasts.
Do not interpret a high within-generator score as robustness to new generators.
These results must be presented alongside source-held-out Native30 comparisons.

Source predictions, model parameters and protocol are in the sibling
`legacy_spectral_locked_v2` directory. Raw/corrected features and heatmaps are
in `legacy_spectral_measurement_v1`; the fullmix control is `legacy_fullmix_v1`.
Scripts: `evaluate_legacy_spectral_locked_v2.py` and
`report_legacy_spectral_locked_v2.py`, in the YuE2 extension source directory.
'''
    (args.output/'REPORT_EN.md').write_text(text)
    with (args.output/'all_representations.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    products={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in args.output.iterdir()}
    (args.output/'COMMIT.json').write_text(json.dumps(dict(source_verification=receipt,products=products),indent=2))
    print(json.dumps(dict(representations=len(rows),status='complete_table_exported')))


if __name__=='__main__':main()
