#!/usr/bin/env python3
"""Post-run independent saved-output arithmetic/hash check; no new draws/fits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main(root):
    output=root/"results/uncertainty_sdrfh10_v1"
    manifest=json.loads((output/"run_manifest.json").read_text())
    checks=0
    def check(value,message):
        nonlocal checks
        if not value:
            raise ValueError(message)
        checks+=1
    check(manifest["status"]=="complete","incomplete output")
    for name,digest in manifest["outputs_sha256"].items():
        check(sha(output/name)==digest,"output hash: "+name)
    check(sha(root/"preregistration/uncertainty_sdrfh10_v1/protocol_frozen.json")==manifest["protocol_sha256"],"protocol hash")
    for name,digest in manifest["code_sha256"].items():
        check(sha(root/"code"/name)==digest,"code hash")
    group=np.load(output/"global_group_multiplicities.npz",allow_pickle=False)
    weights,ids=group["multiplicities"],group["group_id"]
    check(weights.shape==(1000,2339),"group draw shape")
    check(np.all(weights>=0) and np.all(weights.sum(axis=1)==2339),"group draw mass")
    check(list(ids)==sorted(set(ids)),"sorted unique global groups")
    digest=hashlib.sha256(json.dumps(list(ids),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    check(digest==manifest["group_universe_sha256"],"group identity hash")
    observed=rows(output/"bootstrap_replicates.csv")
    check(len(observed)==3000,"group comparison count")
    lookup={}
    for row in observed:
        key=(row["added_combination"],int(row["replicate"]))
        check(key not in lookup,"duplicate group replicate")
        lookup[key]=row
        w=weights[key[1]]
        check(hashlib.sha256(w.astype("<i8").tobytes()).hexdigest()==row["group_multiplicity_sha256"],"unpaired group multiplicities")
        check(int(row["resampled_draws"])==2339,"resampled draw count")
        check(int(row["nonzero_resampled_groups"])==int((w>0).sum()),"nonzero group count")
        for metric in ("auc","ba"):
            for direction in ("human","generator","equal_mean"):
                check(abs(float(row[f"added_{direction}_{metric}"])-float(row[f"baseline_{direction}_{metric}"])
                          -float(row[f"delta_{direction}_{metric}"]))<1e-12,"paired subtraction")
            for model in ("baseline","added"):
                check(abs(.5*(float(row[f"{model}_human_{metric}"])+float(row[f"{model}_generator_{metric}"]))
                          -float(row[f"{model}_equal_mean_{metric}"]))<1e-12,"equal directional macro")
    undefined=rows(output/"undefined_pair_metrics.csv")
    check(len(undefined)==104,"undefined row count")
    by_rep=Counter(int(r["replicate"]) for r in undefined)
    check(len(by_rep)==4,"undefined replicate count")
    check(all(int(r["human_weight"])==0 or int(r["ai_weight"])==0 for r in undefined),"undefined class support")
    for row in observed:
        check(int(row["defined_pair_cells_auc"])==910-by_rep[int(row["replicate"])],"defined AUC cell count")
        check(int(row["defined_pair_cells_ba"])==910-by_rep[int(row["replicate"])],"defined BA cell count")
    columns={"delta_J":"delta_equal_mean_auc","delta_equal_mean_balanced_accuracy":"delta_equal_mean_ba",
             "delta_human_auc":"delta_human_auc","delta_generator_auc":"delta_generator_auc",
             "delta_human_balanced_accuracy":"delta_human_ba","delta_generator_balanced_accuracy":"delta_generator_ba"}
    for summary in rows(output/"uncertainty_summary.csv"):
        values=np.array([float(lookup[(summary["added_combination"],i)][columns[summary["metric"]]]) for i in range(1000)])
        check(np.isfinite(values).all(),"finite macro differences")
        low,high=np.percentile(values,[2.5,97.5])
        check(abs(low-float(summary["percentile_95_ci_low"]))<1e-12,"group low percentile")
        check(abs(high-float(summary["percentile_95_ci_high"]))<1e-12,"group high percentile")
        check(abs(np.mean(values>0)-float(summary["probability_delta_gt_zero"]))<1e-12,"positive fraction")
    source=output/"source_sensitivity"
    saved=np.load(source/"source_multiplicities.npz",allow_pickle=False)
    hd,ad,matrix=saved["human"],saved["ai"],saved["matrix"]
    check(hd.shape==(10000,7) and ad.shape==(10000,13),"source draw shape")
    check(np.all(hd.sum(axis=1)==7) and np.all(ad.sum(axis=1)==13),"source draw mass")
    # Independent explicit two-direction weighted average, not runner einsum.
    rebuilt=np.empty((10000,3,2))
    for i in range(10000):
        pair_weight=hd[i,:,None]*ad[i,None,:]
        rebuilt[i]=(matrix*pair_weight[None,:,:,None,None]).sum(axis=(0,1,2))/(2*7*13)
    source_rows=rows(source/"replicates.csv")
    check(len(source_rows)==30000,"source comparison count")
    added=("S+D+F","S+D+H","S+D+F+H")
    for row in source_rows:
        observed=np.array([float(row["delta_J"]),float(row["delta_equal_mean_balanced_accuracy"])])
        check(np.max(np.abs(observed-rebuilt[int(row["replicate"]),added.index(row["added_combination"])]))<1e-12,"source reconstruction")
    for summary in rows(source/"sensitivity_summary.csv"):
        m=0 if summary["metric"]=="delta_J" else 1
        values=rebuilt[:,added.index(summary["added_combination"]),m]
        low,high=np.percentile(values,[2.5,97.5])
        check(abs(low-float(summary["empirical_source_95_low"]))<1e-12,"source low percentile")
        check(abs(high-float(summary["empirical_source_95_high"]))<1e-12,"source high percentile")
    print(json.dumps({"status":"passed","checks":checks,"result_manifest_sha256":sha(output/"run_manifest.json"),
                      "verifier_sha256":sha(Path(__file__)),"undefined_by_replicate":dict(sorted(by_rep.items())),
                      "no_new_resampling_or_refitting":True},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root.resolve())
