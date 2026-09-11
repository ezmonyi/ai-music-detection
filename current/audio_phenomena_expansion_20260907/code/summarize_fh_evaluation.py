#!/usr/bin/env python3
"""Render the complete F/H extension lattice; no classifier fitting/selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ("S", "D", "R", "P", "F", "H")
QUANTITIES = ("25", "50", "100", "200", "all")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.evaluation_dir / "run_manifest.json").read_text())
    if manifest["stage"] != "dev":
        raise ValueError("This renderer is restricted to development evidence")
    raw = pd.read_csv(a.evaluation_dir / "development_source_holdout_summary.csv")
    raw["quantity"] = raw.quantity.astype(str)
    measures = ("roc_auc_source_macro", "balanced_accuracy_source_macro")
    h = raw[raw.fold_type == "human_source_holdout"]
    g = raw[raw.fold_type == "generator_holdout"]
    joined = h.merge(g, on=["cohort_id", "candidate_key", "combination", "quantity"],
                     suffixes=("_human", "_ai"), validate="one_to_one")
    for input_name, output_name in zip(measures, ("J", "mean_BA")):
        joined[output_name] = .5 * (joined[input_name + "_human"] + joined[input_name + "_ai"])
    coverage = pd.read_csv(a.evaluation_dir / "candidate_coverage.csv")
    joined = joined.merge(coverage[["candidate_key", "coverage_qualified", "eligible_rows"]],
                          on="candidate_key", validate="many_to_one")
    keys = sorted(joined.combination.unique(), key=lambda k: (len(k.split("+")), [ORDER.index(c) for c in k.split("+")]))
    if len(keys) != 63 or set(joined.quantity) != set(QUANTITIES) or len(joined) != 315:
        raise ValueError(f"Incomplete 63 x 5 lattice: {len(keys)} combinations, {len(joined)} rows")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(a.output_dir / "all_63x5_numeric_results.csv", index=False)
    matrix = np.array([[float(joined[(joined.combination == k) & (joined.quantity == q)].J.iloc[0])
                        for q in QUANTITIES] for k in keys])
    fig, ax = plt.subplots(figsize=(10, 21))
    im = ax.imshow(matrix, vmin=.45, vmax=.90, aspect="auto", cmap="viridis")
    ax.set_xticks(range(5), ["25", "50", "100", "200", "All"])
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Maximum global groups per training source; cell = J / mean BA", labelpad=12)
    for r, key in enumerate(keys):
        for c, q in enumerate(QUANTITIES):
            row = joined[(joined.combination == key) & (joined.quantity == q)].iloc[0]
            text = f"{row.J:.3f} / {row.mean_BA:.3f}" + ("*" if not row.coverage_qualified else "")
            ax.text(c, r, text, ha="center", va="center", fontsize=7,
                    color="white" if row.J < .73 else "black")
    fig.colorbar(im, ax=ax, fraction=.025, pad=.03, label="J: equal mean of Human-source and AI-generator macro AUC")
    fig.suptitle("Exploratory source-transfer evaluation: S/D/R/P + phase F + tonal path H\n"
                 "Fixed 30 s cohort; not an untouched external test; * = coverage not qualified", y=.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .976))
    fig.savefig(a.output_dir / "all_63x5_auc_ba.png", dpi=180)
    plt.close(fig)
    all_rows = joined[joined.quantity == "all"].set_index("combination")
    report = ["# Phase and tonal-path extension: development results", "",
              "This is exploratory development evaluation on previously assembled sources. "
              "Historical locked labels are not used for fitting or selection. "
              "No confidence interval or future-source guarantee is implied by these point estimates.", "",
              "J is the equal mean of Human-source-heldout and AI-generator-heldout macro AUC. "
              "BA is the corresponding mean balanced accuracy. All comparisons below use the "
              "same realized cohort, folds and training-quantity policy, not September 5's different seed.", "",
              "| Combination | Human-source AUC | AI-generator AUC | J | Mean BA | Coverage qualified |",
              "|---|---:|---:|---:|---:|---|"]
    show = ["S", "D", "R", "P", "F", "H", "F+H", "S+D+R", "S+D+R+F", "S+D+R+H", "S+D+R+F+H", "S+D+R+P+F+H"]
    for key in show:
        row = all_rows.loc[key]
        report.append(f"| {key} | {row.roc_auc_source_macro_human:.4f} | {row.roc_auc_source_macro_ai:.4f} | "
                      f"{row.J:.4f} | {row.mean_BA:.4f} | {bool(row.coverage_qualified)} |")
    report += ["", "## Matched increments relative to S+D+R", ""]
    for key in ("S+D+R+F", "S+D+R+H", "S+D+R+F+H"):
        row, base = all_rows.loc[key], all_rows.loc["S+D+R"]
        report.append(f"- {key}: delta J = {row.J-base.J:+.4f}; delta mean BA = {row.mean_BA-base.mean_BA:+.4f}.")
    report += ["", "## Limits and next checks", "",
               "- This table reports all predefined candidates; it does not designate a deployable winner.",
               "- F is an exploratory phase descriptor. External intervention sensitivity does not establish AI authorship.",
               "- H is a chroma path descriptor, not chord correctness. Natural annotated construct validation remains separate.",
               "- M requires a separately verified >=45 s cohort. V/B/A/T are pending; they are not replaced by F/H.",
               "- Inspect source-specific false positives and the separate missingness-only/median-only diagnostics.",
               "- Paired global-group uncertainty, codec robustness and the historical descriptive comparison remain necessary before interpretation is finalized.", "",
               "Complete numeric lattice: `all_63x5_numeric_results.csv`; figure: `all_63x5_auc_ba.png`."]
    (a.output_dir / "RESULTS_FH_EN.md").write_text("\n".join(report) + "\n")
    audit = {"rows": len(joined), "combinations": len(keys), "quantities": list(QUANTITIES),
             "evaluation_manifest_sha256": hashlib.sha256((a.evaluation_dir / "run_manifest.json").read_bytes()).hexdigest(),
             "no_fitting_or_model_selection": True}
    (a.output_dir / "presentation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
