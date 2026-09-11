#!/usr/bin/env python3
"""Build a new dated thesis revision without modifying the historical source.

The update JSON contains reviewed LaTeX prose in abstract_extension and scope_note.
Numerical claims must be supplied only after the frozen evaluation finishes.
"""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old[:100]}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-json", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--consistency-audit", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Historical source must be preserved")
    if args.source.parent.resolve() != args.output.parent.resolve():
        raise ValueError("Keep the revised entry point beside the source to preserve all relative references")
    formal = json.loads(args.formal_audit.read_text())
    consistency = json.loads(args.consistency_audit.read_text())
    if formal.get("status") != "formal_complete" or consistency.get("status") != "passed":
        raise ValueError("Final thesis requires completed formal evaluation and a positive independent result audit")
    formal_digest = hashlib.sha256(args.formal_audit.read_bytes()).hexdigest()
    if consistency.get("input_sha256", {}).get(str(args.formal_audit.resolve())) != formal_digest:
        raise ValueError("Independent result audit is not bound to this completed formal audit")
    payload = json.loads(args.update_json.read_text())
    if payload.get("evaluation_complete") is not True:
        raise ValueError("Do not build a final-results revision before evaluation completion")
    original = args.source.read_text()
    text = replace_once(original, r"\author{Undergraduate Thesis Experiment Archive}",
                        r"\author{Thesis Experiment Archive}")
    text = replace_once(text, r"\fancyhead[R]{Updated: 2026-09-04}",
                        r"\fancyhead[R]{Updated: 2026-09-05}")
    text = replace_once(text,
                        "Four-family and Human catalogue-diversity extensions completed: September 4, 2026}",
                        "Earlier catalogue-diversity extension: September 4, 2026"
                        + r"\\Materialized expansion and re-evaluation: September 5, 2026}")
    old_abstract = (
        "A subsequent metadata-before-audio expansion freezes 2,091 Human tracks from nine catalogues "
        "without imposing 500 tracks per catalogue; 630 remain locked, and 144 culturally or acoustically "
        "complementary inputs have been materialized and checksum-verified. These new inputs have not yet "
        "been scored, so they improve the evaluation design but do not change any reported detector accuracy."
    )
    text = replace_once(text, old_abstract, payload["abstract_extension"])
    old_scope = (
        "All reported detector scores still use the earlier FMA/MTG-Jamendo Human controls; "
        "the new nine-catalogue Human manifest is frozen but unscored."
    )
    text = replace_once(text, old_scope, payload["scope_note"])
    text = replace_once(text,
                        "This study establishes separable spectral evidence and an exploratory S+D+R "
                        "development candidate; it does not establish a universal AI-music detector.",
                        "Historical within-domain spectral separation and exploratory development "
                        "candidates do not establish a universal AI-music detector.")
    catalogue_input = r"\input{../../source_diversity_expansion_20260904/latex/thesis_section_source_diversity_en}"
    text = replace_once(text, catalogue_input,
                        r"\begin{quote}\textbf{Historical acquisition-design snapshot (September 4).} "
                        r"The next section preserves the pre-materialization catalogue design. Its "
                        r"item-group counts and partial download totals are historical. The September 5 "
                        r"recording inventory, stronger creator/condition grouping, completed physical "
                        r"audit and numerical re-evaluation appear in Section~\ref{sec:expanded-materialized}."
                        r"\end{quote}" + "\n\n" + catalogue_input)
    text = replace_once(text, r"\section{Integrated Conclusions and Applied-Filter Design}",
                        r"\section{Earlier Applied-Filter Interpretation (September 4)}")
    text = replace_once(text, r"\subsection{Conclusions Supported by the Current Evidence}",
                        r"\noindent The following interpretation is retained as a dated record of the "
                        r"evidence through September 4. The completed expansion and updated conclusions "
                        r"follow in Section~\ref{sec:expanded-materialized}." + "\n\n"
                        + r"\subsection{Conclusions Supported by the Earlier Evidence}")
    insertion = (
        r"\input{../../source_diversity_expansion_20260905/latex/thesis_section_expanded_materialized_en}"
        + "\n\n"
        + r"\input{../../source_diversity_expansion_20260905/latex/thesis_updated_conclusions_en}"
        + "\n\n"
        + r"\section{Reproduction and Integrity}"
    )
    text = replace_once(text, r"\section{Reproduction and Integrity}", insertion)
    text = replace_once(text, r"\label{sec:repro}", r"\label{sec:repro}" + "\n\n"
                        + r"The September 5 expansion is preserved under "
                        r"\filepath{/Users/yi/Documents/code/music/artifacts/source_diversity_expansion_20260905/}. "
                        r"Its exact commands, immutable data roles, code/data hashes, feature-availability "
                        r"audits and frozen candidate files are indexed at the end of "
                        r"Section~\ref{sec:expanded-materialized}.")
    args.output.write_text(text)
    provenance = {
        "historical_source": str(args.source.resolve()),
        "historical_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "revised_source": str(args.output.resolve()),
        "revised_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "update_json_sha256": hashlib.sha256(args.update_json.read_bytes()).hexdigest(),
        "formal_audit_path": str(args.formal_audit.resolve()),
        "formal_audit_sha256": formal_digest,
        "independent_consistency_audit_path": str(args.consistency_audit.resolve()),
        "independent_consistency_audit_sha256": hashlib.sha256(args.consistency_audit.read_bytes()).hexdigest(),
        "historical_source_modified": False,
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
