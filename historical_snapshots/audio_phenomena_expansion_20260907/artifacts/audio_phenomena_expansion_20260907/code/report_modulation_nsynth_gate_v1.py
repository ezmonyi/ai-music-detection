"""Publish an English failed-gate report from independently checked immutable results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = "33a8d556847cf108b97abdc66ce6b59dfd9f22e697f342f5f0846f41934c7991"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source = ROOT / "results/modulation_nsynth_gate_v1"
    audit = ROOT / "audit/modulation_nsynth_gate_independent_v1.json"
    dsp = ROOT / "audit/modulation_nsynth_gate_dsp_replay_v1.json"
    result = json.loads((source / "gate_results.json").read_text())
    commit = json.loads((source / "COMMIT.json").read_text())
    checks = [json.loads(p.read_text()) for p in (audit, dsp)]
    if any(c["passed"] is not True or c["freeze_sha256"] != FREEZE or
           c["result_commit_sha256"] != sha(source / "COMMIT.json") for c in checks):
        raise ValueError("matching independent consistency and waveform audits required")
    for relative, record in commit["products"].items():
        path = source / relative
        if path.is_symlink() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("raw publication changed")
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    if actual != set(commit["products"]) | {"COMMIT.json"}:
        raise ValueError("raw inventory changed")
    decision = result["decision"]
    criteria = decision["criteria"]
    failed = [c for c in criteria if not c["passed"]]
    if len(criteria) != 146 or len(failed) != 5 or decision["criteria_passed"] or checks[0]["gate_criteria_passed"]:
        raise ValueError("this report is bound to the five-failure gate outcome")
    if checks[0]["failed_criterion_ids"] != [c["id"] for c in failed]:
        raise ValueError("independent failure identity mismatch")
    lines = ["# Held-out modulation measurement gate: failed admission", "",
        "The frozen six-descriptor Q candidate is **not admitted** to AI/Human classification. "
        "141 of 146 prospective checks passed; five failed. Successful execution and independent "
        "arithmetic verification do not turn this scientific gate failure into a pass.", "",
        "## Experiment and controls", "",
        "The 26 instrument identities reserved before the development pilot contribute two distinct-PCM "
        "notes each (52 notes). Eight fixed interventions and three views (original, MP3 128 kbit/s, "
        "AAC-LC requested 64 kbit/s) yield 1,248 measurements. Native input is 16 kHz mono, four seconds; "
        "the envelope analysis window is 0.25–2.25 s. No padding, lag optimization, parameter retuning, "
        "AI/Human labels or classifier fitting occurs. The source is the reused NSynth sample-library "
        "corpus, not an independent performed-song recording domain.", "",
        "Q contains fast modulation-power fraction and modulation entropy in each of three carrier "
        "bands (250–1,000; 1,000–3,000; 3,000–7,000 Hz). Carrier frequency is not modulation frequency. "
        "The entropy target is the response to chirped AM minus 48 Hz AM, both depth 0.6. "
        "A note succeeds at an entropy increase of at least 0.10; the instrument-balanced success "
        "rate must reach 90%. All three bands are mandatory.", "",
        "Codec nuisance is each note's maximum absolute feature difference across all eight paired "
        "conditions, divided by its positive original target response. Missing or nonpositive "
        "denominators remain unavailable. At least 80% of notes and instruments must be covered; "
        "at least 90% instrument-balanced success must have ratio ≤0.1, and every available ratio "
        "must be ≤1. Notes with only partial finite pairs have no asserted nuisance bound.", "",
        "## Failed prospective checks", "",
        "Rates average available notes within instrument, then covered instruments equally; they "
        "are not AI/Human accuracy and need not equal pooled note fractions.", "",
        "| Check | Valid notes | Instrument-balanced success | Required | Maximum nuisance ratio |",
        "|---|---:|---:|---:|---:|"]
    table_rows = []
    for c in failed:
        row = (c["id"], c["endpoint"]["valid_notes"], 100*c["success"]["instrument_balanced_mean"],
               c.get("maximum_available_ratio"))
        table_rows.append(row)
        ratio = "—" if row[3] is None else f"{row[3]:.6f}"
        lines.append(f"| `{row[0]}` | {row[1]}/52 | {row[2]:.3f}% | ≥90% | {ratio} |")
    lines += ["", "## Interpretation", "",
        "High-carrier entropy sensitivity falls below the fixed success threshold in all three "
        "views (88.462%), including uncompressed audio. Thus codec damage alone does not explain "
        "the sensitivity failure. AAC also fails the usual nuisance bound for low- and high-carrier "
        "entropy. Their maximum available ratios remain below one; the failed requirement is "
        "the proportion at or below 0.1, not catastrophic ratio >1 or insufficient coverage.", "",
        "All rate-recovery, fast-fraction target, depth-increase, eligibility, gain/polarity and "
        "remaining codec checks passed. This supports only a narrower descriptive observation; "
        "retaining the apparently successful subset would be post-hoc selection and cannot inherit "
        "the failed full-Q gate's validation. Any revised candidate requires a new definition and "
        "fresh external controls. The conditional multi-source Q classification design is therefore "
        "inactive. Existing admitted-family v6 evaluation remains unchanged.", "",
        "AAC actual stream rates range from 14,984 to 62,417 bit/s, rather than being forced to "
        "equal the requested 64,000. All 416 AAC decodes have 64,512 frames; the first 64,000 are "
        "analyzed and all 512-frame tails retained. All 416 MP3 decodes have 64,000 frames and "
        "128,000 bit/s. The two codec settings are not a bitrate-matched codec ranking.", "",
        "## Verification and artifact references", "",
        "Independent consistency replay checks 6,659 products, reconstructs 416 original "
        "interventions, independently probes 832 encoded streams, verifies 832 fixed decode "
        "slices, and recomputes all 146 decisions. Separate waveform DSP replay reproduces "
        "1,875,744 spectrum bins with zero numerical discrepancy using the same NumPy/SciPy "
        "libraries. It is not an independent-library replication. The consistency auditor does "
        "not independently decode encoded audio a second time; logs and retained decoded slices "
        "provide that linkage. Sixteen independent-auditor synthetic tests passed the parent rerun.", ""]
    references = [ROOT / "MODULATION_NSYNTH_GATE_PROTOCOL_EN.md",
        ROOT / "code/modulation_candidate_v1.py", ROOT / "code/modulation_nsynth_gate_v1.py",
        ROOT / "code/audit_modulation_nsynth_gate_v1.py", ROOT / "code/replay_modulation_gate_dsp_v1.py",
        ROOT / "preregistration/modulation_nsynth_gate_v1/draft.json", source / "gate_results.json",
        source / "COMMIT.json", audit, dsp,
        ROOT / "audit/modulation_nsynth_gate_independent_parent_tests_v1.log"]
    for path in references:
        lines.append(f"- [{path.relative_to(ROOT)}]({path}) — SHA-256 `{sha(path)}`")
    tex = [r"\section{Held-out modulation gate: failed admission}",
        r"\label{sec:q-heldout-gate-failed}",
        "The prospectively frozen Q candidate was not admitted to classification: 141 of 146 "
        "checks passed, but all criteria were required. We retained the 52-note, 26-instrument "
        "reserved NSynth partition, eight interventions and three codec views (1,248 measurements). "
        "These are measurement controls, not AI/Human song labels.",
        r"\begin{center}\begin{tabular}{lrr}\hline",
        r"Failed endpoint & Valid notes & Success (\%) \\ \hline"]
    labels = ["Original high-band entropy", "MP3 high-band entropy", "AAC high-band entropy",
              "AAC low-band entropy nuisance", "AAC high-band entropy nuisance"]
    for label, row in zip(labels, table_rows):
        tex.append(f"{label} & {row[1]}/52 & {row[2]:.3f}" + r" \\")
    tex += [r"\hline\end{tabular}\end{center}",
        "Success is instrument-balanced and requires at least 90\\%, not classification accuracy. "
        "The entropy sensitivity requirement is an increase of at least 0.10 for chirped versus "
        "48 Hz amplitude modulation. Codec nuisance ratios compare maximum eight-condition "
        "absolute error with each note's positive original target response; unavailable complete "
        "pairs remain missing. At least 90\\% must have ratio at most 0.1, with none above one. "
        "AAC maxima were 0.420455 and 0.632575 for the failed low/high entropy endpoints. "
        "Coverage was adequate; failure concerned the fixed success-rate requirement.",
        "Independent consistency audit reconstructed all 146 decisions. A separate full waveform "
        "replay reproduced 1,875,744 spectral bins exactly using the same numerical libraries. "
        "This accepts the reproducibility of a negative result, not Q admission. No thresholds "
        "were relaxed, bands discarded, or AI/Human classifiers fitted. Revised descriptors need "
        "fresh external validation. Existing admitted-family experiments are unaffected.",
        r"\paragraph{Reproducibility references.} Paths are relative to the preserved experiment root "
        r"\texttt{artifacts/audio\_phenomena\_expansion\_20260907}."]
    for path in references:
        escaped = str(path.relative_to(ROOT)).replace("_", r"\_")
        tex.append(r"\noindent\texttt{" + escaped + r"}\par")
    out = ROOT / "results/modulation_nsynth_gate_report_v1"
    out.mkdir(exist_ok=False)
    (out / "RESULTS_EN.md").write_text("\n".join(lines) + "\n")
    (out / "THESIS_SECTION_EN.tex").write_text("\n\n".join(tex) + "\n")
    receipt = {"code_sha256": sha(Path(__file__)), "freeze_sha256": FREEZE,
        "source_commit_sha256": sha(source / "COMMIT.json"), "checks": 146, "passed_checks": 141,
        "failed_criterion_ids": [c["id"] for c in failed], "classifier_admitted": False,
        "inputs": {str(p): sha(p) for p in references}, "main_thesis_pdf_rebuilt": False}
    (out / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    products = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in out.iterdir()}
    (out / "COMMIT.json").write_text(json.dumps({"status": "committed", "products": products}, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
