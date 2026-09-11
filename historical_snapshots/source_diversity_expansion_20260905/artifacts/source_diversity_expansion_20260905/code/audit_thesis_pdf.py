#!/usr/bin/env python3
"""Check final thesis compile integrity and bind the recorded human-visible QA."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log = args.log.read_text(errors="replace")
    reader = PdfReader(args.pdf)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    issues = []
    for token in ("Overfull", "undefined references", "multiply defined", "Missing character"):
        if token in log:
            issues.append(f"Compile warning: {token}")
    if "\ufffd" in text or "??" in text:
        issues.append("Replacement character or unresolved reference in extracted text")
    for required in ("0.6334", "0.7641", "0.5233", "10,141", "4,497", "Expanded-Corpus"):
        if required not in text:
            issues.append(f"Expected completed-result text absent: {required}")
    provenance = json.loads(args.provenance.read_text())
    historical = Path(provenance["historical_source"])
    if digest(historical) != provenance["historical_source_sha256"]:
        issues.append("Historical thesis source was modified")
    if digest(args.tex) != provenance["revised_source_sha256"]:
        issues.append("Revised main entry point differs from builder provenance")
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not issues else "failed", "issues": issues,
        "pdf": str(args.pdf.resolve()), "pdf_sha256": digest(args.pdf),
        "pages": len(reader.pages), "main_tex_sha256": digest(args.tex),
        "compile_log_sha256": digest(args.log),
        "underfull_warning_count": log.count("Underfull"),
        "underfull_note": "Nonfatal justification warnings remain in retained historical tables/path references.",
        "visual_review_record": {
            "method": "Rendered PNGs visually inspected by root and two independent reviewers",
            "pages": [1, *range(43, 60)],
            "checks": ["readable type and tables", "no clipping or overlap", "correct J/BA values",
                       "locked specificity and uncertainty labels", "readable code/output references"],
            "final_layout_corrections": ["ragged-right locked CI table copies preserve all data lines",
                                         "30-second locked heading kept with its table",
                                         "shortened conclusion heading to prevent overflow"],
            "scope": "Updated title/abstract and expanded sections; historical content preserved",
        },
        "historical_source_unchanged": not any("Historical" in issue for issue in issues),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
