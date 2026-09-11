#!/usr/bin/env python3
"""Create layout-only copies of frozen result tables; preserve every data row."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    records = []
    old = r"\begin{longtable}{p{2.6cm}p{1.8cm}p{1.8cm}p{2.7cm}p{2.7cm}p{2.7cm}}"
    widths = ("2.6cm", "1.8cm", "1.8cm", "2.7cm", "2.7cm", "2.7cm")
    new = r"\begin{longtable}{" + "".join(
        r">{\raggedright\arraybackslash}p{" + width + "}" for width in widths
    ) + "}"
    for duration in ("10s", "30s"):
        source = root / "latex" / f"results_{duration}_locked_specificity.tex"
        target = root / "latex" / f"presentation_results_{duration}_locked_specificity.tex"
        original = source.read_text()
        if original.count(old) != 1:
            raise ValueError("Unexpected frozen column specification")
        revised = original.replace(old, new, 1)
        assert original.splitlines()[1:] == revised.splitlines()[1:]
        target.write_text(revised)
        records.append({"source": str(source), "source_sha256": sha(source),
                        "presentation_copy": str(target), "presentation_sha256": sha(target),
                        "all_data_and_note_lines_unchanged": True})
    audit = {"status": "passed", "scope": "Column alignment only; frozen exports untouched",
             "code_sha256": sha(Path(__file__)), "tables": records}
    (root / "audit/thesis_table_presentation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
