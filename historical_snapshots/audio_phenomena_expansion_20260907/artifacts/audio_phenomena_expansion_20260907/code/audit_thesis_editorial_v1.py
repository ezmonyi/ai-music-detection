"""Read-only source/PDF checks; emit JSON to stdout, not scientific validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from pypdf import PdfReader


def audit(master: Path, pdf: Path, workspace: Path) -> dict:
    visited: set[Path] = set()
    texts: list[str] = []
    missing_inputs: list[str] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        if not path.is_file():
            missing_inputs.append(str(path))
            return
        visited.add(path)
        text = path.read_text()
        texts.append(text)
        for name in re.findall(r'\\input\{([^}]+)\}', text):
            visit(path.parent / (name if name.endswith('.tex') else name + '.tex'))

    visit(master)
    combined = '\n'.join(texts)
    citations = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', combined)
                 for key in group.split(',')}
    bibfiles = [master.parent / (name.strip() + '.bib')
                for group in re.findall(r'\\bibliography\{([^}]+)\}', combined)
                for name in group.split(',')]
    bibtexts = [p.read_text() for p in bibfiles if p.is_file()]
    bibkeys = re.findall(r'@\w+\s*\{\s*([^,\s]+)', '\n'.join(bibtexts))
    labels = re.findall(r'\\label\{([^}]+)\}', combined)
    refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', combined)
    explicit_paths = sorted(set(p for p in re.findall(r'\\path\{([^}]+)\}', combined)
                                if p.startswith('artifacts/')))
    missing_paths = [p for p in explicit_paths if not (workspace / p).exists()]
    pages = [(p.extract_text() or '') for p in PdfReader(str(pdf)).pages]
    headings = []
    for number, text in enumerate(pages, 1):
        for phrase in ('Related work', 'Operational definitions',
                       'Additional deterministic', 'External measurement',
                       'Interpretation and generalization', 'Reproducibility',
                       'References'):
            if phrase in text:
                headings.append({'page': number, 'phrase': phrase})
    problems = {
        'missing_inputs': missing_inputs,
        'missing_bibliographies': [str(p) for p in bibfiles if not p.is_file()],
        'missing_citations': sorted(citations - set(bibkeys)),
        'duplicate_bibkeys': sorted(k for k in set(bibkeys) if bibkeys.count(k) > 1),
        'missing_labels': sorted(set(refs) - set(labels)),
        'duplicate_labels': sorted(k for k in set(labels) if labels.count(k) > 1),
        'missing_explicit_local_paths': missing_paths,
        'unresolved_pdf_markers_pages': [i for i, t in enumerate(pages, 1) if '??' in t],
    }
    return {
        'status': 'editorial_checks_pass' if not any(problems.values()) else 'editorial_checks_fail',
        'scope': 'syntax/link/selected PDF-text checks only; not numeric or full visual audit',
        'master': str(master.resolve()), 'pdf': str(pdf.resolve()),
        'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(visited | {p.resolve() for p in bibfiles if p.is_file()})},
        'pages': len(pages), 'citation_count': len(citations),
        'explicit_local_path_count': len(explicit_paths),
        'headings_by_page': headings, 'problems': problems,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.master, args.pdf, args.workspace)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'editorial_checks_pass' else 1)
