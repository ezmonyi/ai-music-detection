"""Create explicitly synthetic LaTeX for report layout QA; no actual outcomes."""
from pathlib import Path
import argparse
import test_summarize_bc_reserved_admission_v1 as fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sample = fixture.Fixture(gate=False, failures=True, scientific_nulls=2)
    try:
        sources = sample.sources()
        tables = fixture.m.machine_tables(sources)
        tex = fixture.m.render_latex(sources, tables)
        tex = tex.replace(r'\begin{document}', r'\begin{document}' + '\n' +
            r'\noindent\textbf{SYNTHETIC LAYOUT TEST ONLY. These are not experimental results.}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(tex)
    finally:
        sample.close()


if __name__ == '__main__':
    main()
