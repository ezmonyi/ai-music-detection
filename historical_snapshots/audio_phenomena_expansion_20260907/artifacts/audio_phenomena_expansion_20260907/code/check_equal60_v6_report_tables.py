"""Read-only verification of the versioned English v6 summary tables."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    presentation = root / 'results/presentation_equal60_v6_v1'
    commit = json.loads((presentation / 'COMMIT.json').read_text())
    expected_commit = 'cc0fa8056aaf1f5e0cab36bb1d3386a3cb4c8860be53892a5a016d6af760c5bd'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(presentation / 'COMMIT.json') == expected_commit
    assert {p.name for p in presentation.iterdir()} == set(commit['files']) | {'COMMIT.json'}
    for name, binding in commit['files'].items():
        path = presentation / name
        assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
    def rows(name):
        with (presentation / name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    macro = rows('primary_macro_overview_255.csv')
    source = rows('primary_source_endpoints_all_arms.csv')
    aliases = {'B': 'S+D+R+P', 'B+SC': 'S+D+R+P+SC',
               'Seven': 'S+D+R+P+F+H+M', 'Eight': 'S+D+R+P+F+H+M+SC'}
    families = ['S', 'D', 'R', 'P', 'F', 'H', 'M', 'SC'] + list(aliases)
    fold_types = ['human_source_holdout', 'generator_holdout',
                  'ordinary_group_holdout_descriptive']
    def metric(family, fold, cap='all'):
        found = [r for r in macro if r['fold_type'] == fold
                 and r['combination'] == aliases.get(family, family) and r['quantity'] == cap]
        assert len(found) == 1
        return found[0]
    def ba(family, fold, cap='all'):
        return f"{100 * float(metric(family, fold, cap)['balanced_accuracy']):.2f}"
    report = root / 'EQUAL60_V6_RESULTS_EN.md'
    latex = root / 'EQUAL60_V6_THESIS_SECTION_EN.tex'
    counts = {'markdown_single_addition_rows': 0, 'markdown_cap_rows': 0,
              'markdown_source_rows': 0, 'latex_single_addition_rows': 0, 'latex_cap_rows': 0}
    source_names = {'MTG-Jamendo': 'MTG-Jamendo', 'MAESTRO': 'human_maestro_v3',
                    'MedleyDB': 'human_medleydb', 'MoisesDB': 'human_moisesdb',
                    'Saraga': 'human_saraga_hindustani_v1', 'Mureka': 'Mureka_v9', 'Suno': 'Suno'}
    for line in report.read_text().splitlines():
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == 7 and cells[0] in families:
            expected = []
            for fold in fold_types:
                expected += [f"{float(metric(cells[0], fold)['roc_auc']):.4f}", ba(cells[0], fold)]
            assert cells[1:] == expected, (cells, expected)
            counts['markdown_single_addition_rows'] += 1
        elif len(cells) == 7 and cells[0] in ['25', '50', '100', '200', 'all']:
            expected = [ba(family, fold, cells[0]) for fold in fold_types for family in ['Seven', 'Eight']]
            assert cells[1:] == expected, (cells, expected)
            counts['markdown_cap_rows'] += 1
        elif len(cells) == 4 and cells[0].split(' (')[0] in source_names:
            name = source_names[cells[0].split(' (')[0]]
            expected = []
            for family in ['SC', 'Seven', 'Eight']:
                found = [r for r in source if r['quantity'] == 'all'
                         and r['heldout_source'] == r['source_group'] == name
                         and r['combination'] == aliases.get(family, family)]
                assert len(found) == 1
                expected.append(found[0]['correct_recordings'] + '/' + found[0]['unique_oof_recordings'])
            assert cells[1:] == expected
            counts['markdown_source_rows'] += 1
    for line in latex.read_text().splitlines():
        cells = [c.strip() for c in line.removesuffix('\\\\').split('&')]
        if len(cells) == 4 and cells[0] in families:
            assert cells[1:] == [ba(cells[0], fold) for fold in fold_types]
            counts['latex_single_addition_rows'] += 1
        elif len(cells) == 3 and cells[0] in ['25', '50', '100', '200', 'all']:
            assert cells[1:] == [ba(family, 'generator_holdout', cells[0]) for family in ['Seven', 'Eight']]
            counts['latex_cap_rows'] += 1
    assert list(counts.values()) == [12, 5, 7, 12, 5], counts
    print(json.dumps({'status': 'passed', 'counts': counts,
                      'presentation_commit_sha256': expected_commit,
                      'verified_presentation_products': len(commit['files']),
                      'verified_presentation_bytes': sum(v['bytes'] for v in commit['files'].values()),
                      'bindings': {p.name: sha(p) for p in [report, latex, Path(__file__)]},
                      'scope': 'Exact numerical table transcription and mirror integrity; prose interpretation parent-reviewed; not independent statistical validation.'}, indent=2))


if __name__ == '__main__':
    main()
