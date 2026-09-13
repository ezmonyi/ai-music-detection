"""Inventory project Python/shell sources, pruning dependency environments."""
import hashlib
import json
import os
from pathlib import Path

ROOTS = ['demucs_bias_corrected_1000_20260901', 'demucs_oracle_vocal_eval_20260901',
    'dynamics_rhythm_change_detector_extension_20260903', 'dynamics_rhythm_external_benchmark_20260903',
    'external_filter_test_20260902', 'external_generator_500_heuristics_20260904',
    'external_generator_500_testset_20260904', 'four_family_diversity_ablation_20260904',
    'music_flamingo_midi_rhythm_probe_20260903', 'open_models_spectral_500_20260901',
    'source_diversity_expansion_20260904', 'source_diversity_expansion_20260905']
SKIP = {'venv', '.venv', 'env', 'site-packages', 'node_modules', '.git', '__pycache__',
    'third_party', 'third-party', 'vendor', '.cache', 'checkpoints', 'models'}


def main():
    base = Path('/mnt/nfs-code/users/yi')
    rows = []
    pruned = []
    for name in ROOTS:
        root = base / name
        files = set(root.glob('*.py')) | set(root.glob('*.sh'))
        for sub in ['code', 'scripts']:
            for folder, dirs, names in os.walk(root / sub, followlinks=False):
                for d in list(dirs):
                    p = Path(folder) / d
                    if d in SKIP or (p / 'pyvenv.cfg').is_file() or p.is_symlink():
                        dirs.remove(d)
                        pruned.append(str(p.relative_to(base)))
                files.update(Path(folder) / n for n in names if Path(n).suffix in {'.py', '.sh'})
        for p in sorted(files):
            data = p.read_bytes()
            rows.append(dict(path=str(p.relative_to(base)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest()))
        print(json.dumps(dict(root=name, files=len(files))), flush=True)
    out = base / 'yue2_500_extension_20260911/HISTORICAL_SERVER_CODE_AUDIT_V2_20260913.json'
    with out.open('x') as stream:
        json.dump(dict(roots=ROOTS, pruned=pruned, files=rows,
            scope='Top-level and code/scripts Python and shell files; dependency directories excluded'), stream, indent=2)
    print(json.dumps(dict(files=len(rows), bytes=out.stat().st_size)), flush=True)


if __name__ == '__main__':
    main()
