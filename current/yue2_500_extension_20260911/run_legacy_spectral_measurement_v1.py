"""Reuse frozen legacy DSP on YuE2 and baseline humans; no classifier fitting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OLD = Path('/mnt/nfs-code/users/yi/open_models_spectral_500_20260901')
BASE = Path('/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901')
CODE = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
DATA = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    out = DATA/'legacy_spectral_measurement_v1'
    out.mkdir(exist_ok=False)
    prompt = CODE/'prompts/prompt_manifest.jsonl'
    assert sha(prompt) == 'bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b'
    bias = OLD/'analysis/demucs_frequency_bias.npz'
    assert sha(bias) == 'bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e'
    # Bind the actual calibration before reading any class feature outcomes.
    # Cross-check identical legacy calibration copies rather than refitting.
    assert sha(bias) == sha(BASE/'demucs_frequency_bias.npz')
    humans = [r for r in map(json.loads, (BASE/'manifest.jsonl').read_text().splitlines()) if int(r['label']) == 0]
    prompts = list(map(json.loads, prompt.read_text().splitlines()))
    assert len(humans) == len(prompts) == 500
    rows = humans + [dict(id=r['id'], label=1, class_name='ai', source='yue2', split=r['split'],
                          prompt_id=r['id'], language=r['language']) for r in prompts]
    inventory = json.loads((DATA/'analysis_inputs_v1/inventory.json').read_text())
    assert inventory['rows'] == 500
    inputs = []
    for row in rows:
        name = f"{row['class_name']}_{row['source']}_{row['id']}"
        clip = (BASE/'input'/f'{name}.flac') if int(row['label']) == 0 else DATA/'analysis_inputs_v1/standardized'/f'{name}.flac'
        stem = (BASE/'output/htdemucs'/name) if int(row['label']) == 0 else DATA/'demucs_v1/htdemucs'/name
        for source, target in [(clip, out/'clips'/clip.name), (stem/'vocals.wav',out/'stems/htdemucs'/name/'vocals.wav')]:
            assert source.is_file(), str(source)
            inputs.append(dict(path=str(source), bytes=source.stat().st_size, sha256=sha(source)))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
    manifest = out/'manifest.jsonl'
    with manifest.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    libs = sorted((OLD/'code/analysis_lib').glob('*.py'))
    heatmap = OLD/'code/experiment/analyze_average_class_spectrogram.py'
    save(out/'INPUTS.json', dict(inputs=inputs, rows=1000, bias_sha256=sha(bias),
         manifest_sha256=sha(manifest), scripts={str(p):sha(p) for p in [*libs,heatmap]},
         native30_eligible=inventory['native30_eligible'], padded_rows=inventory['padded_rows'],
         classifier_fits=0, interpretation='legacy_first30_with_padding_not_native30; descriptive_maps_not_classifier_training'))
    print('All 2000 input audio files hashed; starting fixed raw/corrected DSP', flush=True)
    sys.path.insert(0,str(OLD/'code/analysis_lib'))
    import analyze_bias_corrected_vocals as analysis
    _,curve,_ = analysis.load_bias(bias)
    features = out/'features'
    features.mkdir()
    metrics,vectors = analysis.extract(rows,out/'clips',out/'stems',curve,features,False,4)
    assert len(metrics) == 2000
    env = dict(os.environ, PYTHONPATH=str(OLD/'code/analysis_lib'))
    subprocess.run([sys.executable,str(heatmap),'--manifest',str(manifest),'--stems-dir',str(out/'stems'),
                    '--bias-npz',str(bias),'--output-dir',str(out/'spectrograms'),
                    '--comparison-name','Human vs YuE2 (legacy first30 descriptive)'],check=True,env=env)
    for item in inputs:
        assert sha(Path(item['path'])) == item['sha256']
    products={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
              for p in out.rglob('*') if p.is_file() and not p.is_symlink()}
    save(out/'COMMIT.json',dict(status='completed_legacy_measurement_and_descriptive_maps',products=products,
                               classifier_fits=0,rows=1000,metric_rows=2000))
    print('Legacy spectral measurement and heatmaps committed',flush=True)


if __name__ == '__main__':
    main()
