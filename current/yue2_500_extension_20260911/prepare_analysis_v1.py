"""Audit fixed YuE2 native audio and reuse the original 30s standardizer."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
from run_yue2_frozen import digest, requests, save

RC = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
RD = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OLD = Path('/mnt/nfs-code/users/yi/open_models_spectral_500_20260901')
SHA = 'bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b'


def main():
    manifest = RC/'prompts/prompt_manifest.jsonl'
    assert digest(manifest) == SHA
    rows = requests(manifest)
    standardizer = OLD/'code/experiment/standardize_generated_audio.py'
    assert digest(standardizer) == '26f31ad04414c173eab1dd9a660ba6db264d352f68afde1217de64f67d58dd6d'
    spec = importlib.util.spec_from_file_location('original_standardizer', standardizer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = RD/'analysis_inputs_v1'
    out.mkdir(exist_ok=True)
    (out/'receipts').mkdir(exist_ok=True)
    verified = []
    for i, row in enumerate(rows):
        root = RD/'raw'/row['id']
        receipt_path = root/'ACCEPTED.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['manifest_sha256'] == SHA and receipt['id'] == row['id']
        assert receipt['request'] == dict(style=row['prompt_common'], lyrics=row['lyrics_30s'], seed=row['seed'], cot='full')
        source = (root/receipt['audio']).resolve()
        assert source.is_relative_to(root.resolve()) and digest(source) == receipt['audio_sha256']
        audio, sr = sf.read(source, always_2d=True, dtype='float32')
        assert sr == 48000 and audio.shape[1] == 2 and len(audio) > 0 and np.isfinite(audio).all()
        assert receipt['frames'] == len(audio)
        uid = 'ai_yue2_' + row['id']
        target = out/'standardized'/f'{uid}.flac'
        record = out/'receipts'/f'{uid}.json'
        if record.exists():
            item = json.loads(record.read_text())
            assert item['source_sha256'] == digest(source)
            assert item['output_sha256'] == digest(target)
        else:
            assert not target.exists(), 'Unreceipted output: review before retry; never overwrite'
            item = module.standardize(source, target)
            item.update(id=uid, prompt_id=row['id'], split=row['split'], language=row['language'],
                        accepted_receipt_sha256=digest(receipt_path), truncated=receipt['truncated'],
                        native30_eligible=len(audio) >= 30*sr, native60_eligible=len(audio) >= 60*sr,
                        native_finite=True, generator='yue2', manifest_sha256=SHA,
                        standardizer_sha256=digest(standardizer))
            save(record, item)
        assert item['output_frames'] == 1323000 and item['output_sample_rate'] == 44100 and item['output_channels'] == 2
        verified.append(item)
        if (i+1) % 25 == 0: print(f'Native audit and standardization {i+1}/500', flush=True)
    summary = dict(status='native_audio_audited_and_standardized_not_feature_evaluated', rows=len(verified),
                   native30_eligible=sum(x['native30_eligible'] for x in verified),
                   native60_eligible=sum(x['native60_eligible'] for x in verified),
                   padded_rows=sum(x['padding_frames'] > 0 for x in verified),
                   min_native_seconds=min(x['source_duration_s'] for x in verified),
                   max_native_seconds=max(x['source_duration_s'] for x in verified),
                   metadata=verified)
    save(out/'inventory.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'metadata'}), flush=True)
    demucs = OLD/'code/experiment/run_demucs_generated.py'
    assert digest(demucs) == '12d20df30ddfed4a01dd0938002a4b39fd216410ae0700ba6134cd43bd9ddfe3'
    py = '/mnt/nfs-code/users/yi/soulx_svc_cnceleb_moon10_260831/.venv/bin/python'
    # Fail closed if the selected 5090 no longer has sufficient free memory.
    memory = subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=name,memory.free','--format=csv,noheader,nounits'], text=True)
    name, free = memory.strip().rsplit(',', 1)
    assert '5090' in name and int(free) >= 10000, 'GPU 7 unavailable; keep inputs and defer separation'
    subprocess.run([sys.executable, str(demucs), '--input-dir', str(out/'standardized'),
                    '--output-dir', str(RD/'demucs_v1'), '--python', py,
                    '--torch-home', str(OLD/'models/demucs'),
                    '--state-file', str(out/'demucs_state.jsonl'), '--batch-size','10','--gpu-index','7'], check=True)
    print('Demucs finished; feature extraction and classification remain separate stages.', flush=True)


if __name__ == '__main__': main()
