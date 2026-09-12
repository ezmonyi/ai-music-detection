"""Prepare exact native stereo center60 views without padding or inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
CODE = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def crop_bounds(frames, rate):
    required = 60 * rate
    return None if frames < required else ((frames-required)//2, required)


def main():
    manifest = CODE/'prompts/prompt_manifest.jsonl'
    assert sha(manifest) == 'bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows)==500 and len({r['id'] for r in rows})==500
    inventory_path = ROOT/'analysis_inputs_v1/inventory.json'
    inventory = json.loads(inventory_path.read_text())
    out = ROOT/'native60_inputs_v1'
    out.mkdir(exist_ok=False)
    (out/'audio').mkdir()
    write(out/'PROTOCOL.json',dict(manifest_sha256=sha(manifest),inventory_sha256=sha(inventory_path),
          code_sha256=sha(Path(__file__)),duration_s=60,position='native_center_floor',
          padding=False,resampling=False,downmix=False,normalization=False,classifier_fits=0,
          numpy=np.__version__,soundfile=sf.__version__,libsndfile=sf.__libsndfile_version__))
    eligible=[]; excluded=[]
    for index,row in enumerate(rows,1):
        raw_root=ROOT/'raw'/row['id']
        receipt_path=raw_root/'ACCEPTED.json'
        receipt=json.loads(receipt_path.read_text())
        assert receipt['id']==row['id'] and receipt['manifest_sha256']==sha(manifest)
        source=(raw_root/receipt['audio']).resolve()
        assert source.is_relative_to(raw_root.resolve())
        assert sha(source)==receipt['audio_sha256']
        with sf.SoundFile(source) as stream:
            assert stream.samplerate==48000 and stream.channels==2
            assert len(stream)==receipt['frames']
            bounds=crop_bounds(len(stream),stream.samplerate)
            entry=dict(id='ai_yue2_'+row['id'],prompt_id=row['id'],label=1,source_group='YuE2',
                       role=row['split'],group_id='muse:'+row['source_song_id'],
                       native_frames=len(stream),native_sample_rate_hz=stream.samplerate,
                       source_path=str(source),source_sha256=receipt['audio_sha256'],
                       source_receipt_sha256=sha(receipt_path))
            if bounds is None:
                excluded.append(dict(entry,reason='native_duration_below_60s'))
            else:
                start,count=bounds
                stream.seek(start)
                audio=stream.read(count,dtype='float32',always_2d=True)
                assert audio.shape==(2880000,2) and np.isfinite(audio).all()
                target=out/'audio'/(entry['id']+'.wav')
                sf.write(target,audio,48000,subtype='FLOAT')
                readback,sr=sf.read(target,dtype='float32',always_2d=True)
                assert sr==48000 and np.array_equal(audio,readback)
                eligible.append(dict(entry,start_frame=start,frames=count,duration_view_s=60,
                                     input_path=str(target),input_sha256=sha(target),
                                     waveform_float32_sha256=hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()))
        assert sha(source)==receipt['audio_sha256']
        if index%25==0:print(f'Native60 input audit {index}/500; eligible={len(eligible)}',flush=True)
    assert len(eligible)==inventory['native60_eligible']==276
    assert len(eligible)+len(excluded)==500
    write(out/'metadata.json',eligible)
    write(out/'excluded.json',excluded)
    products={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
              for p in out.rglob('*') if p.is_file()}
    write(out/'COMMIT.json',dict(status='native60_inputs_only_no_features_or_inference',
          eligible=len(eligible),excluded=len(excluded),products=products,classifier_fits=0))
    print('Native60 input preparation committed',flush=True)


if __name__=='__main__':
    main()
