"""Independent native60 acceptance: reconstruct membership and crop from raw audio."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    commit=json.loads((root/'COMMIT.json').read_text())
    assert commit['status']=='native60_inputs_only_no_features_or_inference'
    assert digest(args.manifest)=='bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b'
    for name,binding in commit['products'].items():
        path=(root/name).resolve()
        assert path.is_relative_to(root) and path.stat().st_size==binding['bytes']
        assert digest(path)==binding['sha256']
    eligible=json.loads((root/'metadata.json').read_text())
    excluded=json.loads((root/'excluded.json').read_text())
    prompts=[json.loads(line) for line in args.manifest.read_text().splitlines()]
    records=eligible+excluded
    assert len(records)==500 and len({r['id'] for r in records})==500
    lookup={r['prompt_id']:r for r in records}
    assert set(lookup)=={r['id'] for r in prompts}
    eligible_ids={r['id'] for r in eligible}
    expected_eligible=[]
    for i,prompt in enumerate(prompts,1):
        row=lookup[prompt['id']]
        assert row['id']=='ai_yue2_'+prompt['id'] and row['role']==prompt['split']
        assert row['group_id']=='muse:'+prompt['source_song_id']
        assert row['label']==1 and row['source_group']=='YuE2'
        source=Path(row['source_path'])
        assert digest(source)==row['source_sha256']
        # Read the full original independently of the producer's seek/crop path.
        raw,rate=sf.read(source,dtype='float32',always_2d=True)
        assert rate==48000 and raw.shape==(row['native_frames'],2)
        assert np.isfinite(raw).all()
        if len(raw)<2880000:
            assert row['id'] not in eligible_ids and row['reason']=='native_duration_below_60s'
        else:
            expected_eligible.append(row['id'])
            assert row['id'] in eligible_ids
            start=(len(raw)-2880000)//2
            assert row['start_frame']==start and row['frames']==2880000 and row['duration_view_s']==60
            expected=raw[start:start+2880000]
            target=Path(row['input_path'])
            assert target.resolve().is_relative_to(root)
            assert digest(target)==row['input_sha256']
            observed,sr=sf.read(target,dtype='float32',always_2d=True)
            assert sr==48000 and np.array_equal(observed,expected)
            assert sf.info(target).subtype=='FLOAT'
            assert hashlib.sha256(expected.astype('<f4').tobytes()).hexdigest()==row['waveform_float32_sha256']
        assert digest(source)==row['source_sha256']
        if i%25==0:print(f'Independent native60 audit {i}/500',flush=True)
    assert expected_eligible==[r['id'] for r in eligible]
    assert len(eligible)==commit['eligible']==276 and len(excluded)==commit['excluded']==224
    result=dict(status='independent_native60_membership_and_full_decode_crop_verified',
                originals_verified=500,crops_verified=276,excluded_verified=224,
                source_commit_sha256=digest(root/'COMMIT.json'),verifier_sha256=digest(Path(__file__)),
                classifier_fits=0,neural_inference=False)
    with args.receipt.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
