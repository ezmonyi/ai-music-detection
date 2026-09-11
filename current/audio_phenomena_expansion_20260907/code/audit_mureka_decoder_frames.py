#!/usr/bin/env python3
"""Read-only decoder discrepancy diagnosis; never changes an admission gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.source/'contract.json').read_text())
    results = []
    for row in contract['selected']:
        receipt_path = args.source/'items'/(row['id']+'.json')
        receipt = json.loads(receipt_path.read_text())
        path = args.source/'raw'/row['path']
        info = sf.info(path)
        result = dict(id=row['id'], path=str(path), source_expected_sha256=row['sha256'],
                      receipt_sha256=sha(receipt_path), sf_header_frames=info.frames,
                      sf_sample_rate=info.samplerate, sf_channels=info.channels,
                      ffmpeg_decoded_frames=receipt['decoded_frames'],
                      ffmpeg_sample_rate=receipt['sample_rate'], ffmpeg_channels=receipt['channels'])
        result['header_delta_frames'] = info.frames-receipt['decoded_frames']
        result['header_matches'] = (info.frames, info.samplerate, info.channels) == (
            receipt['decoded_frames'], receipt['sample_rate'], receipt['channels'])
        if not result['header_matches']:
            assert sha(path) == row['sha256'], 'Changed source'
            count, finite, h = 0, True, hashlib.sha256()
            with sf.SoundFile(path) as f:
                for data in f.blocks(blocksize=65536, dtype='float32', always_2d=True):
                    count += len(data)
                    finite = finite and bool(np.isfinite(data).all())
                    h.update(np.ascontiguousarray(data, dtype='<f4').tobytes())
            result.update(sf_actual_decoded_frames=count, sf_full_decode_finite=finite,
                          sf_full_decode_float32_sha256=h.hexdigest(),
                          source_actual_sha256=sha(path))
            assert result['source_actual_sha256'] == row['sha256'], 'Source changed during diagnosis'
            print(json.dumps(result), flush=True)
        results.append(result)
    result = dict(status='diagnosis_only', code_sha256=sha(__file__),
                  source_contract_sha256=sha(args.source/'contract.json'),
                  soundfile_version=sf.__version__, libsndfile_version=sf.__libsndfile_version__,
                  count=len(results), header_mismatch_count=sum(not r['header_matches'] for r in results),
                  changed_admission=False, classifier_fitted=False, records=results)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}), flush=True)


if __name__ == '__main__':
    main()
