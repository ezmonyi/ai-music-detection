"""YuE2 transfer of the unchanged center8 BC measurement, with array receipts."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import fcntl
import sys
import numpy as np
import soundfile as sf

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
RD = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT = RD/'native30_bc_v1'
sys.path.insert(0, str(RC))
import run_native30_bc_v2 as bc


def one(row):
    uid, source_receipt, expected = row
    assert bc.base.digest(source_receipt) == expected
    source = bc.base.read_json(source_receipt)
    audio_path = Path(source['view_path'])
    assert bc.base.digest(audio_path) == source['view_sha256']
    receipt = OUT/'items'/(uid+'.json')
    if receipt.exists():
        result = bc.base.read_json(receipt)
        for key in ('arrays', 'metadata', 'input'):
            assert bc.base.file_binding(result[key]['path']) == result[key]
        return result
    bc.graph.inspect_audio(audio_path, input_audio=True)
    values, sr = sf.read(audio_path, dtype='float64', always_2d=True)
    assert sr == 44100
    wave, view = bc.analysis_view(values)
    _, numerical, scalar = bc.numerical_modules()
    result = numerical.extract(wave, 16000)
    metadata = result['metadata']
    metadata['construction_status'] = view['construction_status']
    reduced = scalar.reduce_metadata(metadata, crop=bc.CROP)
    assert reduced['pool_count'] == 2 and reduced['discarded_tail_samples'] == 0
    bc.validate_view(view)
    bc.validate_arrays(result['arrays'], metadata)
    arrays_path = OUT/'arrays'/(uid+'.npz')
    metadata_path = OUT/'metadata'/(uid+'.json')
    assert not arrays_path.exists() and not metadata_path.exists()
    bc.write_arrays(arrays_path, result['arrays'])
    bc.base.write_new(metadata_path, metadata)
    with np.load(arrays_path, allow_pickle=False) as saved:
        bc.validate_arrays(dict(saved), metadata)
    record = dict(id=uid, input=bc.base.file_binding(audio_path),
                  source_receipt=bc.base.file_binding(source_receipt),
                  analysis_view=view, scalar=reduced,
                  features={bc.FEATURE:reduced['median_squared_bicoherence']},
                  arrays=bc.base.file_binding(arrays_path),
                  metadata=bc.base.file_binding(metadata_path),
                  status='measured_not_independently_audited_not_classifier_admitted')
    bc.base.write_new(receipt, record)
    return record


def main():
    assert bc.base.digest(RC/'run_native30_bc_v2.py') == '822f51f3e961bb4edf16389283a91b898393084f34d1a87da0956d7a92cffe0d'
    source = RD/'native30_fhsc_v1'
    commit = bc.base.read_json(source/'COMMIT.json')
    rows = [(uid, source/'items'/(uid+'.json'), sha) for uid, sha in sorted(commit['items'].items())]
    assert len(rows) == 498
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for folder in ('items', 'metadata', 'arrays'):
        (OUT/folder).mkdir(exist_ok=True)
    contract = dict(rows=498, source_commit=bc.base.file_binding(source/'COMMIT.json'),
                    driver=bc.base.file_binding(Path(__file__).resolve()),
                    producer=bc.base.file_binding(RC/'run_native30_bc_v2.py'),
                    numerical_pins=bc.PINS, view=bc.VIEW, classifier_fits=0)
    path = OUT/'contract.json'
    if path.exists():
        assert bc.base.read_json(path) == contract
    else:
        bc.base.write_new(path, contract)
    with ProcessPoolExecutor(max_workers=2) as pool:
        finished = []
        for r in pool.map(one, rows):
            finished.append(r)
            if len(finished)%10 == 0:
                print(f'BC measured {len(finished)}/498', flush=True)
    bc.base.write_new(OUT/'COMMIT.json', dict(status='completed_measurements_not_independent_audit',
                     rows=498, scientific_nulls=sum(r['features'][bc.FEATURE] is None for r in finished),
                     items={r['id']:bc.base.file_binding(OUT/'items'/(r['id']+'.json')) for r in finished},
                     contract=bc.base.file_binding(path), classifier_fits=0))


if __name__ == '__main__':
    main()
