"""Full YuE2 BC repeatability audit; same-code replay, not independent method validation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
RD=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/native30_bc_v1')
sys.path.insert(0,str(RC))
import run_native30_bc_v2 as bc


def one(entry):
    assert bc.base.file_binding(entry['path'])==entry
    record=bc.base.read_json(entry['path'])
    for name in ('input','metadata','arrays','source_receipt'):
        assert bc.base.file_binding(record[name]['path'])==record[name]
    wave,sr=sf.read(record['input']['path'],dtype='float64',always_2d=True)
    assert sr==44100
    analysis,view=bc.analysis_view(wave)
    assert view==record['analysis_view']
    _,module,scalar=bc.numerical_modules()
    replay=module.extract(analysis,16000)
    replay['metadata']['construction_status']=view['construction_status']
    expected=bc.base.read_json(record['metadata']['path'])
    assert replay['metadata']==expected
    assert scalar.reduce_metadata(expected,crop=bc.CROP)==record['scalar']
    with np.load(record['arrays']['path'],allow_pickle=False) as arrays:
        assert set(arrays.files)==set(replay['arrays'])
        for name in arrays.files:
            assert np.array_equal(arrays[name],replay['arrays'][name],equal_nan=True),name
    assert record['features'][bc.FEATURE]==record['scalar']['median_squared_bicoherence']
    return record['id']


def main():
    commit=bc.base.read_json(RD/'COMMIT.json')
    assert commit['rows']==len(commit['items'])==498
    assert bc.base.file_binding(commit['contract']['path'])==commit['contract']
    with ProcessPoolExecutor(max_workers=2) as pool:
        done=[]
        for uid in pool.map(one,commit['items'].values()):
            done.append(uid)
            if len(done)%25==0:
                print(f'BC replay verified {len(done)}/498',flush=True)
    assert len(set(done))==498
    out=RD.parent/'native30_bc_repeatability_audit_v1.json'
    bc.base.write_new(out,dict(status='passed_full_same_code_audio_to_arrays_and_scalar_replay',
                      rows=498,measurement_commit=bc.base.file_binding(RD/'COMMIT.json'),
                      auditor=bc.base.file_binding(Path(__file__).resolve()),
                      independent_implementation=False,classifier_fits=0))
    print('Full BC repeatability audit passed.',flush=True)


if __name__=='__main__':
    main()
