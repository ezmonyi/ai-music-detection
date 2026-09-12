"""Run the original exact60 F/H/M descriptors on verified YuE2 native crops."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0,str(RC/'code'))
import run_mureka60_fhm_v2 as reference


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    original,bindings=reference.code_and_reference()
    source=ROOT/'native60_inputs_v1'
    audit_path=ROOT/'native60_independent_audit_v1.json'
    assert sha(audit_path)=='df066741887939ac1a9562b831f4203ba0087762d9e4c6888c57b21fb8b558ee'
    audit=json.loads(audit_path.read_text())
    assert sha(source/'COMMIT.json')==audit['source_commit_sha256']
    commit=json.loads((source/'COMMIT.json').read_text())
    assert sha(source/'metadata.json')==commit['products']['metadata.json']['sha256']
    rows=json.loads((source/'metadata.json').read_text())
    assert len(rows)==276
    out=ROOT/'native60_fhm_v1'
    out.mkdir(exist_ok=False)
    mapped=[]
    for row in rows:
        path=Path(row['input_path'])
        assert sha(path)==row['input_sha256']
        mapped.append(dict(id=row['id'],label='ai',source_group='YuE2',role=row['role'],
             group_id=row['group_id'],audio_path=str(path),source_audio_sha256=row['input_sha256'],
             audio_offset_s=0,native_sample_rate_hz=48000,duration=60))
    metadata=out/'metadata.csv'
    with metadata.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(mapped[0]));writer.writeheader();writer.writerows(mapped)
    save(out/'LAUNCH.json',dict(input_commit_sha256=sha(source/'COMMIT.json'),
         input_audit_sha256=sha(audit_path),reference_bindings=bindings,
         driver_sha256=sha(Path(__file__)),classifier_fits=0,M_role='diagnostic_not_automatically_admitted_predictor'))
    features=out/'features'
    subprocess.run([sys.executable,str(RC/'code/extract_features.py'),'--metadata',str(metadata),
                    '--output',str(features),'--duration','60','--workers','4'],check=True)
    contract=json.loads((features/'contract.json').read_text())
    for key in ['duration','preflight_only','input_config','F_config','feature_names','code_sha256','runtime']:
        assert contract[key]==original[key],key
    records=list(csv.DictReader((features/'features.csv').open()))
    assert [r['id'] for r in records]==[r['id'] for r in mapped]
    names=[name for group in contract['feature_names'].values() for name in group]
    for row,record in zip(mapped,records):
        item=json.loads((features/'items'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text())
        assert item['extraction_status']=='ok',row['id']
        assert item['analysis_frames']==960000 and item['analysis_sr']==16000
        assert item['role']==row['role'] and item['group_id']==row['group_id']
        assert item['source_audio_sha256']==row['source_audio_sha256']
        for key in names:
            value=item.get(key)
            assert record[key]==('' if value is None else str(value)),(row['id'],key)
        assert sha(Path(row['audio_path']))==row['source_audio_sha256']
    products={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
              for p in out.rglob('*') if p.is_file()}
    save(out/'COMMIT.json',dict(status='completed_native60_fhm_measurement',rows=276,products=products,classifier_fits=0))
    print('Native60 FHM measurement committed',flush=True)


if __name__=='__main__':main()
