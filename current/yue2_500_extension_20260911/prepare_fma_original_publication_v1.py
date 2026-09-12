"""Bind the 500 tested FMA IDs to unmodified source MP3s and track licenses."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

WORK=Path('/mnt/nfs-code/users/yi/macbook_reproducibility_20260909/workspace')
ROOT=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
LICENSES={
 'Attribution-NonCommercial-ShareAlike 3.0 International':'by-nc-sa/3.0/',
 'Attribution-Noncommercial-Share Alike 3.0 United States':'by-nc-sa/3.0/us/',
 'Attribution-Noncommercial 3.0 United States':'by-nc/3.0/us/',
 'Attribution-Noncommercial-No Derivative Works 3.0 United States':'by-nc-nd/3.0/us/',
 'Attribution-NonCommercial-NoDerivatives (aka Music Sharing) 3.0 International':'by-nc-nd/3.0/',
 'Attribution-Share Alike 3.0 United States':'by-sa/3.0/us/',
 'Attribution 3.0 United States':'by/3.0/us/',
 'Attribution 3.0 International':'by/3.0/',
 'Attribution-ShareAlike 3.0 International':'by-sa/3.0/',
 'Attribution-NonCommercial 3.0 International':'by-nc/3.0/',
 'Attribution-NoDerivatives 3.0 International':'by-nd/3.0/',
 'Creative Commons Attribution-NonCommercial-NoDerivatives 4.0':'by-nc-nd/4.0/',
 'Attribution-NoDerivatives 4.0 International':'by-nd/4.0/',
}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    tracks=WORK/'artifacts/fma_metadata_source/fma_metadata/tracks.csv'
    assert sha(tracks)=='f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b'
    original=Path('/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901/manifest.jsonl')
    chosen={int(Path(r['path']).stem):r for r in map(json.loads,original.open()) if r['source']=='fma_medium'}
    metadata=Path('/mnt/nfs-code/users/yi/source_diversity_expansion_20260905/manifests/metadata_10s.csv')
    assert sha(metadata)=='a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4'
    inputs={r['id']:r for r in csv.DictReader(metadata.open()) if r['source_group']=='FMA'}
    licenses={}
    with tracks.open() as f:
        reader=csv.reader(f);headers=list(zip(next(reader),next(reader)));next(reader)
        for row in reader:
            tid=int(row[0])
            if tid in chosen:licenses[tid]=dict(zip(headers,row))
    assert len(chosen)==len(licenses)==len(inputs)==500
    rows=[]
    for tid,legacy in sorted(chosen.items()):
        entry=licenses[tid];uid='human_fma_medium_'+legacy['id'];item=inputs[uid]
        assert entry[('artist','id')]==legacy['artist_id']
        path=WORK/legacy['path'];digest=sha(path)
        assert digest==item['raw_sha256'],uid
        license_name=entry[('track','license')]
        url='https://creativecommons.org/licenses/'+LICENSES[license_name] if license_name in LICENSES else None
        if license_name=='CC0 1.0 Universal':url='https://creativecommons.org/publicdomain/zero/1.0/'
        rows.append(dict(id=uid,fma_track_id=tid,artist=entry[('artist','name')],title=entry[('track','title')],
            group_id=item['group_id'],source_archive_member=legacy['path'],local_path=str(path),
            bytes=path.stat().st_size,sha256=digest,license_as_recorded=license_name,license_url=url,
            status='explicit_version_original_candidate' if url else 'hold_unmapped_or_ambiguous_license',
            source='https://github.com/mdeff/fma',transformations='none; unmodified FMA medium source MP3',
            derivative_publication_authorized=False))
    out=ROOT/'fma_original_publication_plan_v1.json'
    with out.open('x') as f:json.dump(dict(status='source_bytes_verified_not_uploaded',rows=rows,
        tracks_sha256=sha(tracks),legacy_manifest_sha256=sha(original),input_metadata_sha256=sha(metadata),
        counts=dict(Counter(r['status'] for r in rows)),audio_files_verified=500),f,indent=2)
    print(json.dumps(dict(audio_files_verified=500,counts=dict(Counter(r['status'] for r in rows)),plan_sha256=sha(out))),flush=True)


if __name__=='__main__':main()
