"""Bind 200 early FMA originals to preserved track-level attribution and licenses."""
import csv
import json
from pathlib import Path
from collections import Counter
from prepare_fma_original_publication_v1 import LICENSES,sha,WORK

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')


def main():
    audit=ROOT/'early_original_audio_audit_v1'
    commit=json.loads((audit/'COMMIT.json').read_text())
    assert sha(audit/'originals.json')==commit['manifest_sha256']
    chosen={int(Path(r['relative_source_path']).stem):r for r in json.loads((audit/'originals.json').read_text())
            if r['memberships'][0]['source']=='fma_medium'}
    assert len(chosen)==200
    tracks=WORK/'artifacts/fma_metadata_source/fma_metadata/tracks.csv'
    assert sha(tracks)=='f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b'
    rows=[]
    with tracks.open() as stream:
        reader=csv.reader(stream);headers=list(zip(next(reader),next(reader)));next(reader)
        for values in reader:
            tid=int(values[0])
            if tid not in chosen:continue
            source=chosen[tid];entry=dict(zip(headers,values));name=entry[('track','license')]
            url='https://creativecommons.org/licenses/'+LICENSES[name] if name in LICENSES else None
            if name=='CC0 1.0 Universal':url='https://creativecommons.org/publicdomain/zero/1.0/'
            path=WORK/source['relative_source_path']
            assert path.stat().st_size==source['bytes'] and sha(path)==source['sha256']
            rows.append(dict(source,fma_track_id=tid,artist=entry[('artist','name')],
                title=entry[('track','title')],artist_group_id=entry[('artist','id')],
                license_as_recorded=name,license_url=url,
                status='explicit_license_unchanged_original_candidate' if url else 'hold_ambiguous_license',
                transformation='none; preserved FMA medium MP3',derivative_publication_authorized=False))
    assert len(rows)==200
    out=ROOT/'early_fma_publication_plan_v1';out.mkdir(exist_ok=False)
    (out/'records.json').write_text(json.dumps(rows,indent=2))
    summary=dict(status='200_source_files_attributed_not_uploaded',files=200,
        counts=dict(Counter(r['status'] for r in rows)),records_sha256=sha(out/'records.json'),
        tracks_sha256=sha(tracks),source_audit_commit_sha256=sha(audit/'COMMIT.json'),
        audio_uploaded=False)
    (out/'COMMIT.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
