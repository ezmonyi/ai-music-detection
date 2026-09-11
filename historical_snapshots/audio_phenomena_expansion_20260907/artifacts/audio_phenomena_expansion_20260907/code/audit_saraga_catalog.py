#!/usr/bin/env python3
"""Audit a pinned metadata-only Saraga snapshot; never admit/download audio."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

COMMIT = 'bd103295689ad11a5ed19038b7469e91b5c3c124'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    commit = json.loads((root/'github_commit.json').read_text())
    tree = json.loads((root/'github_tree.json').read_text())
    zenodo = json.loads((root/'zenodo_record_4301737.json').read_text())
    require(commit['sha'] == COMMIT and tree['sha'] == commit['commit']['tree']['sha']
            and tree['truncated'] is False, 'Wrong/incomplete repository snapshot')
    expected = {x['path']:x for x in tree['tree'] if x['type'] == 'blob'}
    blobs = {}
    with tarfile.open(root/'repository_metadata.tar.gz', 'r:gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            require(parts[0] == 'saraga-'+COMMIT and '..' not in parts, 'Unexpected archive path')
            if member.isdir():
                continue
            require(member.isfile(), 'Links/nonregular archive entries forbidden')
            name = '/'.join(parts[1:])
            require(name in expected and name not in blobs, 'Unexpected/duplicate file')
            require(member.size == expected[name]['size'] and member.size <= 5_000_000,
                    'Unexpected repository file size')
            data = archive.extractfile(member).read()
            git_digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            require(git_digest == expected[name]['sha'], 'Git blob mismatch: '+name)
            blobs[name] = data
    require(set(blobs) == set(expected), 'Incomplete pinned repository archive')
    require(blobs['LICENSE.md'] == (root/'LICENSE.md').read_bytes(), 'License snapshot mismatch')
    rows = []
    for path, data in sorted(blobs.items()):
        if not path.startswith('dataset/') or not path.endswith('.json'):
            continue
        meta = json.loads(data)
        base = path[:-5]
        audio_md5 = blobs.get(base+'.mp3.md5')
        md5 = audio_md5.decode().strip().split()[0] if audio_md5 else None
        require(md5 is None or re.fullmatch('[a-fA-F0-9]{32}', md5), 'Invalid audio MD5')
        lead = [a['artist']['mbid'] for a in meta.get('artists', []) if a.get('lead')]
        album = [a['mbid'] for a in meta.get('album_artists', [])]
        related = sorted(p[len(base)+1:] for p in blobs if p.startswith(base+'.'))
        rows.append(dict(mbid=meta['mbid'], tradition=path.split('/')[1], title=meta['title'],
            metadata_path=path, metadata_sha256=hashlib.sha256(data).hexdigest(),
            advertised_length_ms=meta.get('length'), audio_md5=md5,
            lead_artist_mbids=sorted(set(lead)), album_artist_mbids=sorted(set(album)),
            release_or_concert_mbids=sorted({r['mbid'] for key in ('concert', 'release') for r in meta.get(key, [])}),
            speech_title_review_flag=bool(re.search(r'speech|introduction|introductory', meta['title'], re.I)),
            companion_files=related, role='catalog_only_not_selected', physical_audio_verified=False))
    require(len({r['mbid'] for r in rows}) == len(rows), 'Duplicate recording MBID')
    summaries = []
    for tradition in sorted({r['tradition'] for r in rows}):
        part = [r for r in rows if r['tradition'] == tradition]
        summaries.append(dict(tradition=tradition, metadata_rows=len(part),
            audio_md5_rows=sum(r['audio_md5'] is not None for r in part),
            advertised_at_least60=sum(isinstance(r['advertised_length_ms'], (int,float))
                                     and r['advertised_length_ms'] >= 60000 for r in part),
            speech_title_review_rows=sum(r['speech_title_review_flag'] for r in part),
            album_artist_ids=len({a for r in part for a in r['album_artist_mbids']}),
            lead_artist_ids=len({a for r in part for a in r['lead_artist_mbids']}),
            annotation_or_track_counts=dict(Counter(s for r in part for s in r['companion_files']))))
    return dict(status='passed_metadata_only', repository_commit=COMMIT,
        verified_git_blob_files=len(blobs), total_repository_bytes=sum(map(len,blobs.values())),
        source_inputs_sha256={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()},
        code_sha256=sha(Path(__file__)), zenodo_record=zenodo['id'],
        archive_version=zenodo['metadata']['version'], zenodo_license=zenodo['metadata']['license'],
        repository_audio_license='CC BY-NC 4.0',
        license_discrepancy='Zenodo metadata says CC BY-NC-SA 4.0; retain both, no redistribution or extra agreement',
        audio_archive_files=[{k:f[k] for k in ('key','size','checksum','links')} for f in zenodo['files']],
        rows=rows, tradition_summary=summaries, classifier_admission=False,
        audio_downloaded=False, musical_content_or_physical_duration_not_verified=True,
        no_overlap_claim='Existing nine HF Hindustani clips lack performer identity; perceptual overlap unresolved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.catalog_dir)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
