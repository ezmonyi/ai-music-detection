"""Recover selected DEAM credits from official metadata, without inferring licenses."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

PIN = '3d1d5ab42e852803a770f2cfbd685c6b464e1ebe76f26b0e10954597902b90f8'


def recover(archive, catalogue, out):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == PIN
    index = {}
    specs = [('2013', 'song_id', 'Artist', 'Song title', None),
             ('2014', 'Id', 'Artist', 'Track', 'Album'),
             ('2015', 'id', 'artist', 'title', 'album')]
    with zipfile.ZipFile(archive) as z:
        for year, identity, artist, title, album in specs:
            member = f'metadata/metadata_{year}.csv'
            text = z.read(member).decode('utf-8-sig')
            for row in csv.DictReader(io.StringIO(text)):
                key = str(int(row[identity].strip()))
                record = dict(artist=row[artist].strip(), title=row[title].strip(),
                              album=row[album].strip() if album else None, source_member=member)
                index.setdefault(key, []).append(record)
    selected = [r for r in csv.DictReader(catalogue.open()) if r['source_group'] == 'human_deam']
    assert len(selected) == 300
    records = []
    for row in selected:
        match = re.search(r'#song_id=(\d+)$', row['source_locator'])
        assert match, row['id']
        key = match.group(1)
        credits = index.get(key, [])
        records.append(dict(id=row['id'], source_song_id=key,
            original_sha256=row['source_sha256'], original_bytes=int(row['source_bytes']),
            attribution_variants=credits, metadata_found=bool(credits),
            per_recording_license_verified=False, audio_publication_verified=False))
    out.mkdir(exist_ok=False)
    data = (json.dumps(records, ensure_ascii=False, indent=2)+'\n').encode()
    (out/'records.json').write_bytes(data)
    summary = dict(records=300, matched=sum(r['metadata_found'] for r in records),
        missing_ids=[r['id'] for r in records if not r['metadata_found']],
        multiple_metadata_rows=sum(len(r['attribution_variants'])>1 for r in records),
        empty_artist_or_title=sum(any(not c['artist'] or not c['title'] for c in r['attribution_variants']) for r in records),
        official_metadata_sha256=PIN, source_catalogue_sha256=hashlib.sha256(catalogue.read_bytes()).hexdigest(),
        records_sha256=hashlib.sha256(data).hexdigest(),
        source_url='https://cvml.unige.ch/databases/DEAM/metadata.zip',
        audio_uploaded=False, whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); recover(args.archive, args.catalogue, args.out)
