"""Preserve mirror-to-source IDs without mislabelling them as artist metadata."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(source, benchmark, out):
    rows=json.loads(source.read_text())
    official={r['track_id'] for r in csv.DictReader(benchmark.open())}
    assert len(rows)==len({r['id'] for r in rows})==239
    assert len({r['medleydb_id'] for r in rows})==239
    assert len(official)==235
    out.mkdir(exist_ok=False)
    with (out/'crosswalk.csv').open('x',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['experiment_id','mirror_id','source_id_recorded_as_medleydb_id','seen_in_official_htdemucs4'])
        for r in sorted(rows,key=lambda r:r['id']):
            writer.writerow(['human_moisesdb_'+r['id'],r['id'],r['medleydb_id'],int(r['medleydb_id'] in official)])
    matched=sum(r['medleydb_id'] in official for r in rows)
    assert matched==235
    (out/'README.md').write_text('''# MoisesDB mirror identity crosswalk

The historical mirror uses new UUIDs. None of its 239 selected IDs directly
matches the official htdemucs4 benchmark IDs. Its confusingly named medleydb_id
field provides 239 unique source IDs; 235 match that official benchmark's 235
IDs. Four are absent from that benchmark; absence is not evidence of invalid
audio. This is not a cross-dataset link to MedleyDB inferred from the field name.

Official benchmark is pinned to wearemusicai/moisesdb revision
3162378eb0653d9f9831a6051110822ab067b8b6, benchmark/htdemucs4.csv.
The mirror is seungheondoh/cmd-moisesdb-metadata at revision
353a59cfd90514ea71e35480b5ab8ffb57d14598.

This link improves source retrieval but does not recover artist names, song
titles, original audio equivalence or permission to republish. Keep experiment
IDs and historical grouping unchanged. Do not silently substitute source IDs
into already-run evaluations. Artist attribution remains unresolved.
''')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'COMMIT.json').write_text(json.dumps(dict(rows=239,benchmark_matched=matched,
        source_sha256=sha(source),benchmark_sha256=sha(benchmark),
        products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()},
        artist_attribution_recovered=False,audio_uploaded=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','benchmark','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();main(a.source,a.benchmark,a.out)
