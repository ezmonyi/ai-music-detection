"""Bind processed Ishizaka inputs to the accepted source-specific CC0 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path, pin):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin
    return json.loads(raw)


def main(objects, originals, out):
    assert not out.exists()
    objects = load(objects, 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92')
    source = load(originals, 'c5db937a1d1e308f533f2e8f2378febddf7db0a3d7197bc3f4e54eb3d649f7fe')
    source = {r['id']:r for r in source}
    assert len(source) == 39
    rows = []; ids = set()
    for obj in objects:
        selected = {m['id'] for m in obj['memberships']}
        if not selected.intersection(source): continue
        assert selected.issubset(source), 'Mixed source object requires explicit review'
        assert all(m['source_group'] == 'human_musicnet' for m in obj['memberships'])
        assert not obj['public_paths']
        for uid in selected:
            assert source[uid]['recording_license'] == 'CC0-1.0'
            assert source[uid]['performer'] == 'Kimiko Ishizaka'
        ids.update(selected)
        rows.append(dict(**obj, original_records=[source[i] for i in sorted(selected)],
            recording_license='CC0-1.0',
            modification='Historical cropped/standardized audio_path input; original recording is separately published'))
    assert len(rows) == 78 and ids == set(source)
    out.mkdir()
    raw = (json.dumps(rows, indent=2)+'\n').encode()
    (out/'objects.json').write_bytes(raw)
    summary = dict(objects=78, original_ids=39, bytes=sum(r['bytes'] for r in rows),
        objects_sha256=hashlib.sha256(raw).hexdigest(),
        source_manifest_sha256='c5db937a1d1e308f533f2e8f2378febddf7db0a3d7197bc3f4e54eb3d649f7fe',
        source_specific_derivative_candidate=True, publication_verified=False,
        whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects', type=Path, required=True)
    p.add_argument('--originals', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.objects, a.originals, a.out)
