"""Metadata-only local artifact inventory. Does not hash/decode audio."""
import datetime
import json
from pathlib import Path


def main():
    root = Path('/Users/yi/Documents/code/music/artifacts')
    target = root / 'audio_phenomena_expansion_20260907/audit/closeout_local_inventory_20260908_v1'
    target.mkdir(exist_ok=False)
    totals = {}
    errors = []
    with (target / 'files.jsonl').open('x') as out:
        for path in sorted(root.rglob('*')):
            if target in path.parents or path == target:
                continue
            try:
                if path.is_symlink():
                    row = {'path': str(path), 'kind': 'symlink', 'target': str(path.readlink())}
                elif path.is_file():
                    st = path.stat()
                    row = {'path': str(path), 'kind': 'file', 'bytes': st.st_size,
                           'mtime_ns': st.st_mtime_ns}
                    group = path.relative_to(root).parts[0]
                    value = totals.setdefault(group, {'files': 0, 'logical_bytes': 0})
                    value['files'] += 1
                    value['logical_bytes'] += st.st_size
                else:
                    continue
                out.write(json.dumps(row, sort_keys=True) + '\n')
            except OSError as exc:
                errors.append({'path': str(path), 'error': str(exc)})
    report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': str(root), 'metadata_only': True, 'remote_inventory': False,
              'atomic_snapshot': False, 'unique_music_count': None,
              'self_output_excluded': str(target), 'groups': totals, 'errors': errors}
    (target / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'groups': len(totals),
                      'files': sum(x['files'] for x in totals.values()), 'errors': len(errors)}))


if __name__ == '__main__':
    main()
