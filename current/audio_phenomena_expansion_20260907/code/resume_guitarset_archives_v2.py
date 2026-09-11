"""Resume a stopped GuitarSet acquisition in a new directory; preserve history."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

EXPECTED = {'audio_mono-mic.zip': (656927981, '275966d6610ac34999b58426beb119c3'),
            'annotation.zip': (39132574, 'b39b78e63d3446f2e54ddb7a54df9b10')}
METADATA_SHA = '1ebc52f572801b6c3834bf8007217274d6036de7ae16a00d0f29fd505876a8e8'


def sha(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def resume_file(url, partial, expected_size, expected_md5, log_dir, max_attempts=12):
    """Each bounded request resumes its own partial; final official hash required."""
    log_dir.mkdir(exist_ok=False)
    attempts = []
    for number in range(max_attempts):
        before = partial.stat().st_size if partial.exists() else 0
        if before > expected_size:
            raise ValueError('oversized partial')
        if before == expected_size:
            if sha(partial, 'md5') != expected_md5:
                raise ValueError('complete bytes have wrong MD5; preserved, not overwritten')
            return attempts
        command = ['curl', '--fail', '--location', '--silent', '--show-error',
                   '--continue-at', '-', '--connect-timeout', '20', '--max-time', '600',
                   '--dump-header', str(log_dir / f'{number:02d}.headers.txt'),
                   '--output', str(partial), url]
        start = time.monotonic()
        run = subprocess.run(command, capture_output=True, text=True, timeout=660)
        after = partial.stat().st_size if partial.exists() else 0
        record = {'attempt': number, 'command': command, 'returncode': run.returncode,
                  'bytes_before': before, 'bytes_after': after,
                  'elapsed_seconds': time.monotonic()-start, 'stdout': run.stdout, 'stderr': run.stderr}
        write_json(log_dir / f'{number:02d}.command.json', record)
        attempts.append(record)
        print(json.dumps({'archive': partial.name, 'attempt': number, 'bytes_before': before,
                          'bytes_after': after, 'returncode': run.returncode}), flush=True)
        if after < before or after > expected_size:
            raise ValueError('unexpected partial-size change')
        if after == expected_size:
            if sha(partial, 'md5') != expected_md5:
                raise ValueError('final official MD5 mismatch')
            return attempts
        # Timeouts and transient transport/HTTP errors keep the accumulated
        # prefix. Curl itself rejects unsupported/mismatched resume responses.
        if run.returncode not in (18, 22, 28, 35, 52, 55, 56):
            raise RuntimeError('non-retryable/incomplete download, returncode=' + str(run.returncode))
        if number + 1 < max_attempts:
            time.sleep(60 if run.returncode == 22 else 10)
    raise RuntimeError('bounded resume attempts exhausted; partial preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-root', required=True, type=Path)
    parser.add_argument('--previous-partial-sha256', required=True)
    parser.add_argument('--previous-partial-bytes', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    previous = args.previous_root.resolve(strict=True)
    metadata_path = previous / 'upstream_metadata.json'
    failure_path = previous / 'EXECUTION_FAILURE.json'
    old_partial = previous / 'audio_mono-mic.zip.part'
    if (previous / 'COMMIT.json').exists() or not failure_path.is_file():
        raise ValueError('stopped uncommitted predecessor with recorded failure required')
    if sha(metadata_path) != METADATA_SHA:
        raise ValueError('metadata identity')
    if old_partial.is_symlink() or old_partial.stat().st_size != args.previous_partial_bytes or sha(old_partial) != args.previous_partial_sha256:
        raise ValueError('predecessor partial identity')
    metadata = json.loads(metadata_path.read_text())
    if metadata['doi'] != '10.5281/zenodo.3371780' or metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('source license/DOI')
    entries = {entry['key']: entry for entry in metadata['files']}
    for name, (size, md5) in EXPECTED.items():
        if entries[name]['size'] != size or entries[name]['checksum'] != 'md5:' + md5:
            raise ValueError('official archive identity')
        if entries[name]['links']['self'] != f'https://zenodo.org/api/records/3371780/files/{name}/content':
            raise ValueError('archive URL')
    root = args.output.resolve()
    if root.is_relative_to(previous):
        raise ValueError('separate output required')
    root.mkdir(exist_ok=False)
    try:
        shutil.copyfile(metadata_path, root / 'upstream_metadata.json')
        shutil.copyfile(failure_path, root / 'predecessor_failure.json')
        write_json(root / 'resume_lineage.json', {'previous_root': str(previous),
            'previous_failure_sha256': sha(failure_path), 'partial_name': old_partial.name,
            'partial_bytes': args.previous_partial_bytes, 'partial_sha256': args.previous_partial_sha256,
            'predecessor_retained_unchanged': True})
        partial = root / old_partial.name
        shutil.copyfile(old_partial, partial)
        if sha(partial) != args.previous_partial_sha256 or sha(old_partial) != args.previous_partial_sha256:
            raise ValueError('partial changed during copy')
        archives = []
        for name, (size, md5) in EXPECTED.items():
            part = root / (name + '.part')
            resume_file(entries[name]['links']['self'], part, size, md5, root / (name + '.attempts'))
            part.rename(root / name)
            archives.append({'name': name, 'bytes': size, 'md5': md5, 'sha256': sha(root/name)})
        if sha(old_partial) != args.previous_partial_sha256 or sha(metadata_path) != METADATA_SHA:
            raise ValueError('predecessor changed during acquisition')
        write_json(root / 'acquisition_receipt.json', {'status': 'archives_verified_not_decoded',
            'metadata_sha256': METADATA_SHA, 'doi': metadata['doi'], 'license': 'CC-BY-4.0',
            'archives': archives, 'code_sha256': sha(Path(__file__)), 'zip_crc_checked': False,
            'audio_decoded': False, 'bc_measured': False, 'classifier_fits': 0, 'external_gate_passed': False})
        products = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in root.rglob('*') if p.is_file()}
        write_json(root / 'COMMIT.json', {'status': 'committed', 'kind': 'archive_acquisition_only', 'products': products})
        print(json.dumps({'status': 'committed', 'commit_sha256': sha(root/'COMMIT.json')}), flush=True)
    except BaseException as exc:
        write_json(root / 'EXECUTION_FAILURE.json', {'error': str(exc), 'type': type(exc).__name__, 'commit_published': False})
        raise


if __name__ == '__main__':
    main()
