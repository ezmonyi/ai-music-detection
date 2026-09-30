"""Replay fixed Drive development samples; this is not an accuracy benchmark.

The --download switch reads only known TAR members after current archive
metadata matches the manifest. Set MUSIC_DETECTOR_DRIVE_TOKEN securely in the
process environment; never put a token in a command line or committed file.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import urllib.request

import numpy as np
import scipy
import soundfile

from .analysis import analyze, DEPLOYMENT_BUNDLE
from .calibration import predict_deployment
from .drive import DriveRangeReader, TarMember, download_member, RangeReadError
from .scoring import FAMILIES

REPOSITORY = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_archive_reader(archive, *, token=None):
    reader = DriveRangeReader(archive['file_id'], archive['size'], token=token)
    request = urllib.request.Request(
        f"https://www.googleapis.com/drive/v3/files/{archive['file_id']}?fields=id,size,md5Checksum,trashed",
        headers={'Authorization': 'Bearer ' + reader.token()})
    try:
        with reader.opener.open(request, timeout=40) as response:
            metadata = json.loads(response.read(65537))
    except Exception:
        raise RangeReadError('Cannot verify current archive metadata; check Drive access without exposing your token.') from None
    if (metadata.get('trashed') or int(metadata.get('size', -1)) != archive['size'] or
            metadata.get('md5Checksum') != archive['md5_at_metadata_read']):
        raise RangeReadError('Current Drive archive differs from manifest or is in Trash.')
    return reader


def replay(manifest_path, reference_path, audio_dir, *, download=False, token=None):
    manifest = json.loads(Path(manifest_path).read_text())
    references = {row['id']: row['features'] for row in json.loads(Path(reference_path).read_text())}
    bundle = json.loads(DEPLOYMENT_BUNDLE.read_text())
    readers, records = {}, []
    combinations = [list(combo) for count in (1, 2, 3)
                    for combo in itertools.combinations(('F', 'H', 'SC'), count)]
    for sample in manifest['samples']:
        path = Path(audio_dir) / (sample['id'] + Path(sample['member_name']).suffix)
        if not path.exists() and download:
            archive_key = sample['archive']
            if archive_key not in readers:
                readers[archive_key] = checked_archive_reader(manifest['archives'][archive_key], token=token)
            download_member(readers[archive_key], TarMember(sample['member_name'], sample['member_offset'],
                            sample['member_size']), path, sample['member_sha256'])
        if not path.is_file() or path.is_symlink() or digest(path) != sample['member_sha256']:
            raise ValueError(f"Missing or mismatched sample {sample['id']}; no analysis performed for it")
        result = analyze(path, ['F', 'H', 'SC'], approved_region=
                         sample['verification'].get('approved_region_for_local_replay'))
        expected, actual = references[sample['id']], result['feature_values']
        if set(expected) != set(actual):
            raise ValueError('Reference columns differ from actual extraction')
        differences = []
        for name, value in expected.items():
            observed = actual[name]
            equivalent = observed is None if value is None else observed is not None and bool(
                np.isclose(value, observed, rtol=1e-7, atol=1e-8))
            if not equivalent:
                differences.append(dict(feature=name, expected=value, observed=observed,
                                        abs_error=abs(value-observed) if value is not None and observed is not None else None))
        scores = []
        for families in combinations:
            names = [name for family in families for name in FAMILIES[family]]
            a = predict_deployment(bundle, {name: actual[name] for name in names}, families)
            e = predict_deployment(bundle, {name: expected[name] for name in names}, families)
            scores.append(dict(combination='+'.join(families), expected_raw=e['raw_score'],
                               observed_raw=a['raw_score'], abs_raw_error=abs(e['raw_score']-a['raw_score']),
                               expected_probability=e['probability'], observed_probability=a['probability'],
                               abs_probability_error=abs(e['probability']-a['probability']),
                               expected_decision=e['decision'], observed_decision=a['decision'],
                               decision_equal=e['decision'] == a['decision']))
        records.append(dict(id=sample['id'], source=sample['source_group'], label=sample['label'],
                            source_sha256=sample['member_sha256'],
                            waveform_equal=result['provenance']['output_waveform_float32_sha256'] ==
                            sample['canonical']['waveform_float32_sha256'],
                            feature_count=len(actual), strict_features_equal=not differences,
                            differences=differences, measurements=actual, scores=scores,
                            provenance=result['provenance']))
        print(f"Verified {sample['id']}: {len(differences)} strict metric differences", flush=True)
    scores = [item for row in records for item in row['scores']]
    return dict(scope='fixed_development_sample_infrastructure_replay_not_accuracy_estimation',
                manifest_sha256=digest(manifest_path), reference_sha256=digest(reference_path),
                bundle_sha256=digest(DEPLOYMENT_BUNDLE),
                runtime=dict(python=platform.python_version(), system=platform.system(),
                             numpy=np.__version__, scipy=scipy.__version__, soundfile=soundfile.__version__,
                             libsndfile=soundfile.__libsndfile_version__),
                tolerance=dict(rtol=1e-7, atol=1e-8), results=records,
                summary=dict(samples=len(records), source_hash_matches=len(records),
                             waveform_matches=sum(row['waveform_equal'] for row in records),
                             strict_feature_passes=sum(row['strict_features_equal'] for row in records),
                             combination_checks=len(scores),
                             decision_changes=sum(not item['decision_equal'] for item in scores),
                             maximum_raw_score_error=max(item['abs_raw_error'] for item in scores),
                             maximum_probability_error=max(item['abs_probability_error'] for item in scores)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPOSITORY / 'manifests/drive_sources.json')
    parser.add_argument('--reference', type=Path, default=REPOSITORY / 'tests/fixtures/sample_reference_features.json')
    parser.add_argument('--audio-dir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--rclone-remote', help='Reuse only this already configured local Drive OAuth remote')
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error('Refusing to overwrite an existing replay report; use a new output path.')
    from .auth import drive_token
    token = drive_token(args.rclone_remote) if args.download else None
    report = replay(args.manifest, args.reference, args.audio_dir, download=args.download,
                    token=(lambda: token) if token else None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
