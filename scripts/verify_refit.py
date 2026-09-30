"""Quantify a fresh Drive-input refit's numerical and decision differences."""
import argparse
import json
from pathlib import Path
import numpy as np
from music_detector.calibration import ridge_scores, probabilities, sha_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--refit', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original, refit, protocol, features = [json.loads(p.read_text()) for p in
                                          (args.original, args.refit, args.protocol, args.features)]
    if args.output.exists():
        parser.error('Refusing to replace an existing verification receipt')
    if original['models'].keys() != refit['models'].keys() or original['protocol_sha256'] != refit['protocol_sha256']:
        raise ValueError('Model set or protocol changed')
    index = {row['id']: row for row in features}
    ids = protocol['split']['test']['ids']
    records = []
    for combination in original['models']:
        old, new = original['models'][combination]['payload'], refit['models'][combination]['payload']
        columns = old['ridge_model']['columns']
        if columns != new['ridge_model']['columns']:
            raise ValueError('Feature schema changed')
        matrix = np.asarray([[index[uid][name] for name in columns] for uid in ids], dtype=float)
        raw_old, raw_new = (ridge_scores(matrix, model['ridge_model']) for model in (old, new))
        p_old, p_new = (probabilities(raw, model['calibration']) for raw, model in ((raw_old, old), (raw_new, new)))
        records.append(dict(combination=combination, test_rows=len(ids),
                            max_raw_error=float(np.max(np.abs(raw_old-raw_new))),
                            max_probability_error=float(np.max(np.abs(p_old-p_new))),
                            decisions_changed=int(np.count_nonzero((p_old >= .5) != (p_new >= .5)))))
    report = dict(scope='fresh_drive_feature_refit_not_neural_reinference',
                  original_sha256=sha_file(args.original), refit_sha256=sha_file(args.refit),
                  byte_identical=sha_file(args.original) == sha_file(args.refit),
                  protocol_sha256=original['protocol_sha256'], feature_input_sha256=sha_file(args.features),
                  comparisons=records, summary=dict(combinations=len(records),
                  decisions_compared=sum(row['test_rows'] for row in records),
                  decisions_changed=sum(row['decisions_changed'] for row in records),
                  max_raw_error=max(row['max_raw_error'] for row in records),
                  max_probability_error=max(row['max_probability_error'] for row in records)),
                  interpretation='Numerical differences are quantified, not hidden by hash replacement. '
                  'Identical decisions do not imply identical model bytes or external validity.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
