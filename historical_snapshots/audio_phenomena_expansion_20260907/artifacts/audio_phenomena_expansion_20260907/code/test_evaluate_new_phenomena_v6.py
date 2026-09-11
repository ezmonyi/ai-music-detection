"""Synthetic v6 contract/numeric checks only; no real extraction or fitting."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate_new_phenomena_v6 as E


def table(groups=4):
    rows = []
    for source, label in [('human_a', 0), ('human_b', 0), ('ai_a', 1), ('ai_b', 1)]:
        for i in range(groups):
            for salt in range(10000):
                group = f'synthetic_v6_{source}_{i}_{salt}'
                if E.BASE.hash_fold(group, 5, E.SEED) == i % 4:
                    break
            rows.append(dict(__id=f'synthetic_v6_{source}_{i}', __label=label, __source=source,
                __group=group, __role='development', duration_view='60s', native_sample_rate_hz='44100',
                **{c: np.nan if (i+j) % 3 == 0 else float(i-label+j/100) for j, c in enumerate(E.DESCRIPTORS)}))
    return pd.DataFrame(rows)


def csv_write(path, rows, columns):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def fixture(root):
    package = root/'package'; package.mkdir()
    lineage = root/'synthetic_lineage.json'
    E.write_json(lineage, {'synthetic_test_only': True})
    source = table(); metadata = []; features = []
    for row in source.to_dict('records'):
        metadata.append(dict(id=row['__id'], label=row['__label'], source_group=row['__source'],
                             group_id=row['__group'], role='development', duration_view='60s', native_sample_rate_hz='44100'))
        features.append(dict(id=row['__id'], **{c: '' if np.isnan(row[c]) else str(row[c]) for c in E.DESCRIPTORS}))
    csv_write(package/'metadata_60s.csv', metadata, E.PREP5.META)
    csv_write(package/'features_60s.csv', features, ['id', *E.DESCRIPTORS])
    csv_write(package/'origin_ledger.csv', [{'id': r['id']} for r in metadata], ['id'])
    E.write_json(package/'family_config.json', E.family_config())
    contract = dict(lineage_validated=True, original_54_values_preserved=True, native_stereo_only=True,
                    historical_locked_pilot_ids_and_groups_excluded=True, synthetic_fixture_only=True,
                    input_files_sha256={str(lineage): E.sha(lineage)})
    E.write_json(package/'preparation_audit.json', dict(schema_version=6, status='prepared_not_authorized_for_fitting',
        rows=16, synthetic_test_only=True, fitting_authorized=False, scoring_authorized=False,
        contract=contract, contract_sha256=E.digest(contract), files_sha256={n: E.sha(package/n) for n in E.PRODUCTS}))
    E.write_json(package/'COMMIT.json', dict(status='committed', files={n: {
        'sha256': E.sha(package/n), 'bytes': (package/n).stat().st_size} for n in E.PRODUCTS | {'preparation_audit.json'}}))
    return package


class V6Tests(unittest.TestCase):
    def test_actual_gate_decision_and_independent_replay_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); package = fixture(root)
            freeze_path = root/'gate_freeze.json'; raw_path = root/'COMMIT.json'
            gate_path = root/'gate_result.json'; review_path = root/'gate_review.json'
            E.write_json(freeze_path, dict(candidate='SC_L6', candidate_features=E.SC_COLUMNS,
                         selected=[{'track_id': f'synthetic_{i}'} for i in range(7)],
                         bindings={'/original_host/protocol.md': 'a'*64}))
            E.write_json(raw_path, dict(freeze_sha256=E.sha(freeze_path), products={'synthetic': {'sha256': 'b'*64}}))
            gate = dict(measurement_gate_passed=True, candidate='SC_L6', candidate_features=E.SC_COLUMNS,
                        classifier_fitted=False, decision='eligible_for_separately_frozen_exploratory_study',
                        primary_checks=196, primary_passed=196, codec_pairs=28, eligible_codec_pairs=28,
                        freeze_sha256=E.sha(freeze_path), raw_commit_sha256=E.sha(raw_path))
            review = dict(passed=True, classifier_fitted=False, freeze_sha256=E.sha(freeze_path),
                          raw_commit_sha256=E.sha(raw_path), primary_checks_replayed=196, ratios_replayed=168)
            extraction_dir = root/'synthetic_extraction'; extraction_dir.mkdir()
            E.write_json(extraction_dir/'COMMIT.json', {'synthetic_only': True})
            proof = E.read_json(package/'preparation_audit.json')
            proof['synthetic_test_only'] = False
            proof['contract']['input_files_sha256'][str(extraction_dir/'COMMIT.json')] = E.sha(extraction_dir/'COMMIT.json')

            def reseal():
                gate_path.write_text(E.canonical(gate).decode())
                review.setdefault('gate_receipt_sha256', E.sha(gate_path))
                review_path.write_text(E.canonical(review).decode())
                proof['contract']['external_gate'] = dict(receipt_path=str(gate_path), receipt_sha256=E.sha(gate_path),
                    freeze_path=str(freeze_path), raw_commit_path=str(raw_path))
                proof['contract']['independent_gate_review'] = dict(receipt_path=str(review_path), receipt_sha256=E.sha(review_path))
                proof['contract_sha256'] = E.digest(proof['contract'])
                (package/'preparation_audit.json').write_text(E.canonical(proof).decode())
                marker = E.read_json(package/'COMMIT.json')
                marker['files']['preparation_audit.json'] = dict(sha256=E.sha(package/'preparation_audit.json'),
                    bytes=(package/'preparation_audit.json').stat().st_size)
                (package/'COMMIT.json').write_text(E.canonical(marker).decode())

            reseal()
            # Isolate actual gate validation only. This mock does NOT validate
            # real lineage or admit these16 fake identities to the real cohort;
            # strict preparation rejection has separate tests.
            with patch.object(E.PREP6, 'validate_package', return_value=proof):
                E.validate_package(package, False)
                gate['measurement_gate_passed'] = False; reseal()
                with self.assertRaisesRegex(ValueError, 'gate not passed'): E.validate_package(package, False)
                gate['measurement_gate_passed'] = True; review['passed'] = False; reseal()
                with self.assertRaisesRegex(ValueError, 'Independent gate'): E.validate_package(package, False)
                review['passed'] = True; review['gate_receipt_sha256'] = '0'*64; reseal()
                with self.assertRaisesRegex(ValueError, 'Independent gate'): E.validate_package(package, False)

    def test_exact_grid_and_immutable_v5(self):
        self.assertEqual(list(E.FAMILIES), ['S', 'D', 'R', 'P', 'F', 'H', 'M', 'SC'])
        self.assertEqual(len(E.DESCRIPTORS), 60)
        self.assertEqual(E.DESCRIPTORS[:54], E.V5.DESCRIPTORS)
        self.assertEqual(len(E.COMBINATIONS), 255)
        self.assertEqual(len(set(E.COMBINATIONS)), 255)
        self.assertEqual([c for c in E.COMBINATIONS if 'SC' not in c.split('+')], E.V5.COMBINATIONS)
        self.assertEqual(len(E.V5.COMBINATIONS), 127)
        self.assertEqual(E.QUANTITIES, (25, 50, 100, 200, 'all'))
        for name, expected in E.V5.FIXED_CORE.items():
            self.assertEqual(E.sha(E.ROOT/'code'/name), expected)

    def test_shared_schedule_exclusions_and_accounting(self):
        schedule, omitted, count = E.make_schedule(table(), True)
        self.assertEqual(count['primary_fits'], count['valid_folds']*5*255)
        self.assertEqual(count['diagnostic_fits'], count['valid_folds']*255*2)
        self.assertTrue(omitted)
        for record, train, test, fold in schedule:
            self.assertFalse(set(train['__group']) & set(test['__group']))
            self.assertFalse(set(train['__id']) & set(test['__id']))
            if fold['fold_type'] != 'ordinary_group_holdout_descriptive':
                self.assertNotIn(fold['heldout_source'], set(train['__source']))
        for i in range(0, len(schedule), 5):
            self.assertEqual(len({r['test_id_set_sha256'] for r, *_ in schedule[i:i+5]}), 1)
        with self.assertRaisesRegex(ValueError, 'Real38'):
            E.make_schedule(table(), False)
        with self.assertRaisesRegex(ValueError, 'Role leakage'):
            E.make_schedule(table().assign(__role='locked'), True)

    def test_caps_nested_and_all_selected_group_rows_retained(self):
        source = table(52)
        source = pd.concat([source, source.iloc[[0]].assign(__id='synthetic_v6_extra')], ignore_index=True)
        previous = set()
        for cap in E.QUANTITIES:
            selected = E.BASE.deterministic_quantity(source, cap, E.SEED)
            self.assertTrue(previous <= set(selected['__id'])); previous = set(selected['__id'])
            if cap != 'all': self.assertLessEqual(selected.groupby('__source')['__group'].nunique().max(), cap)
            for group in selected['__group'].unique():
                self.assertEqual(set(selected.loc[selected['__group'] == group, '__id']),
                                 set(source.loc[source['__group'] == group, '__id']))

    def test_sc_modes_core_numerics_and_train_only_transforms(self):
        schedule, _, _ = E.make_schedule(table(), True)
        record, train, test, _ = next(s for s in schedule if s[0]['quantity'] == 'all')
        train = train.copy(); train[E.SC_COLUMNS[0]] = np.nan
        for mode in E.MODES:
            model, score, _ = E.model_result(train, test, record, 'SC', mode, 1)
            changed = test.copy(); changed[E.DESCRIPTORS] = 1e9
            second, _, _ = E.model_result(train, changed, record, 'SC', mode, 1)
            self.assertEqual(model, second)
            self.assertEqual(model['medians'][0], 0.)
            self.assertEqual(model['ridge'], 10.); self.assertEqual(model['threshold'], .5)
            self.assertEqual(model['prediction_link'], 'identity')
            self.assertEqual(len(model['mean']), 12 if mode == E.MODES[0] else 6)
            np.testing.assert_array_equal(score, E.V2.predict_candidate(test, model))
        model, score, _ = E.model_result(train, changed, record, 'D', E.MODES[0], 1)
        self.assertTrue(np.any((score < 0) | (score > 1)))

    def test_no_fit_draft_and_missing_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); package = fixture(root)
            with patch.object(E.V2, 'fit_candidate', side_effect=AssertionError('draft cannot fit')), \
                 patch.object(E.V2, 'predict_candidate', side_effect=AssertionError('draft cannot predict')):
                E.main(['--stage', 'draft', '--package-dir', str(package), '--output-dir', str(root/'draft'), '--synthetic-test-only'])
                with self.assertRaisesRegex(ValueError, 'Parent-frozen'):
                    E.main(['--stage', 'run', '--package-dir', str(package), '--output-dir', str(root/'never'), '--synthetic-test-only'])
            self.assertFalse((root/'never').exists())
            result = E.read_json(root/'draft/preregistration_draft.json')
            self.assertFalse(result['fitting_started'])
            self.assertEqual(result['contract']['combinations'], E.COMBINATIONS)
            self.assertFalse((root/'draft/fold_models.jsonl').exists())
            E.V5.verify_publication(root/'draft')
            with self.assertRaisesRegex(ValueError, 'Exact2174'): E.load_table(package, False)
            with self.assertRaisesRegex(ValueError, 'Invalid v6'): E.validate_package(package, False)
            (package/'features_60s.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): E.validate_package(package, True)

    def test_one_family_stream_and_reject_unverified_internal_run(self):
        schedule, _, _ = E.make_schedule(table(), True)
        record, train, test, fold = next(s for s in schedule if s[0]['quantity'] == 'all')
        with patch.object(E.V2, 'fit_candidate', side_effect=AssertionError('unapproved fit')):
            with self.assertRaisesRegex(ValueError, 'authorization'):
                E.run_evaluation(Path('unused'), [], {}, {'status': 'draft'})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = dict(schedule=[record], accounting=dict(primary_fits=1, primary_prediction_rows=len(test),
                            diagnostic_fits=2, diagnostic_prediction_rows=2*len(test)))
            auth = dict(status='frozen_verified', contract_sha256=E.digest(contract))
            with patch.object(E, 'COMBINATIONS', ['SC']), patch.object(E.V5, 'recheck'), \
                 patch.object(E.V2, 'choose_train_threshold', side_effect=AssertionError('no tuning')):
                result = E.run_evaluation(output, [(record, train, test, fold)], contract, auth)
            self.assertEqual(result['model_instances'], 3)
            models = [json.loads(line) for line in (output/'fold_models.jsonl').read_text().splitlines()]
            self.assertEqual({r['feature_mode'] for r in models}, set(E.MODES))
            self.assertTrue(all(r['test_id_set_sha256'] == record['test_id_set_sha256'] for r in models))


if __name__ == '__main__':
    unittest.main()
