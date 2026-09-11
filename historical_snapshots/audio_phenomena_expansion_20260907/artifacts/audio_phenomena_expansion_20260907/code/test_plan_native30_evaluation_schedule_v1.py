"""Synthetic schedules only; no corpus audio, features, model fitting or scores."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import plan_native30_evaluation_schedule_v1 as m


def row(ident, source, label, group=None, component=None):
    return {'id': ident, 'source_group': source, 'label': label, 'role': 'development',
            'group_id': group or 'group_' + ident, 'component_id': component or 'component_' + (group or ident)}


def screen_for(rows):
    components = {}
    for r in rows:
        c = components.setdefault(r['component_id'], {'component_id': r['component_id'], 'members': [],
                                                      'candidate_screen_members': [], 'protected_relationships': [], 'link_tokens': []})
        c['members'].append(r['id'])
        c['candidate_screen_members'].append(r['id'])
    return {'rows': copy.deepcopy(rows), 'components': list(components.values())}


def bucket_group(prefix, bucket, ordinal):
    n = 0
    found = []
    while len(found) <= ordinal:
        group = prefix + '_' + str(n)
        if m.hash_fold(group) == bucket:
            found.append(group)
        n += 1
    return found[-1]


def fixture(paired=False):
    rows = []
    for source, label in [('H1', '0'), ('H2', '0'), ('A1', '1'), ('A2', '1')]:
        prefix = 'paired' if paired and source in {'H1', 'A1'} else source
        for bucket in range(5):
            for ordinal in range(3):
                group = bucket_group(prefix, bucket, ordinal)
                for sample in range(2):
                    rows.append(row(f'{source}_{bucket}_{ordinal}_{sample}', source, label, group, 'c_' + group))
    return sorted(rows, key=lambda r: r['id']), screen_for(rows)


def table(rows):
    return pd.DataFrame(rows).rename(columns={'id': '__id', 'source_group': '__source', 'label': '__label',
                                              'group_id': '__group', 'role': '__role'}).assign(__label=lambda x: x['__label'].astype(int))


class LegacyPrimitiveParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(m.__file__).with_name('frozen_evaluate_expanded_20260905.py').resolve()
        m.binding(path, m.PINS[path.name])
        spec = importlib.util.spec_from_file_location('_pinned_metadata_only_base', path)
        cls.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.base)

    def test_base_hash_exact_on_ascii_unicode_and_empty(self):
        for group in ('', 'muse:suno_cn_000107', 'aime:test', 'artist\nname', '作曲家'):
            self.assertEqual(m.hash_fold(group), self.base.hash_fold(group, 5, 20260907))
        self.assertEqual(m.id_set_hash(['b', 'a', 'b']), m.hashlib.sha256(b'a\nb\n').hexdigest())

    def test_reference_masks_match_base_on_source_and_crossclass_shared_groups(self):
        for paired in (False, True):
            rows, screen = fixture(paired)
            folds = self.base.make_folds(table(rows), 5, 20260907)
            self.assertEqual(len(folds), 25)
            for fold in folds:
                train, test = m.reference_cell(rows, fold['fold_type'], fold['heldout_source'], fold['opposite_group_fold'])
                self.assertEqual({r['id'] for r in train}, set(fold['train']['__id']))
                self.assertEqual({r['id'] for r in test}, set(fold['test']['__id']))

    def test_caps_equal_base_when_groups_do_not_cross_sources(self):
        rows = [row(f'{s}_{n}_{j}', s, label, f'{s}_g{n}') for s, label in [('H', '0'), ('A', '1')]
                for n in range(240) for j in range(2)]
        for cap, selected, info in m.coupled_caps(rows):
            legacy = self.base.deterministic_quantity(table(rows), cap, 20260907)
            self.assertEqual({r['id'] for r in selected}, set(legacy['__id']))
            self.assertFalse(info['underfilled_sources'])

    def test_shared_group_cap_is_deliberately_new_and_never_splits_global_group(self):
        ordered = sorted(['g' + str(n) for n in range(26)], key=lambda g: (m.rank_hash(g), g))
        rows = [row('h_' + g, 'H', '0', g) for g in ordered]
        rows.append(row('a_shared', 'A', '1', ordered[-1]))
        legacy = self.base.deterministic_quantity(table(rows), 25, 20260907)
        self.assertIn('a_shared', set(legacy['__id']))
        self.assertNotIn('h_' + ordered[-1], set(legacy['__id']))
        coupled = m.coupled_caps(rows)
        self.assertNotIn('a_shared', {r['id'] for r in coupled[0][1]})
        self.assertNotIn('h_' + ordered[-1], {r['id'] for r in coupled[0][1]})
        self.assertEqual(coupled[0][2]['underfilled_sources']['A']['deficit'], 1)
        self.assertTrue({'a_shared', 'h_' + ordered[-1]} <= {r['id'] for r in coupled[1][1]})


class CoupledCapsTests(unittest.TestCase):
    def test_nesting_hardcaps_and_whole_groups_with_crosssource_crosslabel_rows(self):
        rows = []
        for n in range(310):
            for source, label in [('A', '1'), ('B', '0'), ('C', '1')]:
                if source == 'B' and n % 3 == 0:
                    continue
                group = f'g{n}' if source != 'C' or n % 2 else f'c{n}'
                rows.extend([row(f'{source}_{n}_{j}', source, label, group) for j in (0, 1)])
        previous = set()
        by_group = {}
        for r in rows:
            by_group.setdefault(r['group_id'], set()).add(r['id'])
        for cap, selected, info in m.coupled_caps(rows):
            ids = {r['id'] for r in selected}
            self.assertTrue(previous <= ids)
            previous = ids
            for group_ids in by_group.values():
                self.assertTrue(not ids & group_ids or group_ids <= ids)
            if cap != 'all':
                self.assertTrue(all(n <= cap for n in info['selected_source_groups'].values()))
        self.assertEqual(previous, {r['id'] for r in rows})

    def test_permutation_does_not_change_selected_ids(self):
        rows, _ = fixture(True)
        first, second = m.coupled_caps(rows), m.coupled_caps(list(reversed(rows)))
        self.assertEqual([m.population(r)['ids'] for _, r, _ in first], [m.population(r)['ids'] for _, r, _ in second])

    def test_empty_rows_still_report_all_five_caps(self):
        output = m.coupled_caps([])
        self.assertEqual([c for c, _, _ in output], list(m.CAPS))
        self.assertTrue(all(not rows for _, rows, _ in output))


class ScheduleTests(unittest.TestCase):
    def test_127_subsets_without_M_and_exact_shared_references(self):
        rows, screen = fixture()
        schedule = m.make_schedule(rows, screen)
        self.assertEqual(len(schedule['combinations']), 127)
        self.assertEqual(len(set(schedule['combinations'])), 127)
        self.assertEqual(schedule['families'], ['S', 'D', 'R', 'P', 'F', 'H', 'SC'])
        self.assertNotIn('M', {x for c in schedule['combinations'] for x in c.split('+')})
        self.assertEqual(schedule['accounting']['intended_fold_cells'], 25)
        self.assertEqual(schedule['accounting']['eligible_uncapped_fold_cells'], 25)
        self.assertEqual(schedule['accounting']['intended_combination_cap_cells'], 25 * 7 * 127)
        self.assertEqual(schedule['accounting']['primary'], {'intended_cells': 25 * 5 * 127,
                                                            'eligible_cells': 25 * 5 * 127, 'omitted_cells': 0})
        self.assertEqual(schedule['accounting']['diagnostic'], {'intended_cells': 25 * 2 * 127,
                                                               'eligible_cells': 25 * 2 * 127, 'omitted_cells': 0})
        mapping = {r['schedule_uid']: r for r in schedule['cap_schedules']}
        counts = m.Counter(r['schedule_uid'] for r in schedule['combination_schedule_cells'])
        self.assertEqual(set(counts.values()), {127, 381})
        self.assertEqual(set(counts), set(mapping))
        cells = {r['fold_uid']: r for r in schedule['fold_cells']}
        for cap in mapping.values():
            self.assertEqual(cap['test_id_set_sha256'], cells[cap['test_population_ref']]['test']['id_set_sha256'])
            self.assertEqual(cap['uncapped_train_id_set_sha256'], cells[cap['uncapped_train_population_ref']]['uncapped_train']['id_set_sha256'])
        self.assertEqual(schedule['classifier_fits'], 0)
        self.assertFalse(schedule['fitting_authorized'])

    def test_diagnostics_allcap_only_share_exact_primary_schedule_ids(self):
        rows, screen = fixture(True)
        result = m.make_schedule(rows, screen)
        schedules = {r['schedule_uid']: r for r in result['cap_schedules']}
        primary = {(e['schedule_uid'], e['combination']) for e in result['combination_schedule_cells']
                   if e['feature_mode'] == 'values_plus_missing'}
        observed = m.Counter()
        for e in result['combination_schedule_cells']:
            observed[e['feature_mode']] += 1
            if e['feature_mode'] in {'median_only', 'missingness_only'}:
                self.assertEqual(e['analysis_role'], 'diagnostic')
                self.assertEqual(schedules[e['schedule_uid']]['quantity'], 'all')
                self.assertIn((e['schedule_uid'], e['combination']), primary)
            else:
                self.assertEqual(e['feature_mode'], 'values_plus_missing')
                self.assertEqual(e['analysis_role'], 'primary')
        self.assertEqual(observed, {'values_plus_missing': 25 * 5 * 127,
                                    'median_only': 25 * 127, 'missingness_only': 25 * 127})

    def test_all_held_source_dependencies_remove_paired_source_all_buckets(self):
        rows, screen = fixture(True)
        result = m.make_schedule(rows, screen)
        self.assertEqual(result['shared_groups']['cross_label_groups'], 15)
        for cell in result['fold_cells']:
            if cell['heldout_source'] in {'H1', 'A1'}:
                paired_source = 'A1' if cell['heldout_source'] == 'H1' else 'H1'
                self.assertIn(paired_source, cell['sources_fully_removed_by_dependency_purge'])
                self.assertNotIn(paired_source, cell['uncapped_train']['source_rows'])
                self.assertNotIn(cell['heldout_source'], cell['uncapped_train']['source_rows'])
                self.assertGreater(cell['v6_style_reference_train_before_dependency_purge']['source_rows'][paired_source], 0)
            self.assertFalse(set(cell['test']['groups']) & set(cell['uncapped_train']['groups']))
            self.assertFalse(set(cell['test']['components']) & set(cell['uncapped_train']['components']))

    def test_component_purge_crosses_group_buckets_and_transitive_aliases(self):
        rows, _ = fixture()
        h = next(r for r in rows if r['id'] == 'H1_0_0_0')
        a = next(r for r in rows if r['id'] == 'A1_1_0_0')
        for r in rows:
            if r['group_id'] == a['group_id']:
                r['component_id'] = h['component_id']
        screen = screen_for(rows)
        schedule = m.make_schedule(rows, screen)
        ordinary = next(c for c in schedule['fold_cells'] if c['fold_type'] == 'ordinary_group_holdout_descriptive' and c['opposite_group_fold'] == 0)
        self.assertIn(a['id'], ordinary['v6_style_reference_train_before_dependency_purge']['ids'])
        self.assertNotIn(a['id'], ordinary['uncapped_train']['ids'])
        self.assertIn(a['id'], ordinary['excluded_partition']['test_dependency_closure']['ids'])

    def test_full_graph_protected_bridge_outside_selected_rows(self):
        rows, _ = fixture()
        selected = row('protected_selected', 'H1', '0', 'protected_group', 'protected_c1')
        rows.append(selected)
        bridge = row('outside_bridge', 'H1', '0', 'bridge_alias', 'protected_c1')
        locked = row('outside_locked', 'H1', '0', 'bridge_alias', 'protected_c2')
        locked['role'] = 'locked'
        screen = screen_for(rows + [bridge, locked])
        screen['components'][-1]['protected_relationships'] = [{'id': locked['id'], 'role': 'locked'}]
        result = m.make_schedule(rows, screen)
        self.assertEqual(result['protected_excluded_population']['ids'], ['protected_selected'])
        self.assertEqual(result['eligible_population']['rows'], len(rows) - 1)
        for cell in result['fold_cells']:
            self.assertNotIn(selected['id'], cell['test']['ids'] + cell['uncapped_train']['ids'])

    def test_each_uncapped_fold_is_an_exact_partition(self):
        rows, screen = fixture(True)
        result = m.make_schedule(rows, screen)
        eligible = set(result['eligible_population']['ids'])
        for cell in result['fold_cells']:
            parts = [set(cell['test']['ids']), set(cell['uncapped_train']['ids'])]
            parts += [set(p['ids']) for p in cell['excluded_partition'].values()]
            self.assertEqual(set().union(*parts), eligible)
            self.assertEqual(sum(map(len, parts)), len(eligible))

    def test_omitted_cells_retained_with_reasons_not_repaired(self):
        rows = [row('h', 'H', '0'), row('a', 'A', '1')]
        result = m.make_schedule(rows, screen_for(rows))
        self.assertEqual(result['accounting']['intended_fold_cells'], 15)
        self.assertEqual(result['accounting']['eligible_cap_cells'], 0)
        self.assertEqual(len(result['omitted_fold_cells']), 15)
        self.assertTrue(all(c['omission_reasons'] for c in result['cap_schedules']))
        self.assertEqual(len(result['combination_schedule_cells']), 15 * 7 * 127)
        self.assertEqual(result['accounting']['primary']['eligible_cells'], 0)
        self.assertEqual(result['accounting']['diagnostic']['eligible_cells'], 0)
        self.assertEqual(result['classifier_fits'], 0)

    def test_same_source_cannot_have_conflicting_labels_but_globalgroup_can(self):
        rows, screen = fixture(True)
        rows[0]['label'] = '0'
        with self.assertRaisesRegex(ValueError, 'source has conflicting'):
            m.make_schedule(rows, screen)

    def test_duplicate_ids_and_non_development_fail(self):
        rows, screen = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            m.make_schedule(rows + [rows[0]], screen)
        rows[0]['role'] = 'locked'
        with self.assertRaisesRegex(ValueError, 'must be development'):
            m.make_schedule(rows, screen)

    def test_metadata_schedule_cannot_read_audio_or_feature_files(self):
        rows, screen = fixture()
        for r in rows:
            r['input'] = {'path': '/DO_NOT_OPEN/audio.wav', 'sha256': 'a' * 64}
            r['feature_path'] = '/DO_NOT_OPEN/features.json'
        with patch.object(Path, 'open', side_effect=AssertionError('pure metadata scheduling opened a file')):
            result = m.make_schedule(rows, screen)
        self.assertEqual(result['audio_files_opened'], 0)
        self.assertEqual(result['feature_files_opened'], 0)


class PublicationTests(unittest.TestCase):
    def test_json_only_draft_publication_and_no_overwrite(self):
        rows, screen = fixture()
        result = m.make_schedule(rows, screen)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            meta = root / 'fixture.json'
            meta.write_bytes(m.canonical({'synthetic': True}))
            result['input_bindings'] = {str(meta): m.binding(meta)}
            output = root / 'draft'
            publication = m.publish(result, output)
            self.assertEqual({p.name for p in output.iterdir()}, {'schedule_draft.json', 'COMMIT.json'})
            self.assertEqual(publication['classifier_fits'], 0)
            self.assertFalse(m.read(output / 'COMMIT.json')['fitting_authorized'])
            self.assertEqual(m.read(output / 'schedule_draft.json'), result)
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                m.publish(result, output)

    def test_parent_contract_pin_required_before_opening_any_input(self):
        with patch.object(Path, 'open', side_effect=AssertionError('opened unpinned metadata')):
            for digest in ('', None, 'not-a-hash'):
                with self.assertRaisesRegex(ValueError, 'parent-pinned'):
                    m.build('/missing/contract.json', digest, '/missing/plan.json', '/missing/screen.json')

    def test_wrong_parent_pin_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'contract.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                m.build(path, '0' * 64, path, path)


if __name__ == '__main__':
    unittest.main()
