import tempfile
import types
import unittest
from pathlib import Path
import prepare_native30_sdrp_operational_draft_v2 as m
import test_prepare_native30_sdrp_operational_draft_v1 as old


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root / 'extract_expanded_four_family.py'
        self.path.write_bytes(b'unchanged numerical code\n')
        self.binding = m.v1.binding(self.path)
        self.backend = types.SimpleNamespace(o=types.SimpleNamespace(EXTRACTOR_SHA=self.binding['sha256']))

    def test_actual_schema_without_extractor(self):
        prior = {'old_code_root': str(self.root), 'other': 'preserved'}
        result = m.resolve_prior(prior, self.backend)
        self.assertEqual(result['extractor'], self.binding)
        self.assertEqual(result['other'], 'preserved')
        self.assertNotIn('extractor', prior)

    def test_modified_scientific_code_rejected(self):
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            m.resolve_prior({'old_code_root': str(self.root)}, self.backend)

    def test_conflicting_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            m.resolve_prior({'old_code_root': str(self.root), 'extractor': {}}, self.backend)

    def test_symlink_root_rejected(self):
        link = self.root / 'link'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            m.resolve_prior({'old_code_root': str(link)}, self.backend)

    def test_full_build_actual_schema_keeps_nonauthorizing_scope(self):
        f = old.Fixture(); self.addCleanup(f.close)
        actual = f.code / 'extract_expanded_four_family.py'
        actual.write_bytes(b'extractor\n')
        f.prior.pop('extractor')
        f.prior['old_code_root'] = str(f.code)
        f.backend.o.EXTRACTOR_SHA = m.v1.digest(actual)
        with old.mock.patch.object(m.v1, 'OPERATIONAL_FREEZE_SHA', f.operational['_sha256']), \
             old.mock.patch.object(m.v1, 'OPERATIONAL_SHA', f.files['operational_runner']['sha256']):
            draft = m.build(f.operational['_path'], f.operational['_sha256'],
                            f.completion['path'], f.completion['sha256'], str(f.output),
                            str(Path(__file__).resolve()), backend=f.backend,
                            context_loader=f.context_loader, cpu_runtime={'cpu': 'synthetic'})
        self.assertEqual(draft['extractor'], m.v1.binding(actual))
        self.assertEqual(draft['draft_builder']['path'], str(Path(m.__file__).resolve()))
        self.assertFalse(draft['feature_extraction_authorized'])
        self.assertEqual(f.loader_audio, [False])
        self.assertFalse(f.output.exists())


if __name__ == '__main__':
    unittest.main()
