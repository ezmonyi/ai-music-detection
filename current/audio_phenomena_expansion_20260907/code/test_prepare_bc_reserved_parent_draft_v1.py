import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('bc_reserved_candidate_under_test',
    HERE / 'prepare_bc_reserved_parent_draft_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def probe_canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.code = self.root / 'code'; self.code.mkdir()
        self.source = self.root / 'reserved'; self.source.mkdir()
        self.draft_dir = self.root / 'draft'; self.draft_dir.mkdir()
        self.output = self.root / 'result'
        self.calls = []
        self.producer_file = self.code / 'producer.py'; self.producer_file.write_text('producer\n')
        self.producer_sha = m.digest(self.producer_file)
        self.tool_files = {}
        for name in ('ffmpeg', 'ffprobe', 'libcodec.so', 'libprobe.so', 'ld-linux.so'):
            path = self.code / name; path.write_bytes((name + '\n').encode())
            self.tool_files[name] = path
        toolchain = {'ffmpeg': m.binding(self.tool_files['ffmpeg']),
            'ffprobe': m.binding(self.tool_files['ffprobe']),
            'codec_libraries': {'libcodec.so': m.binding(self.tool_files['libcodec.so'])}}
        codecs = {'mp3_128k': {'encoder': 'libmp3lame'},
                  'opus_96k': {'encoder': 'libopus'}}
        scope = {'synthetic_only': True, 'reserved_audio_read': False}
        operation = ['synthetic cast', 'encode', 'decode']
        self.probe = types.SimpleNamespace(VERSION='synthetic_probe', STATUS='committed_synthetic',
            SCOPE=scope, CODECS=codecs, OPERATION_ORDER=operation,
            value_hash=lambda value: hashlib.sha256(probe_canonical(value)).hexdigest(),
            tool_bindings=lambda tools: {'ffmpeg': m.binding(tools['ffmpeg']['path']),
                'ffprobe': m.binding(tools['ffprobe']['path']),
                'codec_libraries': {name: m.binding(entry['path'])
                                    for name, entry in tools['codec_libraries'].items()}})
        results = {'version': self.probe.VERSION,
            'status': 'synthetic_codec_roundtrip_characterized_not_admission',
            **scope, 'codec_recipes': codecs, 'operation_order': operation,
            'toolchain': toolchain}
        results_path = self.code / 'results.json'; m.write_json_new(results_path, results)
        results_entry = m.binding(results_path)
        commit = {'version': self.probe.VERSION, 'status': self.probe.STATUS, **scope,
            'results_sha256': self.probe.value_hash(results),
            'products': {'results.json': {k: results_entry[k] for k in ('bytes', 'sha256')}},
            'toolchain_bindings_before': self.probe.tool_bindings(toolchain),
            'toolchain_bindings_end': self.probe.tool_bindings(toolchain)}
        commit_path = self.code / 'synthetic_COMMIT.json'; m.write_json_new(commit_path, commit)
        self.rows = []
        for index in range(m.EXPECTED['recordings']):
            item = f'id{index:03d}'
            audio = self.source / (item + '.wav'); audio.write_bytes(b'audio')
            annotation = self.source / (item + '.jams'); annotation.write_bytes(b'annotation')
            self.rows.append({'item_id': item, 'split_role': 'reserved',
                'source_audio': m.binding(audio), 'source_annotation': m.binding(annotation)})
        self.document = {'version': 'bc_guitarset_reserved_admission_draft_v1',
            'status': 'prospective_reserved_admission_protocol_draft_not_frozen_not_authorized',
            'source_root': str(self.source), 'output_root': str(self.output),
            'roster': self.rows, 'reserved_roster_sha256': m.value_hash(self.rows),
            'codec_recipes': codecs, 'authorities': {
                'synthetic_codec_COMMIT': m.binding(commit_path),
                'synthetic_codec_results': results_entry}}
        draft_path = self.draft_dir / 'draft.json'; m.write_json_new(draft_path, self.document)
        self.draft_entry = m.binding(draft_path)
        draft_commit = {'status': 'committed_nonauthorizing_reserved_admission_draft',
            'reserved_audio_accessed': False, 'draft': self.draft_entry}
        commit_path = self.draft_dir / 'COMMIT.json'; m.write_json_new(commit_path, draft_commit)
        self.draft_commit_entry = m.binding(commit_path)
        self.runtime_files = {}
        for name in ('scipy', 'upfirdn', 'kernel', 'ufuncs'):
            path = self.code / (name + '.so'); path.write_bytes(name.encode())
            self.runtime_files[name] = m.binding(path)
        self.auditor_runtime = {'scipy': '1.synthetic', 'modules': {
            'scipy': self.runtime_files['scipy'],
            'scipy.signal._upfirdn': self.runtime_files['upfirdn'],
            'scipy.signal._upfirdn_apply': self.runtime_files['kernel'],
            'scipy.special._ufuncs': self.runtime_files['ufuncs']}}
        self.auditor_module = types.SimpleNamespace(
            independent_auditor_runtime=lambda: self.auditor_runtime)
        self.auditor = self.code / 'auditor.py'
        self.auditor.write_text(
            "VERSION='synthetic_auditor'\n"
            "PRODUCER_VERSION='run_bc_guitarset_reserved_admission_v1'\n"
            f"PRODUCER_SHA='{self.producer_sha}'\n"
            f"DRAFT_SHA='{self.draft_entry['sha256']}'\n"
            f"DRAFT_COMMIT_SHA='{self.draft_commit_entry['sha256']}'\n"
            f"EXPECTED={m.EXPECTED!r}\n")
        self.auditor_tests = self.code / 'test_auditor.py'; self.auditor_tests.write_text('tests\n')
        self.modules = {'codec_probe': self.probe,
            'codec': types.SimpleNamespace(runtime_snapshot=lambda: {
                'python': 'synthetic', 'executable': m.binding(self.producer_file)})}
        self.producer = types.SimpleNamespace(__file__=str(self.producer_file),
            FREEZE_VERSION='bc_guitarset_reserved_admission_parent_freeze_v1',
            load_frozen=lambda root: self.modules,
            verify_draft=lambda directory, sha, modules: (self.document, self.draft_commit_entry),
            storage_estimate=lambda: {'required_free_bytes': 12 * 1024**3})

    def close(self):
        self.tmp.cleanup()

    def ldd(self, command, **kwargs):
        self.calls.append(list(command))
        if command[1] == str(self.tool_files['ffmpeg']):
            stdout = (f"libcodec.so => {self.tool_files['libcodec.so']} (0x1)\n"
                      f"{self.tool_files['ld-linux.so']} (0x2)\nlinux-vdso.so.1 (0x3)\n")
        else:
            stdout = (f"libprobe.so => {self.tool_files['libprobe.so']} (0x1)\n"
                      f"{self.tool_files['ld-linux.so']} (0x2)\n")
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr='')

    def build(self, **changes):
        values = {'draft_dir': str(self.draft_dir),
            'draft_commit_sha': self.draft_commit_entry['sha256'],
            'auditor_path': str(self.auditor), 'auditor_sha': m.digest(self.auditor),
            'auditor_tests': str(self.auditor_tests),
            'auditor_tests_sha': m.digest(self.auditor_tests),
            'builder_tests': str((HERE / 'test_prepare_bc_reserved_parent_draft_v1.py').resolve()),
            'producer': self.producer, 'modules': self.modules,
            'auditor_module': self.auditor_module, 'ldd_run': self.ldd}
        values.update(changes)
        with mock.patch.object(m, 'PRODUCER_SHA', self.producer_sha), \
             mock.patch.object(m, 'DRAFT_SHA', self.draft_entry['sha256']), \
             mock.patch.object(m, 'DRAFT_COMMIT_SHA', self.draft_commit_entry['sha256']):
            return m.build(**values)


class ReservedParentCandidateTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.addCleanup(self.f.close)

    def test_valid_build_has_complete_nonauthorizing_freeze_fields(self):
        candidate = self.f.build()
        expected = ('version', 'status', 'draft_COMMIT', 'draft', 'output_root',
            'authorized_scope', 'expected_measurements', 'storage_estimate', 'runtime',
            'toolchain', 'codec_recipes', 'runner', 'runner_tests', 'independent_auditor',
            'independent_auditor_tests', 'independent_auditor_runtime',
            'linked_library_bindings')
        self.assertTrue(all(key in candidate for key in expected))
        self.assertEqual(candidate['status'], m.CANDIDATE_STATUS)
        self.assertEqual(candidate['authorized_scope'], m.NONAUTHORIZING_SCOPE)
        self.assertFalse(candidate['authorization_boundary']['this_candidate_authorizes_reserved_audio_access'])
        self.assertEqual(candidate['independent_auditor_runtime'], self.f.auditor_runtime)

    def test_ldd_union_is_flat_complete_and_ignores_vdso(self):
        candidate = self.f.build()
        expected = {str(self.f.tool_files[name]) for name in
                    ('libcodec.so', 'libprobe.so', 'ld-linux.so')}
        self.assertEqual(set(candidate['linked_library_bindings']), expected)
        self.assertEqual(len(self.f.calls), 4)  # ffmpeg+ffprobe, before+end
        self.assertTrue(all(call[0] == '/usr/bin/ldd' for call in self.f.calls))

    def test_reserved_payloads_are_never_hashed_or_opened(self):
        original = m.digest
        def guarded(path):
            self.assertFalse(Path(path).is_relative_to(self.f.source), str(path))
            return original(path)
        with mock.patch.object(m, 'digest', side_effect=guarded):
            candidate = self.f.build()
        self.assertFalse(candidate['metadata_validation']['reserved_source_bytes_opened_or_hashed'])
        self.assertEqual(candidate['metadata_validation']['reserved_source_files_statted'], 180)

    def test_wrong_or_unpinned_auditor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exact stable auditor'):
            self.f.build(auditor_sha='')
        self.f.auditor.write_text("VERSION='unrelated'\n")
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            self.f.build(auditor_sha=m.digest(self.f.auditor))

    def test_source_stat_or_symlink_boundary_is_rejected_without_hashing(self):
        path = Path(self.f.rows[0]['source_audio']['path'])
        path.write_bytes(b'changed-size')
        with self.assertRaisesRegex(ValueError, 'stat/size'):
            self.f.build()

    def test_codec_results_metadata_binding_mutation_is_rejected(self):
        results = Path(self.f.document['authorities']['synthetic_codec_results']['path'])
        results.write_bytes(results.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'authority changed|binding SHA'):
            self.f.build()

    def test_ldd_failure_and_missing_dependency_are_rejected(self):
        def failed(*args, **kwargs):
            return types.SimpleNamespace(returncode=1, stdout='', stderr='not found')
        with self.assertRaisesRegex(ValueError, 'ldd failed'):
            self.f.build(ldd_run=failed)

    def test_dependency_graph_change_during_build_is_rejected(self):
        count = 0
        def changing(command, **kwargs):
            nonlocal count
            count += 1
            result = self.f.ldd(command, **kwargs)
            if count > 2 and command[1] == str(self.f.tool_files['ffprobe']):
                result.stdout = result.stdout.replace('libprobe.so', 'libcodec.so').replace(
                    str(self.f.tool_files['libprobe.so']), str(self.f.tool_files['libcodec.so']))
            return result
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            self.f.build(ldd_run=changing)

    def test_existing_output_and_candidate_overwrite_are_rejected(self):
        self.f.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'new canonical'):
            self.f.build()
        self.f.output.rmdir()
        candidate = self.f.build()
        path = self.f.root / 'candidate.json'
        m.write_json_new(path, candidate)
        with self.assertRaisesRegex(ValueError, 'new canonical candidate'):
            m.write_json_new(path, candidate)

    def test_actual_producer_rejects_candidate_until_separate_authorization(self):
        candidate = self.f.build()
        freeze = self.f.root / 'candidate.json'; m.write_json_new(freeze, candidate)
        producer = m.load_producer()
        proof = {'results': {'toolchain': candidate['toolchain'],
                             'codec_recipes': candidate['codec_recipes']}}
        modules = {'codec_probe': types.SimpleNamespace(
            verify_result=lambda root, sha: proof)}
        with mock.patch.object(producer, 'verify_draft', return_value=(
                self.f.document, self.f.draft_commit_entry)), \
             self.assertRaisesRegex(ValueError, 'parent freeze schema'):
            producer.verify_freeze(self.f.draft_dir, producer.DRAFT_COMMIT_SHA,
                                   freeze, m.digest(freeze), modules)


if __name__ == '__main__':
    unittest.main()
