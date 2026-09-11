"""Metadata-only v1 successor resolving the actual old_code_root schema.

No scientific code, runtime, completion validation or authorization is changed.
The failed v1 and its tests remain immutable historical evidence.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import prepare_native30_sdrp_operational_draft_v1 as v1

V1_SHA = 'c369c5246e8ada56f862d0519eb01e36b174482ba3b81959f161fde90f72732f'


def resolve_prior(prior, backend):
    """Resolve only the unchanged extractor accepted by actual prepare()."""
    root = Path(prior['old_code_root'])
    v1.require(root.is_absolute() and root.resolve() == root and root.is_dir()
               and not root.is_symlink(), 'canonical old_code_root required')
    extractor = v1.binding(root / 'extract_expanded_four_family.py', backend.o.EXTRACTOR_SHA)
    if 'extractor' in prior:
        v1.require(prior['extractor'] == extractor, 'conflicting historical extractor binding')
    return {**prior, 'extractor': extractor}


def build(operational_freeze, operational_freeze_sha, completion, completion_sha,
          output_root, builder_tests, *, backend=None, context_loader=None, cpu_runtime=None):
    v1.binding(Path(v1.__file__).resolve(), V1_SHA)
    backend = v1.load_backend() if backend is None else backend
    loader = backend.load_operational().setup_context if context_loader is None else context_loader

    def actual_context(*args, **kwargs):
        context = loader(*args, **kwargs)
        v1.require(isinstance(context, tuple) and len(context) == 6, 'operational metadata context')
        return (context[0], resolve_prior(context[1], backend), *context[2:])

    draft = v1.build(operational_freeze, operational_freeze_sha, completion,
                     completion_sha, output_root, builder_tests, backend=backend,
                     context_loader=actual_context, cpu_runtime=cpu_runtime)
    draft['draft_builder_predecessor'] = draft['draft_builder']
    draft['draft_builder'] = v1.binding(Path(__file__).resolve())
    draft['schema_resolution'] = 'old_code_root/extract_expanded_four_family.py with unchanged EXTRACTOR_SHA'
    return draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('operational-parent-freeze', 'operational-parent-freeze-sha256',
                 'completion', 'completion-sha256', 'output-root', 'tests', 'draft'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mode', choices=('preflight', 'publish'), default='preflight')
    args = parser.parse_args()
    draft = build(args.operational_parent_freeze, args.operational_parent_freeze_sha256,
                  args.completion, args.completion_sha256, args.output_root, args.tests)
    result = {'status': 'metadata_preflight_passed_not_authorized',
              'feature_extraction_authorized': False,
              'rows': draft['terminal_completion_metadata_validation']['rows']}
    if args.mode == 'publish':
        v1.write_json_new(Path(args.draft), draft)
        result.update(status='nonauthorizing_draft_published', draft=v1.binding(args.draft))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
