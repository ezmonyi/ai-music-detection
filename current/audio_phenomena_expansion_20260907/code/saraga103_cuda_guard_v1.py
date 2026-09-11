#!/usr/bin/env python3
"""Same-process CUDA gate for unchanged frozen neural console entrypoints.

No model/library monkeypatching. No CPU fallback, downloads, or GPU substitution.
The injectable torch/dispatch parameters are synthetic-test seams, not CLI flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import tempfile


VERSION = 'saraga103_same_process_cuda_guard_v1'
ENTRYPOINTS = {'allinone': 'all-in-one-infer', 'beats': 'beat_this'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def probe(torch, physical_gpu, environ):
    """CUDA checks occur in the same Python process that will execute the CLI."""
    require(type(physical_gpu) is int and physical_gpu > 0, 'Nonzero explicit physical GPU required')
    require(environ.get('CUDA_VISIBLE_DEVICES') == str(physical_gpu), 'CUDA_VISIBLE_DEVICES assignment mismatch')
    require(torch.cuda.is_available(), 'CUDA unavailable; refusing CPU fallback before entrypoint')
    require(torch.cuda.device_count() == 1, 'Exactly one visible CUDA device required')
    torch.cuda.init()
    require(torch.cuda.is_initialized(), 'CUDA initialization did not complete')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    require('RTX 5090' in properties.name, 'Visible CUDA device is not an RTX 5090')
    tensor = torch.empty((1,), dtype=torch.float32, device='cuda:0')
    require(tensor.device.type == 'cuda' and tensor.device.index == 0, 'CUDA allocation used wrong device')
    torch.cuda.synchronize(0)
    del tensor
    return dict(cuda_available=True, visible_device_count=1, cuda_initialized=True,
        allocation_device='cuda:0', allocation_and_synchronization_passed=True,
        cuda_visible_devices=environ['CUDA_VISIBLE_DEVICES'], physical_gpu_index=physical_gpu,
        logical_device_index=0, device_name=properties.name,
        device_uuid=str(properties.uuid) if getattr(properties, 'uuid', None) is not None else None,
        device_total_memory_bytes=int(properties.total_memory), torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda))


def publish(path, body):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.exists(), 'Guard evidence target conflict')
    require(path.parent.is_dir(), 'Guard evidence parent missing')
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.cuda-guard-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink()


def launch(stage, physical_gpu, entrypoint, argv, evidence, *, torch, dispatch=runpy.run_path):
    entrypoint, evidence = Path(entrypoint), Path(evidence)
    require(stage in ENTRYPOINTS and entrypoint.name == ENTRYPOINTS[stage], 'Unexpected neural entrypoint')
    require(entrypoint.is_absolute() and entrypoint.is_file() and not entrypoint.is_symlink()
            and entrypoint.resolve() == entrypoint, 'Noncanonical neural entrypoint')
    require(not evidence.exists(), 'Existing guard evidence forbids redispatch')
    if stage == 'allinone':
        require('-d' in argv and argv[argv.index('-d') + 1] == 'cuda', 'All-In-One CUDA arguments required')
    else:
        require('--gpu' in argv and argv[argv.index('--gpu') + 1] == '0', 'Beat This logical GPU0 arguments required')
    original = [str(entrypoint), *argv]
    code_before, entry_before = file_sha(__file__), file_sha(entrypoint)
    before = probe(torch, physical_gpu, os.environ)
    process = dict(pid=os.getpid(), hostname=socket.gethostname(),
                   python_executable=str(Path(sys.executable).resolve()))
    print(json.dumps(dict(event='cuda_guard_preflight_passed', guard_version=VERSION,
                          stage=stage, original_command=original, process=process, device=before),
                     sort_keys=True), flush=True)
    previous = sys.argv
    sys.argv = original.copy()
    try:
        try:
            dispatch(str(entrypoint), run_name='__main__')
        except SystemExit as error:
            require(error.code is None or error.code == 0, 'Neural entrypoint exited unsuccessfully: ' + str(error.code))
    finally:
        sys.argv = previous
    # A post-entrypoint failure makes the whole run fail, with no success receipt.
    after = probe(torch, physical_gpu, os.environ)
    require(before == after, 'CUDA device/runtime changed during neural entrypoint')
    require(file_sha(__file__) == code_before and file_sha(entrypoint) == entry_before,
            'Guard/entrypoint changed during neural execution')
    result = dict(status='passed_same_process_cuda_guard', guard_version=VERSION, stage=stage,
        original_command=original, original_entrypoint_sha256=entry_before,
        guard_sha256=code_before, process=process, before=before, after=after,
        entrypoint_dispatched_same_process=True, original_argv_preserved=True,
        entrypoint_returned_successfully=True, model_or_torch_monkeypatched=False,
        cpu_fallback_authorized=False)
    publish(evidence, result)
    print(json.dumps(dict(event='cuda_guard_completed', evidence=str(evidence),
                          evidence_sha256=file_sha(evidence)), sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=tuple(ENTRYPOINTS), required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--entrypoint', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('original_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(args.original_arguments and args.original_arguments[0] == '--', 'Exact original arguments delimiter required')
    import torch
    launch(args.stage, args.physical_gpu, args.entrypoint, args.original_arguments[1:], args.evidence, torch=torch)


if __name__ == '__main__':
    main()
