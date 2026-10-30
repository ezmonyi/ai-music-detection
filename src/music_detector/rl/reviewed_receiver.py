"""Explicit receiver-specific audit binding; original replay defaults stay strict."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .numerical_calibration import digest, write_once
from .numerical_audit_v2 import evaluate_audit, validate_plan


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):value.update(block)
    return value.hexdigest()


def check_options(options, original):
    if digest(original) != '3e6e8987f6313b37352146c1835e973d062976980012987e537ad39117a932a0':
        raise ValueError('Original offline objective/guard configuration changed')
    expected = {**original, 'replay_tolerance': .0003, 'replay_steps': 31}
    if options != expected or original['replay_tolerance'] != 1e-6:
        raise ValueError('Reviewed v2 permits only the explicit replay tolerance and stronger full-time probe')


def receiver_flags():
    import torch
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
        'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}


def audit_evidence(audit_root, previous_root, authorization_path):
    root, previous_root = Path(audit_root), Path(previous_root)
    plan_path = root / 'frozen/PROTOCOL_V2_PROPOSAL.json'
    plan = json.loads(plan_path.read_text())
    previous = json.loads((previous_root / 'frozen/PRE_REGISTERED_SELECTION.json').read_text())
    auth = json.loads(Path(authorization_path).read_text())
    validate_plan(plan, previous)
    if (auth.get('user_instruction') != '批准 v2 复核，通过后继续 RL'
            or auth.get('proposal_file_sha256') != file_sha(plan_path)
            or auth.get('formal_training_authorized_only_after_independent_audit_passes') is not True):
        raise ValueError('User authorization does not match this fixed audited proposal')
    summary = json.loads((root / 'audit/SUMMARY.json').read_text())
    controls = json.loads((root / 'audit/NATIVE_CONTROLS.json').read_text())
    passed_controls = (len(controls) == 2 and {v['behavior'] for v in controls} == {'g100','g10'}
        and all(v['steps_verified'] == 31 and v['tolerance'] == 1e-6 and v['maximum_abs_logprob_difference'] <= 1e-6 for v in controls))
    rows = [json.loads(line) for line in (root / 'audit/audit_measurements.jsonl').read_text().splitlines() if line.strip()]
    recalculated = evaluate_audit(rows, plan, controls_passed=passed_controls)
    if (recalculated['audit_passed'] is not True or summary.get('audit_passed') is not True
            or summary.get('status') != 'reviewed_v2_independent_audit_passed'
            or summary.get('corrupted_logprob_controls_rejected') is not True
            or summary.get('original_cached_probabilities_modified') is not False
            or summary.get('numerical_flags_changed') is not False
            or summary.get('audit') != recalculated['audit']):
        raise ValueError('Independent audit did not pass; no reviewed receiver profile can be issued')
    flag_path = root / 'audit/RECEIVER_FLAGS.json'
    flags = json.loads(flag_path.read_text())
    if flags != receiver_flags():
        raise ValueError('Current numerical receiver differs from the audited receiver')
    files = [plan_path, root / 'audit/SUMMARY.json', root / 'audit/NATIVE_CONTROLS.json',
             root / 'audit/audit_measurements.jsonl', root / 'audit/INPUT_VERIFICATION.json', flag_path, Path(authorization_path)]
    return {'kind': 'reviewed_rtx3090_cross_gpu_logprob_v2', 'receiver_logprob_tolerance': .0003,
        'original_default_tolerance': 1e-6, 'native_repeat_tolerance': 1e-6,
        'receiver_hostname': os.uname().nodename, 'receiver_flags': flags,
        'evidence': [{'path': str(p.resolve()), 'sha256': file_sha(p)} for p in files],
        'plan_sha256': digest(plan), 'measured_transitions': len(rows),
        'observed_maximum_error': summary['audit']['absolute_logprob_error_maximum'],
        'quality_drift_sampler_reward_guards_changed': False,
        'recorded_behavior_probabilities_replaced': False, 'v1_failure_preserved': True}


def validated_profile(path):
    profile = json.loads(Path(path).read_text())
    if (profile.get('kind') != 'reviewed_rtx3090_cross_gpu_logprob_v2'
            or profile.get('receiver_logprob_tolerance') != .0003
            or profile.get('original_default_tolerance') != 1e-6
            or profile.get('native_repeat_tolerance') != 1e-6
            or profile.get('measured_transitions') != 2976
            or profile.get('receiver_hostname') != os.uname().nodename
            or profile.get('receiver_flags') != receiver_flags()):
        raise ValueError('Reviewed profile is invalid for the current receiver')
    for entry in profile['evidence']:
        if file_sha(entry['path']) != entry['sha256']:
            raise ValueError('Audited numerical evidence changed after profile issue')
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    issue = commands.add_parser('issue')
    for name in ('audit-root','previous-root','authorization','output'):issue.add_argument('--'+name, required=True)
    train = commands.add_parser('train')
    for name in ('profile','original-options','options','index','checkpoint','checkpoint-sha256','validation-data','test-data','runtime','output'):
        train.add_argument('--'+name, required=True)
    train.add_argument('--resume');train.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    if args.command == 'issue':
        profile = audit_evidence(args.audit_root, args.previous_root, args.authorization)
        write_once(args.output, profile)
        print(json.dumps({'status':'explicit_reviewed_receiver_profile_issued','profile_sha256':file_sha(args.output),'training_started':False}))
        return
    profile = validated_profile(args.profile)
    original = json.loads(Path(args.original_options).read_text())
    options = json.loads(Path(args.options).read_text())
    check_options(options, original)
    binding = {'receiver_profile_sha256':file_sha(args.profile), 'receiver_profile':profile,
               'numerical_replay_gate_changed_explicitly':True,
               'original_strict_default_unchanged':True, 'options_sha256':file_sha(args.options)}
    if args.resume:
        previous_run = Path(args.resume).resolve().parent.parent
        prior = json.loads(Path(str(previous_run)+'.NUMERICAL_PROTOCOL.json').read_text())
        if prior != binding:
            raise ValueError('Offline resume numerical profile changed')
    output = Path(args.output)
    if output.exists():raise FileExistsError('Do not overwrite a training run')
    write_once(str(output)+'.NUMERICAL_PROTOCOL.json', binding)
    from .offline_train import load_options, train_offline
    value = train_offline(args.index, args.checkpoint, args.checkpoint_sha256, args.validation_data,
        args.test_data, args.output, load_options(args.options), overrides=json.loads(Path(args.runtime).read_text()),
        resume=args.resume, max_updates=args.max_updates)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
