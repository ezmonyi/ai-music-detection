"""Independent, user-reviewed v2 receiver audit; never runs an optimizer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import traceback

from .numerical_calibration import digest, summarize, write_once


def validate_plan(plan, previous):
    if (plan.get('schema_version') != 2 or plan.get('candidate_absolute_logprob_budget') != .0003
            or plan.get('native_and_repeated_forward_tolerance') != 1e-6
            or plan.get('production_protocol_activation') is not False
            or plan.get('formal_optimizer_updates') != 0
            or plan.get('audit_may_increase_budget') is not False
            or plan.get('planned_independent_audit_transitions') != 2976
            or plan.get('all_four_candidates') is not True
            or plan.get('all_31_stochastic_steps') is not True
            or plan.get('repeat_each_forward') is not True):
        raise ValueError('Only the explicitly reviewed, fixed v2 audit is supported')
    if abs(plan['ratio_relative_error_ceiling'] - math.expm1(.0003)) > 1e-15:
        raise ValueError('V2 mathematical budget changed')
    groups = plan['fresh_audit_groups']
    keys = {g['key'] for g in groups}
    old = {g['key'] for g in previous['groups']}
    if len(groups) != 24 or len(keys) != 24 or keys & old:
        raise ValueError('V2 audit groups must be new and unique')
    for g in groups:
        if (g['split'] != 'audit' or g['shard'] not in ('portable', 'legacy24', 'legacy26')
                or g['key'] != f"{g['shard']}/g{g['group']:06d}"
                or g['behavior'] != ('g100' if g['shard'] == 'portable' else 'g10')):
            raise ValueError('Wrong v2 group or behavior identity')
        if ((g['shard'] == 'portable' and g['group'] // 25 in [0, *previous['portable_blocks']])
                or (g['shard'] == 'legacy24' and g['group'] in (0, 1))):
            raise ValueError('Previously inspected discovery archive/group is not independent')
    if any(sum(g['behavior'] == b for g in groups) != 12 for b in ('g100', 'g10')):
        raise ValueError('V2 requires 12 groups per behavior')
    if sorted({g['group'] // 25 for g in groups if g['shard'] == 'portable'}) != sorted(plan['portable_blocks']):
        raise ValueError('Wrong archive block coverage')


def evaluate_audit(rows, plan, *, controls_passed):
    expected = {(g['key'], s, t) for g in plan['fresh_audit_groups'] for s in range(4) for t in range(10, 41)}
    identities = [(r['key'], r['sample'], r['step']) for r in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError('Incomplete or duplicate full-stochastic-time audit')
    behavior = {g['key']: g['behavior'] for g in plan['fresh_audit_groups']}
    if any(r['split'] != 'audit' or r['behavior'] != behavior[r['key']] for r in rows):
        raise ValueError('Wrong split/policy in independent measurements')
    stats = summarize(rows)
    repeat = stats['same_gpu_repeat_logprob_error_maximum']
    passed = bool(controls_passed and stats['absolute_logprob_error_maximum'] <= .0003
                  and repeat is not None and repeat <= 1e-6)
    return {'status': 'reviewed_v2_independent_audit_passed' if passed else 'reviewed_v2_independent_audit_failed',
        'audit_passed': passed, 'receiver_logprob_tolerance': .0003, 'original_default_tolerance': 1e-6,
        'native_repeat_tolerance': 1e-6, 'audit': stats, 'plan_sha256': digest(plan),
        'audit_refitted_budget': False, 'formal_optimizer_updates': 0, 'production_protocol_activation': False,
        'can_prepare_explicit_receiver_profile': passed,
        'scope': 'This receiver and frozen model/runtime/behaviors; not a universal cross-GPU guarantee'}


def run(plan_path, authorization_path, previous_root, runtime_path, base, export, scratch, output):
    import torch
    from .numerical_calibration_gpu import G100_SHA, G10_SHA, TEST_SHA, VALIDATION_SHA, _file_sha
    from .offline_archive import hashes
    from .portable_delivery import _read_archive
    from .offline_protocol import checkpoint, experiment_contract, model_signature, policy_fingerprint, reject_overlap, runtime_config, validate_record, verify_members
    from .offline_trajectory import verify_replay
    from .policy import load_adapter_state
    from .flow import gaussian_log_prob, transition_mean
    from .trainer import create_backend, setup_policy, _memory

    plan_path, previous_root, base, export, scratch, output = map(Path, (plan_path, previous_root, base, export, scratch, output))
    plan = json.loads(plan_path.read_text())
    auth = json.loads(Path(authorization_path).read_text())
    previous = json.loads((previous_root / 'frozen/PRE_REGISTERED_SELECTION.json').read_text())
    validate_plan(plan, previous)
    if (auth['proposal_file_sha256'] != _file_sha(plan_path)
            or auth['approved_fixed_cross_gpu_logprob_tolerance'] != .0003
            or auth['user_instruction'] != '批准 v2 复核，通过后继续 RL'):
        raise ValueError('Concrete user-reviewed protocol identity mismatch')
    if output.exists() or scratch.exists():
        raise FileExistsError('Preserve earlier attempts; do not overwrite or repeat an audit')
    output.mkdir(parents=True); scratch.mkdir(parents=True)
    start = time.monotonic()
    backend = None

    def state(phase, **extra):
        value = {'utc': datetime.now(timezone.utc).isoformat(), 'phase': phase,
                 'elapsed_seconds': time.monotonic() - start, 'formal_optimizer_updates': 0, **extra}
        temp = output / 'state.writing'; temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(output / 'state.json')
        print(json.dumps(value), flush=True)

    def expired(_signal, _frame):
        raise TimeoutError('Independent v2 audit timebox expired; no optimizer was run')

    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    try:
        write_once(output / 'FROZEN_V2_PLAN.json', plan)
        write_once(output / 'USER_AUTHORIZATION.json', auth)
        state('verifying_v1_and_new_input_identities')
        assert str(torch.__version__) == '2.10.0+cu128' and torch.version.cuda == '12.8'
        assert 'RTX 3090' in torch.cuda.get_device_name() and torch.cuda.is_bf16_supported(including_emulation=False)
        v1 = json.loads((previous_root / 'study/INPUT_VERIFICATION.json').read_text())
        assert _file_sha(export / 'EXPORT_INDEX.json') == v1['source_export_index_sha256']
        c100, cfg = checkpoint(base / 'runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt', G100_SHA)
        c10, cfg10 = checkpoint(base / 'runs/a6000-srf-pilot-main-v2/checkpoints/group_000010.pt', G10_SHA)
        assert experiment_contract(cfg) == experiment_contract(cfg10)
        cfg = runtime_config(cfg, json.loads(Path(runtime_path).read_text()))
        policies = {'g100': c100['policy'], 'g10': c10['policy']}
        fp = {b: policy_fingerprint(p) for b, p in policies.items()}
        assert fp == v1['policy_fingerprints'] and fp['g100'] != fp['g10']
        heldout = []
        for name, expected in (('validation', VALIDATION_SHA), ('test', TEST_SHA)):
            path = base / f'data/musiccaps_a6000_curated_500_20261003/{name}.jsonl'
            assert _file_sha(path) == expected
            heldout += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        index = json.loads((export / 'EXPORT_INDEX.json').read_text())
        portable = scratch / 'portable'; portable.mkdir()
        verified = []
        for name in ['metadata.tar', *[f'groups_{b*25:06d}_{b*25+24:06d}.tar' for b in plan['portable_blocks']]]:
            state('verifying_new_archive', archive=name)
            receipt = index['archives'][name]
            assert hashes(export / name) == receipt['archive']
            _read_archive(export / name, receipt['files'], output=portable)
            verified.append({'name': name, **receipt['archive']})
        roots = {'portable': portable, 'legacy24': base / 'data/acestep_grpo_offline_g10_24x4_20261003',
                 'legacy26': base / 'data/acestep_grpo_offline_g10_extension_26x4_20261003'}
        prompts = {name: json.loads((root / 'PROMPTS.json').read_text()) for name, root in roots.items()}
        for name, root in roots.items():
            assert policy_fingerprint(torch.load(root / 'behavior_policy.pt', map_location='cpu', weights_only=True)) == fp['g100' if name == 'portable' else 'g10']
        earlier_rows = [prompts[g['shard']][g['group']] for g in previous['groups']]
        records = {}
        for g in plan['fresh_audit_groups']:
            row = prompts[g['shard']][g['group']]
            assert row['split'] == 'train'
            reject_overlap([row], heldout + earlier_rows, description='v1 observations and heldout')
            folder = roots[g['shard']] / 'groups' / f"g{g['group']:06d}"
            metadata = json.loads((folder / 'group.json').read_text())
            assert metadata['closed'] and metadata['group'] == g['group'] and metadata['prompt_id'] == row['prompt_id']
            verify_members(folder, metadata['files'])
            records[g['key']] = folder
        new_rows = [prompts[g['shard']][g['group']] for g in plan['fresh_audit_groups']]
        for i, row in enumerate(new_rows):
            reject_overlap([row], new_rows[:i], description='another v2 audit group')
        write_once(output / 'INPUT_VERIFICATION.json', {'archives': verified, 'policy_fingerprints': fp,
            'model_checkpoint_sha256': {'g100': G100_SHA, 'g10': G10_SHA}, 'groups_verified': 24,
            'train_only_and_disjoint_from_v1': True, 'full_132GB_cache_verification_complete': False})
        state('loading_locked_receiver_and_verifying_native_controls')
        backend = create_backend(cfg); setup_policy(cfg, backend); backend.decoder.requires_grad_(False)
        assert model_signature(backend.provenance()) == model_signature(c100['boundary']['backend'])
        flags = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                 'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                 'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                 'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 'float32_matmul_precision': torch.get_float32_matmul_precision(),
                 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}
        earlier_flags = json.loads((previous_root / 'study/NATIVE_CONTROLS.json').read_text())['controls'][0]['replay_runtime']
        assert flags == earlier_flags, 'Receiver numerical flags changed after calibration'
        write_once(output / 'RECEIVER_FLAGS.json', flags)
        controls = []
        for b in ('g100', 'g10'):
            load_adapter_state(backend.decoder, policies[b])
            native = previous_root / 'study/native_controls'
            condition = torch.load(native / f'{b}_condition.pt', map_location='cpu', weights_only=True)
            record = torch.load(native / f'{b}_trajectory.pt', map_location='cpu', weights_only=True)
            controls.append({'behavior': b, **verify_replay(backend, condition, cfg, record, maximum_steps=50, tolerance=1e-6)})
            altered = dict(record); altered['variance'] = record['variance'].clone() * 1.01
            try:
                validate_record(altered, cfg)
            except ValueError:
                pass
            else:
                raise ValueError('Corrupted variance was not rejected')
        write_once(output / 'NATIVE_CONTROLS.json', controls)
        rows = []
        with (output / 'audit_measurements.jsonl').open('x') as stream, torch.no_grad():
            for b in ('g100', 'g10'):
                load_adapter_state(backend.decoder, policies[b])
                for g in plan['fresh_audit_groups']:
                    if g['behavior'] != b:
                        continue
                    folder = records[g['key']]
                    condition = torch.load(folder / 'condition.pt', map_location='cpu', weights_only=True)
                    for sample in range(4):
                        record = torch.load(folder / f's{sample:02d}_trajectory.pt', map_location='cpu', weights_only=True)
                        validate_record(record, cfg)
                        ids = torch.where(record['likelihood_mask'])[0].tolist()
                        assert ids == list(range(10, 41)), 'Do not omit or change stochastic times'
                        for i in ids:
                            x = record['states'][i].to(backend.device); action = record['states'][i+1].to(backend.device)
                            t, dt = float(record['timesteps'][i]), float(record['dt'][i])
                            velocity = backend.velocity(x, t, condition).float()
                            value = gaussian_log_prob(action, transition_mean(x, velocity, t, dt, cfg.sampling.noise_level), record['variance'][i].to(backend.device), cfg.sampling.logprob_reduction)
                            repeated_v = backend.velocity(x, t, condition).float()
                            repeated = gaussian_log_prob(action, transition_mean(x, repeated_v, t, dt, cfg.sampling.noise_level), record['variance'][i].to(backend.device), cfg.sampling.logprob_reduction)
                            row = {'key': g['key'], 'behavior': b, 'split': 'audit', 'sample': sample, 'step': i, 'time': t,
                                   'recorded_logprob': float(record['old_logprobs'][i]), 'replayed_logprob': float(value),
                                   'repeated_logprob': float(repeated), 'repeat_velocity_max_abs_error': float((velocity-repeated_v).abs().max())}
                            rows.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                    state('measuring_independent_v2_audit', group=g['key'], transitions=len(rows), **_memory(backend.device))
        result = evaluate_audit(rows, plan, controls_passed=True)
        corrupt = all(abs(r['replayed_logprob'] - r['recorded_logprob'] - .01) > .0003 for r in rows[::124])
        if not corrupt:
            result.update(audit_passed=False, status='reviewed_v2_independent_audit_failed')
        result.update(corrupted_logprob_controls_rejected=corrupt, native_controls=controls,
                      original_cached_probabilities_modified=False, numerical_flags_changed=False,
                      elapsed_seconds=time.monotonic()-start, **_memory(backend.device))
        write_once(output / 'SUMMARY.json', result)
        state('independent_v2_audit_complete', result_status=result['status'], audit_passed=result['audit_passed'])
        return result
    except Exception as error:
        write_once(output / 'FAILURE.json', {'error': str(error), 'traceback': traceback.format_exc(), 'formal_optimizer_updates': 0})
        state('independent_v2_audit_failed_preserved', error=str(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
        if backend is not None:
            del backend
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'authorization', 'previous-root', 'runtime', 'base', 'export', 'scratch', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute is required for real CUDA audit; no optimizer is implemented here')
    print(json.dumps(run(args.plan, args.authorization, args.previous_root, args.runtime, args.base, args.export, args.scratch, args.output), indent=2))


if __name__ == '__main__':
    main()
