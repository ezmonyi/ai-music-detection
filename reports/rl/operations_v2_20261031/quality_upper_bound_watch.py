"""CPU-only conservative feasibility proof with the unchanged quality guards.

This is never a surrogate artifact scorer: passing quality means only possible
admission, while failing an unchanged pre-analyzer guard proves rejection.
No cached results, probabilities, options, or frozen scores are written here.
"""
from datetime import datetime, timezone
import collections
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/root/music-offline-grpo1000-v2-20261031-2128889')
WORK=Path('/mnt/ai-music-offline-grpo1000-v2-20261031-2128889')


def upper_bound(quality_rejected,scored_no_signal):
    return 1050-len(set(quality_rejected)|set(scored_no_signal))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=path.with_name(path.name+'.writing');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    assert not (WORK/'QUALITY_BOUND_EXIT.json').exists(),'Never repeat a completed CPU feasibility check'
    sys.path.insert(0,str(ROOT/'source/src'))
    from dataclasses import asdict
    import soundfile as sf
    import torch
    from music_detector.rl.config import config_from_dict
    from music_detector.rl.flow import group_advantages
    from music_detector.rl.rewards import ArtifactReward, _GateFailure
    source=ROOT/'source/src/music_detector/rl/rewards.py'
    assert sha(source)=='34537c343ac8a9582f71f264b018835f511f75818af6abe307beaaec3fc689b7','Original quality guard implementation changed'
    cfg=config_from_dict(json.loads((ROOT/'data/portable_1000/CONFIG.json').read_text()))
    def forbidden_analyzer(*args,**kwargs):raise AssertionError('A CPU quality proof must never invoke an analyzer')
    reward=ArtifactReward(cfg.reward.families,cfg.reward.bundle_sha256,device='cpu',guard=cfg.reward.guards,analyzer=forbidden_analyzer)
    checks=(WORK/'QUALITY_CHECKS.jsonl').open('x')
    start=time.monotonic();entries=[];bad=set();scored_bad=set();counts=collections.Counter()
    try:
        for folder in sorted((ROOT/'data/portable_1000/groups').glob('g*')):
            if (WORK/'PIPELINE_EXIT.json').exists():break
            rejections=[]
            for sample in range(4):
                c,sr=sf.read(folder/f's{sample:02d}_candidate.wav',dtype='float32',always_2d=True)
                b,bsr=sf.read(folder/f's{sample:02d}_base.wav',dtype='float32',always_2d=True)
                assert sr==bsr and type(sr) is int
                try:
                    candidate=reward._as_stereo(c,'candidate');base=reward._as_stereo(b,'baseline')
                    if candidate.shape!=base.shape:raise _GateFailure('shape_mismatch','candidate/base shapes differ')
                    reward._paired_gates(candidate,base,sr)
                    failure=None
                except _GateFailure as error:
                    failure={'code':error.code,'details':error.details,'message':str(error)};counts[error.code]+=1
                rejections.append({'sample':sample,'quality_rejected':failure is not None,'failure':failure})
            all_rejected=all(r['quality_rejected'] for r in rejections)
            if all_rejected:bad.add(folder.name)
            scored=WORK/'scores/portable/groups'/(folder.name+'.json')
            if scored.exists():
                full=json.loads(scored.read_text())['results']
                for cheap,actual in zip(rejections,full):
                    assert not (cheap['quality_rejected'] and actual['valid']),'CPU guard contradicts a finalized full score'
            entries.append({'group':folder.name,'all_candidates_quality_rejected':all_rejected,'candidate_checks':rejections,
                'source_group_sha256':sha(folder/'group.json')})
            checks.write(json.dumps(entries[-1],allow_nan=False)+'\n');checks.flush()
            for p in (WORK/'scores/portable/groups').glob('g*.json'):
                values=json.loads(p.read_text())['results']
                adv=group_advantages(torch.tensor([r['reward'] for r in values]),clip=5.)
                if not (any(r['valid'] for r in values) and adv.abs().max()>1e-7):scored_bad.add(p.stem)
            maximum=upper_bound(bad,scored_bad)
            state={'utc':datetime.now(timezone.utc).isoformat(),'node_id':2128889,'phase':'cpu_quality_only',
                'groups_checked':len(entries),'all_quality_rejected_groups':len(bad),'full_scored_no_signal_groups':len(scored_bad),
                'maximum_possible_signal_groups':maximum,'target_updates':1000,'legacy_assumed_all_50_usable':True,
                'neural_scoring_performed':False,'quality_guard_implementation_sha256':sha(source),
                'quality_guard_options':asdict(reward.guard),'elapsed_seconds':time.monotonic()-start,'rejection_reasons':dict(counts)}
            write(WORK/'QUALITY_BOUND_STATE.json',state)
            if len(entries)%25==0:print(json.dumps(state),flush=True)
            if maximum<1000:
                proof={**state,'phase':'single_epoch_1000_update_target_proven_impossible',
                    'quality_rejected_groups':sorted(bad),'full_scored_no_signal_group_ids':sorted(scored_bad),
                    'no_guard_reward_sampler_precision_or_input_modified':True,'formal_optimizer_updates':0,
                    'groups_checked_in_order':entries}
                write(WORK/'QUALITY_GUARD_UPDATE_UPPER_BOUND.json',proof)
                if not (WORK/'PIPELINE_EXIT.json').exists():
                    active=json.loads((WORK/'ACTIVE_STAGE_PID.json').read_text());assert active['stage']=='score_portable'
                    pid=active['child_pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                    assert b'music_detector.rl.offline_cli' in cmd and str(ROOT/'data/portable_1000').encode() in cmd
                    assert os.getpgid(pid)==pid
                    os.killpg(pid,signal.SIGTERM)
                write(WORK/'QUALITY_BOUND_EXIT.json',proof);return
        write(WORK/'QUALITY_BOUND_EXIT.json',{'phase':'cpu_quality_bound_complete_without_proven_impossibility',
            'groups_checked':len(entries),'quality_rejected_groups':sorted(bad),'possible_update_upper_bound':upper_bound(bad,scored_bad),
            'does_not_establish_full_artifact_admission':True,'neural_scoring_performed':False,
            'elapsed_seconds':time.monotonic()-start})
    except Exception as error:
        write(WORK/'QUALITY_BOUND_FAILURE.json',{'error_type':type(error).__name__,'error':str(error),
            'groups_checked':len(entries),'scientific_pipeline_not_stopped':True});raise
    finally:checks.close()


if __name__=='__main__':main()
