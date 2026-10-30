"""Stop an impossible fixed-epoch target early, without relaxing any guard."""
import json
import hashlib
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/root/music-offline-grpo1000-v2-20261031-2128889')
WORK=Path('/mnt/ai-music-offline-grpo1000-v2-20261031-2128889')
sys.path.insert(0,str(ROOT/'source/src'))


def main():
    import torch
    from music_detector.rl.flow import group_advantages
    seen={};start=time.monotonic();quality_rejected=set();quality_receipt=None
    while time.monotonic()-start<48*3600:
        if (WORK/'PIPELINE_EXIT.json').exists():return
        folder=WORK/'scores/portable/groups'
        for path in sorted(folder.glob('g*.json')):
            if path.name in seen:continue
            value=json.loads(path.read_text());results=value['results']
            assert len(results)==4 and [r['sample'] for r in results]==list(range(4))
            advantages=group_advantages(torch.tensor([r['reward'] for r in results]),clip=5.0)
            signal_present=bool(any(r['valid'] for r in results) and advantages.abs().max()>1e-7)
            seen[path.name]={'path':str(path),'has_learning_signal':signal_present}
        complete=WORK/'QUALITY_BOUND_EXIT.json'
        if complete.exists() and quality_receipt is None:
            q=json.loads(complete.read_text());checks=WORK/'QUALITY_CHECKS.jsonl'
            rows=[json.loads(line) for line in checks.read_text().splitlines() if line.strip()]
            assert len(rows)==q['groups_checked']
            for row in rows:
                samples=row['candidate_checks'];assert len(samples)==4 and [s['sample'] for s in samples]==list(range(4))
                assert row['all_candidates_quality_rejected']==all(s['quality_rejected'] for s in samples)
                if row['all_candidates_quality_rejected']:quality_rejected.add(row['group'])
            assert quality_rejected==set(q['quality_rejected_groups'])
            quality_receipt={'path':str(checks),'sha256':hashlib.sha256(checks.read_bytes()).hexdigest(),
                'all_quality_rejected_groups':len(quality_rejected),'groups_checked':len(rows)}
        scored_rejected={Path(name).stem for name,v in seen.items() if not v['has_learning_signal']}
        combined=quality_rejected|scored_rejected
        bad=len(combined)
        if bad>50:
            # 1000 new + 50 legacy groups, one epoch. Even if every other
            # group were usable, >50 unusable new groups makes 1000 updates
            # impossible. A finished CPU proof can establish rejection before
            # expensive neural scoring; union identities, never sum overlaps.
            proof={'status':'fixed_single_epoch_target_mathematically_unreachable','scored_new_groups':len(seen),
                'unusable_new_groups':bad,'maximum_possible_signal_groups':1050-bad,'target_real_updates':1000,
                'full_score_no_signal_groups':len(scored_rejected),'quality_proof':quality_receipt,
                'distinct_proven_unusable_new_group_ids':sorted(combined),
                'max_epochs':1,'legacy_groups_assumed_all_usable_for_conservative_bound':50,
                'no_guard_or_reward_modified':True,'formal_optimizer_updates':0,'groups':list(seen.values())}
            with (WORK/'INSUFFICIENT_SIGNAL_UPPER_BOUND.json').open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
            active=json.loads((WORK/'ACTIVE_STAGE_PID.json').read_text())
            assert active['stage']=='score_portable'
            pid=active['child_pid'];command=Path(f'/proc/{pid}/cmdline').read_bytes()
            assert b'music_detector.rl.offline_cli' in command and str(ROOT/'data/portable_1000').encode() in command
            assert os.getpgid(pid)==pid
            os.killpg(pid,signal.SIGTERM)
            return
        time.sleep(15)


if __name__=='__main__':main()
