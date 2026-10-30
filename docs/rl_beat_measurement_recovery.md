# Recovery after the 2026-10-31 BeatThis scoring failure

The run ended after 211 completed scoring groups. BeatThis raised
`ValueError: Not all downbeats are beats.` while exporting a candidate's beat
and downbeat arrays. `analyzer_worker.py` exited nonzero; the strict subprocess
bridge converted that exit into `RuntimeError`, so it bypassed the existing
`ArtifactReward.score` input-invalid `ValueError` handling and terminated the
entire scoring stage. This happened before any new optimizer update.

The completed 211 group scores, every archived output and all original input
TARs were preserved. The paid node has been released. No new lease is authorized
by this document and no new GPU/model validation was performed after release.

## Prepared opt-in adapter

`src/music_detector/rl/measurement_rejections.py` provides an optional factory:

```python
from music_detector.rl.measurement_rejections import (
    POLICY, create_measurement_safe_analyzer,
)
from music_detector.rl.rewards import ArtifactReward

# cfg must already contain the permitted runtime relocations.
analyzer = create_measurement_safe_analyzer(cfg.reward, policy=POLICY)
reward = ArtifactReward(
    cfg.reward.families,
    cfg.reward.bundle_sha256,
    device=cfg.reward.device,
    guard=cfg.reward.guards,
    analyzer=analyzer,
)
```

This factory is **not enabled in the existing CLI**. A future recovery driver
must explicitly inject `reward` into the existing `reward=` arguments of
`score_shard`, `train_offline` and `evaluate_offline`, including validation.
Do not monkeypatch the original default or alter its fatal-error contract.
Record `analyzer.error_policy_receipt` in a separate immutable sidecar and
include that sidecar in the recovery/resume identity. No worker hash or legacy
receipt may be edited to pretend a different worker is the original one.

The adapter verifies the unchanged original worker and bridge file hashes,
recognizes only the observed error's exact final exception and expected worker,
rhythm and BeatThis stack frames, and then requests a fresh status receipt from
the original worker. A changed or unavailable runtime remains fatal. When all
checks pass, the adapter raises a typed `ValueError`, allowing the existing
invalid-reward path to reject the measurement. It fabricates no feature, beat,
downbeat, score or valid flag. Successful measurements are passed through as
the original object. Other subprocess, CUDA, protocol and missing-weight errors
remain fatal. Original worker/bridge/extractor code and strict default behavior
are unchanged.

Twelve standard-library regression tests passed, and the classifier recognized
the actual archived traceback. This is CPU evidence for the adapter contract;
it does not establish that the failed audio reproduces on GPU, that all new
groups are usable, or that the RL target has completed.

## Next authorized recovery session

1. Obtain fresh compute authorization. Do not reconnect the released node,
   restart its heartbeat, or silently rent another machine.
2. Restore the original environment and unchanged cache from the persistent
   source TARs. Keep the failed run, old controller exit and its 211 scoring
   groups immutable. Validate source/member SHA-256 and held-out identities.
3. On the new receiver, repeat required numerical/model admission. Reproduce
   the failing group (the next group after the 211 closed groups) using the
   original analyzer, then confirm the explicit adapter returns invalid without
   changing a good control's measured features. Exercise a fatal negative
   control as well. The exact candidate was not identified in the old log;
   preserve per-candidate progress before each new call.
4. Create a new recovery score/output namespace and immutable error-policy
   sidecar. The 211 completed scores can be reused only after source checksums,
   original analyzer identity and unchanged successful-measurement semantics
   are established. Resume from the unclosed group, rather than overwriting
   the failure record or starting completed scoring again.
5. Wire the same explicit reward adapter into training validation and final
   evaluation. Perform full eligibility preflight before two real updates and
   same-checkpoint continuation to the fixed 1,000-update target. Keep all
   invalid candidates and all existing numerical/drift/quality gates.
6. Give recoverable failures a bounded diagnosis/recovery window. The previous
   supervisor released immediately on any terminal stage error, before the
   agent could distinguish a recoverable measurement error. That policy was
   too aggressive for this case. The future owner must preserve failure evidence,
   permit a verified recovery on the same paid node when possible, and retain
   a bounded protected-release fallback so a failed job cannot idle overnight.
7. Only after actual successful training generate the fixed 50 held-out official
   SFT/new-RL pairs, verify local delivery, and protect artifacts before release.

The current run remains **0 new updates, 0 new RL checkpoints, 0 requested test
pairs**. A passing numerical audit and prepared recovery code must not be
reported as completed RL or improved perceived musical realism.
