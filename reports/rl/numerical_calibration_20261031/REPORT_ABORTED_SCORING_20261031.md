# Offline GRPO stopped before training: scoring failure and protected release

The current RTX 3090 run did **not** complete its 1,000-update objective.
It completed 211 of 1,000 new prompt-group scores, then stopped on a BeatThis
measurement exception. There were **0 additional optimizer updates, 0 new RL
checkpoints and 0 new official/new-RL test pairs**. The paid instance was
released after backup verification; an independent official read confirmed its
released status and stopped compute billing. The official final bill is CNY9.86.

## What passed before the failure

The user-reviewed v2 numerical audit passed all 2,976 independent transitions
under its fixed `3e-4` receiver budget, with maximum absolute log-density error
`1.02996826171875e-4`. Repeated/native controls retained `1e-6` and measured zero
error. This earlier numerical admission was successful; the later failure is
in frozen rhythm measurement, before policy optimization.

Full restoration verified 42 original TARs, 132,352,450,560 bytes and every
restored member: 1,000 new prompt groups, 4,000 full FP32 trajectories and 8,000
candidate/base WAVs. The additional older 50-group cache remained available.
The full-cache CPU quality check finished without invoking a neural analyzer.

| Evidence at termination | Observed result |
| --- | ---: |
| Fully scored new groups | 211 / 1,000 |
| Candidates in completed scoring groups | 844 |
| Admitted candidates in that prefix | 516 / 844 (61.14%) |
| All-invalid groups in the scored prefix | 8 |
| Full-cache quality-only checked groups | 1,000 |
| Groups rejected by quality guards for all four candidates | 24 |
| Conservative possible signal-group upper bound, including 50 older groups | 1,025 |
| Full merge/eligibility preflight completed | No |
| Additional optimizer updates | 0 / 1,000 |
| New test WAVs | 0 / 100 requested |

The 1,025 upper bound assumes every other new group and all 50 older groups
are usable. It does **not** demonstrate that 1,000 actual updates are feasible.
It also does not prove insufficient data. The run ended because of a technical
measurement failure, rather than the mathematical feasibility-stop rule.
The 211-group prefix is ordered and incomplete; its admission fraction is not
an unbiased estimate of the complete cache or an AI/human discrimination result.

Invalid candidates in that prefix were recorded as follows:

| Recorded rejection | Candidates |
| --- | ---: |
| Clipping peak | 187 |
| Insufficient high-band energy (`lowpass`) | 84 |
| Baseline ineligible | 44 |
| Candidate analysis invalid | 9 |
| Baseline analysis invalid | 3 |
| Stereo collapse | 1 |
| Total | 328 |

## Exact failure and propagation

After group 211 completed, the next candidate analysis reached:

```text
music_detector.neural._track_beats
  -> beat_this.utils.save_beat_tsv
  -> beat_this.utils.infer_beat_numbers
ValueError: Not all downbeats are beats.
```

This states that the exporter's downbeat-subset invariant was not satisfied.
The captured trace does not establish the exact candidate, numerical offset or
underlying cause of the inconsistent beat arrays; those require a controlled
reproduction of the cached audio. The next cache group is the immediate
reproduction target. No timing snapping, synthetic beats or valid features
were introduced to work around the error.

The original subprocess bridge raises `RuntimeError` for every nonzero worker
exit. `ArtifactReward.score` handles ordinary measurement `ValueError` as an
invalid completion, but that original exception type was lost at the process
boundary. The wrapped `RuntimeError` terminated `score_portable` with exit 1.
The preserved pipeline terminal record reports this failure and zero updates.

The closeout supervisor then archived the terminal outputs and released the
instance immediately. This protected storage and stopped charges, but its
policy was too aggressive: it did not allow the agent to diagnose and repair
this recoverable input-error propagation before release. A future supervisor
needs a bounded recovery window, with a protected-release fallback. The
experiment's unfinished scientific target must remain visible.

## Preservation and resource closeout

- All 314 archived output members were reverified by SHA-256 locally, in addition
  to the cloud archive member readback and local transport checks.
- The work archive is 1,415,656 bytes, with SHA-256
  `eede88700c03f3b906bd3061a8042ee72428c2120114208cf77f17f4ba4341f6`.
- Source input TARs and cloud products remain on the persistent volume. The
  132 GB source dataset was not downloaded to the Mac.
- Official node state is released (status 4). Two stable billing observations
  and another independent official read confirmed compute billing stopped.
  The final official bill is CNY9.86. No second rental, recharge or expansion
  was performed.
- Existing environment snapshots 227273 and 229669 are both completed/enabled
  in the final provider read. No new snapshot of the 132 GB scratch duplicate
  was created; the environment dependencies were unchanged.
- The heartbeat was deleted. Checkpoint/mirror watchers and the temporary idle
  sleep assertion ended; the empty local TensorBoard server was also stopped.

## Prepared recovery and remaining work

`src/music_detector/rl/measurement_rejections.py` adds a **non-default, opt-in**
adapter for precisely the observed BeatThis error. It pins the unchanged worker
and bridge sources, checks the complete expected traceback and requests a fresh
original-worker status receipt before returning a typed input rejection. It
fabricates no feature or valid score. Other worker, CUDA, setup and protocol
failures remain fatal, and successful measurements pass through unchanged.

Twelve CPU-only contract regressions passed and the matcher recognized the
actual saved traceback. The adapter was not used in this failed run and has
not been validated on the failing audio/GPU. It is not enabled in the existing
CLI. See `docs/rl_beat_measurement_recovery.md` for explicit integration,
immutable-policy sidecars and control requirements.

Remaining scientific work is a correctly scoped recovery on newly authorized
compute, full frozen scoring/eligibility preflight, two real updates plus the
same-checkpoint continuation to 1,000, and the fixed 50-prompt official/new-RL
audio comparison. No additional paid instance was started after this release.
No RL improvement, detector evasion or reduction in AI-like listening quality
is claimed from the present run.

## References

- Numerical admission: `REPORT_INDEPENDENT_V2_AUDIT_20261031.md` and `v2/` raw
  audit evidence; scientific additions `3fd46e8` and `8857c3e`.
- Executed monitoring/closeout source archive: commits `4dea346` and `fe2741d`,
  `reports/rl/operations_v2_20261031/`.
- Primary scoring failure: `diagnostics/score_portable.log`, extracted from the
  SHA-verified work archive; `SCORING_FAILURE_ANALYSIS.json` and
  `PREFIX_GROUP_SCORES.csv` contain the derived prefix statistics.
- Local complete packet:
  `/Users/yi/Documents/projects/music/rl_results/offline_grpo1000_v2_20261031_2128889`.
- Local transport and member receipts: `OUTPUT_ARCHIVES_VERIFICATION.json`,
  `REMOTE_DELIVERY_MANIFEST.json`, `REMOTE_DELIVERY_ACK.json` and `FINAL_CLOSEOUT.json`.
- Persistent cloud root:
  `/mnt/ai-music-offline-grpo1000-v2-20261031-2128889`.
- Persistent original inputs: `/mnt/upload-export-20261009` and
  `/mnt/ai-music-rl-20261003`.
- LaTeX text: `thesis_aborted_scoring_appendix.tex`.

The detailed postmortem and optional recovery source were produced locally and
versioned after release. The cloud volume already contains the original terminal
report and complete failed-run evidence; this later postmortem was not uploaded
to an inaccessible released node.
