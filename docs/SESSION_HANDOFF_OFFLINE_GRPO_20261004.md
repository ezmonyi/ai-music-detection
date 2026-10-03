# Session handoff: RTX 5060 collection first, offline continuation later

Prepared 2026-10-04, Asia/Shanghai. This is a code/data-lineage handoff, not a
claim that the new 5060 cache or offline ACE training has already run.

## Current human instruction

The human will manually expand the cloud disk later. **Set that issue aside;
deliver the repository to the separate RTX 5060 machine for data collection and
cloud-disk upload.** No new GPU rental, disk expansion, automatic cloud training,
Clash/VPN tunnel or old heartbeat is authorized by this delivery.

All current local updates belong under:

```text
/Users/yi/Documents/projects/music/
```

The former `/Users/yi/Documents/code/music/` directory was moved. Its path still
appears in dated evidence; those historical receipts have not been rewritten.
Git worktree pointers were repaired to the new locations. Active checkout:

```text
/Users/yi/Documents/projects/music/.worktrees/acestep-online-rl
branch: codex/acestep-online-rl
repository: https://github.com/ezmonyi/ai-music-detection
```

Use [the 5060 instructions](rtx5060_collection.md) first. The deferred
[offline continuation instructions](rl_offline_continuation.md) explain how
new 5060 data and the existing cloud cache are combined. Do not confuse a
prompt group, a WAV file, a trajectory and a real optimizer update.

## Completed A6000 experiment and its limits

Online ACE-Step 1.5 non-Turbo SFT 2B, LoRA rank 32/all-linear, frozen VAE/text/
condition encoders; reward S+R+F with unchanged bundle/admission guards.
Native 30-second/48-kHz-stereo generation; 50 flow steps; noise 0.7 in t=0.2–0.8;
seed 11; LR 3e-5; clipping 0.2; beta 0.01 under exact-transition KL.

- V2: 100 training prompt groups, **96 actual optimizer updates**; four completely
  invalid groups skipped. Training admitted 249/400 candidates (62.25%). The
  failed v1 and intermediate results are preserved, not overwritten or used as
  the continuation checkpoint.
- Final test: 50 same-prompt/seed pairs, all 100 WAVs retained. 35 pairs were
  admitted. Candidate-minus-base raw score: mean **+0.0168780737**, median
  −0.0009834332; 18 lower/17 higher; bootstrap 95% CI [−0.02506,+0.06102],
  paired t p≈0.458. Lower was desired: average improvement was **not established**.
- A 70% admission rate is not AI/human classification accuracy. No validated
  perceptual improvement or reliable vocal-artifact reduction is claimed.
  Generated candidate outputs are still AI-generated.
- TensorBoard audit: 40 train/eval scalar tags, 3,370 values matched JSON logs;
  gradients were finite and genuine updates occurred. Near-zero on-policy actor
  loss at ratio≈1 does not mean zero learning.
- Only 112/249 admitted train candidates were vocal-eligible. Most within-group
  reward variance separated valid/invalid outputs. Do not hide this by silently
  modifying the frozen S+R+F reward or guard.
- Exact-transition KL uses sigma²·|dt|; pinned upstream regularization uses sigma².
  At dt=0.02 the numerical KL scale differs by 50. Beta 0.01 is not an upstream
  beta 0.01. The causal contribution of KL versus actor gradients was not logged.
- The old 50 test prompts have already been inspected. Future repeated evaluation
  is a longitudinal comparison, not a pristine blind generalization test.

Local historical evidence (small reports/manifests/logs, not the full dataset):

```text
/Users/yi/Documents/projects/music/rl_results/acestep_a6000_srf_20261003/
  FINAL_CLOSEOUT_20261003.txt
  FINAL_CLOSEOUT_20261003.json
  audit_20261003/AUDIT_REPORT_20261003.txt
  audit_20261003/AUDIT_NUMBERS_20261003.json
  audit_20261003/ALL_TENSORBOARD_METRICS_20261003.txt
  audit_20261003/TRAIN_EVAL_AUDIT_20261003.png
  audit_20261003/raw/
```

The operational historical runbook is:
`.rl-runs/matpool-setup-20261003/RUNBOOK.json` in the active worktree. Its latest
`closeout` supersedes older nested “pending” observations. Do not publish private
operational files or credentials from that ignored directory.

## Authoritative cloud inputs and node state

**MatPool My Disk → Region 1 → ai-music-rl-20261003**. Former mount:
`/mnt/ai-music-rl-20261003`. Directory names below are relative to that folder.

| Input | Relative path / identity |
|---|---|
| Initial policy for new 5060 cache and later training | `runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt` |
| Old cloud-cache behavior policy | `runs/a6000-srf-pilot-main-v2/checkpoints/group_000010.pt` |
| Immutable cloud first24 | `data/acestep_grpo_offline_g10_24x4_20261003` |
| Immutable cloud extension26 | `data/acestep_grpo_offline_g10_extension_26x4_20261003` |
| Original curated 500 text prompts / held-out files | `data/musiccaps_a6000_curated_500_20261003` |
| Base model/VAE/text trees | `checkpoints/` (restore the whole directories) |
| Old 50 evaluation pairs | `runs/a6000-srf-final-test-v2` |
| Persistent original-member proof | `release-package/PERSISTENT_MEMBERS_VERIFICATION.json` |

Group-100 SHA-256 (508,991,837 bytes):

```text
c28a4502247cb4736ec1f5b20322381192baef877d3a377188e7ee9a35b2ccd2
```

**V2** group-10 SHA-256 (508,991,837 bytes):

```text
909c003d33c9a150c6ac947969ccf139f4f4ae52d9a1d8ab5f8e36b1958047e4
```

Do not substitute `runs/a6000-srf-pilot-main/checkpoints/group_000010.pt`: it is
the failed **v1** policy with a different SHA.

Original validation SHA: `d6a0839cc2c69f5f8f83ece5e37fa31d7956d0f591773f8645d3296b325589ef`.
Original test SHA: `d9090e509eacca21ff64decc945fe31a197dd97d0edc47011b1e694cc52a1406`.

Cloud main offline dataset: **50 groups / 200 full FP32 trajectories / 400 WAVs**,
6,949,573,569 bytes, 722 files; 123 valid/77 invalid candidates, one wholly invalid
group. Frozen **v2 group-10** behavior, **zero offline optimizer updates**. The
original two-group probe is separate: total including it is 52 groups, 208
trajectories, 416 WAVs. Do not call the superseded 100/300-group plans complete.

Cloud backup proof covers 1,584 original experiment members / 21,793,613,504
bytes, verified on 2026-10-03 11:24:16 UTC. Original member manifest SHA:
`222a502e59e77a9855e806f1af8d6eeb5806dc6ef256635f61773768d62f6428`.
The duplicate 14-GB archive was removed before snapshot; its old creation
receipt does not mean that archive still exists. Original run members remain.

Snapshot **music-rl-acestep15-srf-20261003**, environment ID **227273**, Region 1;
file **2107364_1791027231.snap**, displayed **23.2G**, saved 2026-10-03 19:49:35.
It protects the runtime outside `/mnt`, not the separately retained disk data.
Save completion was visibly verified; a snapshot restore test and payload SHA
readback were **not** performed.

Only **qKEVx1** was stopped/released at provider time **2026-10-03 19:54** and
compute billing stopped. Its former SSH endpoint is not an active training
machine. The RL heartbeat is paused. Do not restart either automatically.
Observed disk-plan expiry was 2026-11-02; confirm current retention with the human.
The old 55-GB disk cannot hold the new cache, but expansion is now explicitly a
**later manual human task**, not a current agent action.

Bulk local checkpoint/music downloads were removed after remote verification,
at the human's request. Old local-verification timestamps do not imply those
large files still exist on this Mac. The frozen MyDisk originals are the source.
Post-stop supplemental reports/screenshots were retained locally; their cloud
upload was not confirmed. Google Drive upload of the offline cache was not
completed and no direct/VPN route-ready receipt exists.

## Delivered code and evidence boundaries

- `collector_cli.py`: no-action default, Linux/WSL2 doctor, explicit native
  two-group GPU probe, OS-released collection lock, 25-group checkpointed slices.
- `portable_collect.py` / `staged_backend.py`: group-100 frozen-policy collection,
  sequential candidates, native CUDA component staging, full FP32 records,
  same-device likelihood replay, immutable closed prefix and recoverable quarantine.
- `portable_delivery.py`: exact-member source verification, non-overlapping TAR
  exports, safe hash-checked full/prefix restoration, explicit configured-rclone
  upload. Default full remote readback versus separately identified server SHA.
  No local-data deletion, remote sync/purge, inferred cloud API or proxy activation.
- `scripts/prepare_rtx5060_assets.py`: optional explicit hf-mirror public-core
  downloader with bytes/SHA checks and owned-partial resume. It does not rebuild
  the complete historical tree: restore the original `checkpoints/` instead.
- `portable_replay.py`: sampled cloud likelihood replay of a two-group probe;
  no scoring, optimizer or fabricated real-GPU status.
- `offline_cache.py`: exact legacy/new cache verification, isolated frozen scoring
  sidecars, no-overlap merge and distinct g10/g100 behavior binding.
- `offline_train.py`: CPU-tested bounded off-policy continuation, warm start from
  g100, fresh offline AdamW, actual-update accounting, strict offline resume,
  drift rejection, TensorBoard monitoring and evaluation triplets. **Deferred**.
- `configs/rl/rtx5060_*`, `requirements/rl_collector_5060.txt`,
  `manifests/rl_5060_model_inputs.json`: reviewable templates/pins, no credentials.
- `tests/test_rl_offline_continuation.py`: synthetic CPU tests, not audio-quality
  evidence or a real 5060/A6000 offline performance benchmark.

Native training/collector source is pinned ACE commit
`ca1e85fe9430179831e6bc6be790c332190a3866`; Flow-GRPO reference
`879042cf5707f8b90daa98d147d7deac2317c5da`. Original runtime commit was
`574cdc86ddf052d782e516cb039a22aca9357570`; prior backup-tool head
`d0d2989188a500c9464cb1cd5400248921447d86`. Record the delivered Git commit from
the checkout, not the old runtime commit. Do not rewrite historical outputs.

## Next session: decisions and safe order

1. Obtain access to the **separate 5060** and confirm Linux/WSL2, driver, VRAM,
   RAM and local disk. This Mac is not that machine; no access exists in this
   handoff. Do not assume that the old released A6000 can be SSHed into.
2. Restore the original component folders, v2 group-100 policy, old held-out JSONL
   and prepare provenance-complete new training metadata. Keep lyrics empty for
   MusicCaps; exclude the entire historical pool and held-out caption/source IDs.
3. Copy private session/runtime JSON, run dry-run → doctor → real two-group probe.
   Confirm unchanged sampler, native WAVs, all-member hashes and recorded VRAM.
   Preserve an OOM/failure; no invented admission receipt or quantized substitute.
4. If an authorized cloud machine becomes available, export/import the small
   probe and use `music-offline-rl replay` before the expensive full collection.
   Otherwise explicitly record that cross-GPU replay remains untested. Never
   replace cached old probabilities or silently loosen tolerance to pass it.
5. Collect the fixed 1,000 groups, with the probe included, retaining every invalid
   candidate. Verify the finalized cache. The data can be collected before the
   human expands cloud space; do not pretend upload succeeded while it is full.
6. Upload to a new private dedicated cloud-disk folder. MatPool web/client works
   without a rented GPU; SFTP requires an actual running instance. Ask for the
   intended storage route/configuration when missing, not for an old SSH password.
   No local Clash/VPN use is allowed. Retain 5060 data until real cloud proof.
7. Only later: restore the saved cloud environment, frozen Torch-2.8 analyzer and
   Torch-2.10 trainer, score new WAVs, merge old24+26 and new data, preflight the
   actual learning signal and replay every behavior policy before optimization.
   Target 1,000 **additional real** updates; a shortfall is a failure to meet that
   target, not permission to loop stale data or weaken guards.
8. Later test triplets compare SFT base, online g100 and offline candidate. Preserve
   all results/SHAs and use a new frozen blind set for generalization claims.
   Future node backup/snapshot/shutdown needs that future session's authorization;
   do not shut down the human's dedicated 5060 without explicit authorization.

## Copyable opening prompt for the next agent

> Read this handoff and docs/rtx5060_collection.md completely. The immediate task
> is to prepare my separate RTX 5060, run the unchanged native two-group collection
> probe, then collect the frozen v2-group100 offline cache and upload it to my
> confirmed cloud-disk folder. Cloud expansion is a later manual task. Use the
> supplied runtime/access details; do not rent machines, start the released
> qKEVx1, enable any local VPN/Clash proxy, change sampler/reward/guards, or launch
> cloud parameter training. Start with source/checkpoint/held-out SHA checks,
> doctor and measured VRAM/probability replay. Keep invalid outputs and originals;
> record incomplete cloud verification honestly. When data is ready, hand off the
> verified cache, plan, runtime, logs and receipts for a separate offline training
> session using the existing cloud 50 groups plus new data.
