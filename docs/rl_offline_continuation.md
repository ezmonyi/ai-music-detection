# Bounded offline Flow-GRPO continuation (deferred cloud experiment)

The 2026-10-04 priority is the [RTX 5060 collector](rtx5060_collection.md), not
cloud provisioning or training. These entry points are implemented and tested
with a synthetic CPU model, but no real offline ACE optimizer run or cross-GPU
replay is claimed. The old online pilot and offline-cache collection are separate
completed experiments.

## Inputs and policy lineage

Initialize from online **v2 group 100** (96 real updates). The original cloud
50-group cache was collected using **v2 group 10**, whereas new 5060 data uses
frozen group 100. Both original behavior checkpoints must be supplied when
merging. Each transition keeps its own recorded behavior log density: group-10
data is never mislabeled as group-100 data and no old likelihood is overwritten.

The unchanged experiment contract pins source weights, LoRA rank/targets,
native sampler, S+R+F bundle/guards and held-out files. Runtime paths are portable;
the historical online trainer's exact-resume rules remain unchanged. New 5060
WAVs are scored on the frozen cloud analyzer (separate Torch 2.8 environment),
with sidecars outside the immutable raw cache.

Example commands on an authorized future cloud machine after restoring the
saved environment and current code. Replace the new-upload path; the existing
cloud paths below are the recorded **data/** paths, not TensorBoard run names.

```sh
python -m music_detector.rl.offline_cli replay \
  --checkpoint /mnt/ai-music-rl-20261003/runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt \
  --checkpoint-sha256 c28a4502247cb4736ec1f5b20322381192baef877d3a377188e7ee9a35b2ccd2 \
  --shard /mnt/new-5060-upload/restored-cache --max-groups 2 \
  --runtime configs/rl/offline_cloud_runtime.example.json --output /mnt/new-5060-upload/replay.json

python -m music_detector.rl.offline_cli score \
  --shard /mnt/new-5060-upload/restored-cache --output /mnt/new-5060-upload/scores \
  --runtime configs/rl/offline_cloud_runtime.example.json

python -m music_detector.rl.offline_cli merge \
  --checkpoint /mnt/ai-music-rl-20261003/runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt \
  --checkpoint-sha256 c28a4502247cb4736ec1f5b20322381192baef877d3a377188e7ee9a35b2ccd2 \
  --behavior-checkpoint /mnt/ai-music-rl-20261003/runs/a6000-srf-pilot-main-v2/checkpoints/group_000010.pt \
  --shard /mnt/ai-music-rl-20261003/data/acestep_grpo_offline_g10_24x4_20261003 \
  --shard /mnt/ai-music-rl-20261003/data/acestep_grpo_offline_g10_extension_26x4_20261003 \
  --shard /mnt/new-5060-upload/restored-cache \
  --score /mnt/new-5060-upload/scores/SCORES.json \
  --validation-data /mnt/ai-music-rl-20261003/data/musiccaps_a6000_curated_500_20261003/validation.jsonl \
  --test-data /mnt/ai-music-rl-20261003/data/musiccaps_a6000_curated_500_20261003/test.jsonl \
  --output /mnt/new-5060-upload/merged-index.json
```

Do not merge the old two-group GPU probe by default; the main cloud cache is 50
groups. Failed candidates stay in each complete four-candidate group. A whole
invalid group is recorded but not counted as an optimizer update.

## Objective and boundaries

The pinned [upstream Flow-GRPO loop](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py)
refreshes trajectories online. Reusing this frozen cache is an explicitly
**off-policy research experiment**, not standard on-policy GRPO.

For each group, standardize its four frozen rewards, use up to four stochastic
transitions per candidate, replay that group's behavior policy and verify the
selected old densities within 1e-6. Restore the current policy, then optimize
the clipped policy-ratio surrogate plus original-SFT reference KL. Ratios use
dimension-mean log density, as in the existing implementation; they are **not
exact full-trajectory importance weights**. All deterministic steps are masked.

Default options in `configs/rl/offline_1000.json` request **1,000 additional real
updates**, one epoch, AdamW LR 3e-5, clip 0.2, advantage clip 5, gradient clip 1.
The offline start has a fresh AdamW optimizer because this is a different
objective; it is a weight warm start, not exact continuation of online AdamW.
Only subsequent offline `--resume` restores that offline optimizer/RNG/cursor.
CPU fixtures verify sliced/resumed policy and optimizer equality bit for bit.

Groups are rejected without an optimizer step if max absolute log-ratio >5,
clip fraction >0.8, normalized surrogate ESS <0.2, or behavior KL >0.1. After
25 consecutive skips the run fails with a preserved checkpoint/diagnostics.
These thresholds are conservative implementation defaults, **not validated
offline-RL guarantees**. Do not loosen them silently to reach 1,000 updates.
The one-epoch signal count is an upper bound: drift can still cause shortfall.

`exact_transition` KL retains the old pilot denominator
`2 * sigma_t^2 * abs(dt)`. At 50 equal steps this is numerically 50 times the
pinned upstream `2 * sigma_t^2` regularizer. Beta 0.01 therefore is not a portable
upstream beta. The explicit `upstream_sigma` option is a separate declared
ablation; it must not be selected using the old test results or disguised as a
bug fix. No old sampler/reward/guard was modified for this delivery.

## Preflight, smoke slice, training and monitoring

Run `preflight` with the merged index, original initialization SHA, fixed
validation/test files and options. It rehashes every member, checks provenance,
no duplicate/held-out overlap, behavior identities and available signal. This
can read about 150 GB and is intentional. An insufficient cache raises an error.

Use the same arguments for `train`, adding a fresh `--output`, the copied cloud
runtime JSON and optionally `--max-updates 2` for the first real-update slice.
Resume that slice's `checkpoints/final.pt` into a **fresh** output directory with
`--resume`; keep the same 1,000-update options/index/code/runtime boundary.
Those two updates are included in the 1,000 target, not extra discarded training.
Missing TensorBoard fails before GPU model loading. Insufficient usable data
has a non-success CLI exit status, not a fabricated completion.

Actual events cover train loss, reference/behavior KL, grad norm, learning rate,
clip fraction, surrogate ESS, validity, skips, optimizer updates, throughput and
CUDA allocated/reserved/peak/free/used memory when available. Validation runs
every 50 real updates. `eval/loss` is negative admitted frozen reward, **not
policy loss**. Unavailable metrics remain absent rather than fake zeros.

Start TensorBoard on loopback after a real run begins; do not expose an
unauthenticated public port. Its run folder is `<fresh-output>/tensorboard/run-*`.
Keep JSONL logs and pipeline states as authoritative status sources. No new
monitoring heartbeat or rented node is launched by the code.

## Evaluation

`evaluate` accepts the original initialization checkpoint/SHA, continued
checkpoint/SHA and explicit held-out JSONL. It saves same-seed ODE audio
triplets: original SFT `base`, online group-100 `initial`, and continued
`candidate`. Report candidate-minus-initial and candidate-minus-base metrics,
all admission/invalid rates and the joint-admitted selection-conditioned subset.
Do not report generated outputs as human music or detector evasion as improved
perceptual quality.

The old 50 test prompts have already been examined. A repeat is a longitudinal
comparison, **not a fresh blind test**. Use a separately frozen new test set for
a new generalization claim; never tune rewards/checkpoints using test outcomes.
Preserve native audio, every checkpoint, logs, score sidecars and SHA-256 receipts
before any future node-release action. Disk expansion/rental/release requires
the human's future session context; none occurs in this deliverable.
