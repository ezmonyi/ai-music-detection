# ACE-Step online artifact-reward RL: experimental training repository

Status (2026-10-04): the A6000 v2 pilot completed 100 groups / 96 real updates,
50 test pairs and a separate 50-group frozen-policy cache. The node was released;
the test did not establish improvement. The original preparation plan below is
preserved separately. Current priority: [RTX 5060 collection and cloud delivery](rtx5060_collection.md),
with [bounded offline continuation](rl_offline_continuation.md) deferred until data
is ready. Actual 5060 fit, offline ACE training and LoRA/full parity are unverified.
See [the research review](rl_lora_vs_full.md) and [session handoff](SESSION_HANDOFF_OFFLINE_GRPO_20261004.md).

## Scope and boundaries

This extension uses the ACE-Step 1.5 **SFT 2B** flow/DiT decoder. VAE, text encoder,
condition encoder, tokenizer/detokenizer and all non-decoder weights stay frozen.
Both native PyTorch LoRA and full-decoder training use the same RL loop. This is
not the 8-step Turbo configuration used in the earlier generated music corpus.
The published ACE checkpoint is not advertised as an RL-ready DiT; this is a new
research adapter, not an official ACE training recipe.

The adapter targets the audited upstream revision
`ca1e85fe9430179831e6bc6be790c332190a3866`. The source must be pinned and all weights
must already exist locally. Training must not silently download weights or
substitute a different model. GPU feasibility and real audio admission remain to
be measured on the intended machine before starting a large run.

## Files

| Component | Location |
|---|---|
| Evidence and LoRA/full ablation rationale | `docs/rl_lora_vs_full.md` |
| Text-only prompt preparation, exclusions, split provenance | `src/music_detector/rl/data.py` |
| ACE local-checkpoint adapter and frozen reference | `src/music_detector/rl/acestep_backend.py` |
| Stochastic flow transitions and policy objective | `src/music_detector/rl/flow.py` |
| LoRA/full parameter selection and serialization | `src/music_detector/rl/policy.py` |
| Frozen artifact reward and admission checks | `src/music_detector/rl/rewards.py` |
| Online rollout, replay, checkpoint/resume, paired evaluation | `src/music_detector/rl/trainer.py` |
| TensorBoard events and monitoring semantics | `src/music_detector/rl/monitoring.py`, `docs/rl_tensorboard.md` |
| CLI | `src/music_detector/rl/cli.py` |
| CPU fixture / actual-GPU templates | `configs/rl/` |

## Environment

Use a separate Python 3.11/3.12 GPU environment. Do not mutate the detector web
service environment. For CPU-only tests, installing this repository with
`pip install -e '.[rl,test]'` adds PyTorch; it does **not** install the complete ACE
runtime or download checkpoints. The CPU tests were run with PyTorch 2.8.0.

For the real adapter, first install dependencies from the pinned ACE checkout in
that separate environment, then this repository. Upstream currently declares
Linux x86-64 PyTorch 2.10.0+cu128, Transformers >=4.51,<4.58 and Diffusers >=0.37;
use its pinned lockfile where available and record the resolved `pip freeze` and
CUDA/driver versions. These are upstream requirements, not a GPU environment
validated by this project. Do not treat the CPU environment as an ACE lockfile.

Expected local layout:

```text
ACE-Step-1.5/                         # audited git revision
  checkpoints/
    acestep-v15-sft/                 # model/config/weights + silence_latent.pt
    vae/                            # waveform VAE
    Qwen3-Embedding-0.6B/            # frozen text encoder and tokenizer
```

Replace both `/REPLACE/...` paths in a copied config. Inspect local checkpoint
provenance/license before use. All generated audio, local data and checkpoints go
under ignored `.rl-runs/`, `.rl-data/`, or a separate experiment volume. Do not
commit weights, credentials, raw music or licensed prompt text into the code repo.

## 500 text prompts, not 500 supervised audio targets

Online RL needs prompts, fresh model-generated trajectories and a scalar reward;
it does not require human audio as paired training targets. The default data
script makes **500 unique prompts total: 400 train / 50 validation / 50 test**.
It groups identical normalized captions and source identifiers before selecting
representatives. Splits are deterministic and independent of source row order.
Manifest and sidecar hashes record selection, exclusions, revision and license.
This removes exact/dependency overlap, not semantic near-duplicates or unknown
pretraining overlap.

Use existing, provenance-complete local metadata where possible:

```bash
python -m music_detector.rl.data \
  --input /path/to/prompts.jsonl \
  --exclude-manifest /path/to/historical_prompt_manifest.jsonl \
  --exclude-manifest /path/to/detector_heldout_manifest.jsonl \
  --output-dir .rl-data/music_prompts_500
```

Alternatively, explicitly fetch only the official MusicCaps caption CSV and
metadata, never its YouTube audio:

```bash
python -m music_detector.rl.data --musiccaps-hf \
  --exclude-manifest /path/to/historical_prompt_manifest.jsonl \
  --output-dir .rl-data/musiccaps_500
```

MusicCaps is pinned to revision `0a51889b340037bb75a9a0858af2e4ece21f7f89` and its
CC-BY-SA-4.0 metadata license is recorded. Source attribution and share-alike
obligations remain applicable. A caption of a source excerpt is used as a
generation prompt; `duration_s=30` is the requested output duration, not a claim
that the original excerpt was 30 seconds. No lyrics are invented. Empty lyrics
are allowed; this does **not** guarantee vocals. Supplied lyric line breaks and
vocal language are retained; absent language is `unknown`, not inferred English.
If rewarding the vocal-only S
family, first curate eligible vocal prompts and audit admission rates.

Historical/held-out exclusions are **required for a scientific run**, but the
script cannot discover missing manifests for you. Without them, the outputs
must not be described as external held-out data. This initial code delivery
includes only authored toy prompts, not a newly cleared 500-prompt experiment.
The repository preserves the old Muse manifest builder and its recorded frozen
manifest hash `bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b`,
but not the manifest itself. Retrieve that manifest from the archived project
before claiming historical prompt disjointness. Historical `source_song_id`,
`source_track_id`, `acestep_caption`, `lyrics_30s`, and `source_license` aliases
are supported by the preparation/exclusion path.

## CPU end-to-end check

Run from the repository root, using an installed package or `PYTHONPATH=src`:

```bash
python -m music_detector.rl.cli train \
  --config configs/rl/toy_smoke.json --data configs/rl/toy_train.jsonl \
  --output .rl-runs/toy

python -m music_detector.rl.cli evaluate \
  --config configs/rl/toy_smoke.json --data configs/rl/toy_validation.jsonl \
  --checkpoint .rl-runs/toy/checkpoints/group_000003.pt \
  --output .rl-runs/toy-eval
```

The tiny synthetic backend is deliberately labeled as non-music evidence.
It exercises actual nonzero gradients and optimizer updates, not just imports.

## Real-GPU pilot and resume

```bash
python -m music_detector.rl.cli validate --config /path/to/local_lora.json
python -m music_detector.rl.cli train \
  --config /path/to/local_lora.json --data .rl-data/music_prompts_500/train.jsonl \
  --output .rl-runs/lora-pilot --max-updates 1
```

`validate` checks configuration only. A one-group pilot is the first real test of
loading, conditioning, waveform decoding, admission, gradient flow and peak VRAM.
It generates four candidate and four paired base clips (each 30 seconds, 50
steps); it is not a memory-free preflight. The VAE and text encoder remain loaded.
No rented machine is provisioned by this repository.

Resume the same experiment config into a **new directory**, preserving all prior
artifacts:

```bash
python -m music_detector.rl.cli train \
  --config /path/to/local_lora.json --data .rl-data/music_prompts_500/train.jsonl \
  --resume .rl-runs/lora-pilot/checkpoints/group_000001.pt \
  --output .rl-runs/lora-continuation
```

Config, data, RL source hashes, backend provenance and runtime boundary must
match. `--max-updates`
limits the number of additional prompt groups without changing the experiment's
full group budget. Checkpoints store trainable weights, optimizer, RNG, completed
groups and update counts. Only load trusted checkpoints; loading uses PyTorch's
restricted `weights_only=True`. Full-decoder checkpoints are large.

## Sampling and objective

For the ACE path `x_t = t noise + (1-t) data`, the decoder predicts
`v_t = noise - data`; integration runs from t=1 to t=0. For negative `dt`, use

```text
sigma² = eta² t/(1-t)
mu = x * (1 + sigma² dt/(2t)) + v * (1 + sigma²(1-t)/(2t)) * dt
x_next ~ Normal(mu, sigma² * (-dt) I)
```

Only the interior window [0.2,0.8] is stochastic. The prefix/suffix and endpoint
steps are Euler ODE steps and have no Gaussian policy likelihood. State and
density arithmetic use float32. Each prompt produces four distinct seeded
trajectories sequentially; four seeded stochastic timesteps per trajectory are
retained on CPU for replay. Conditioning is computed once per prompt/group.

The reward is `-tanh(frozen_raw_detector_score / score_scale)` after admission.
Group-centered/std-normalized advantages drive the clipped policy surrogate;
equal-variance Gaussian KL penalizes drift from the frozen original decoder.
Sampled next states and old log densities are detached. The VAE and reward are
not differentiated. Same-seed base audio is used for quality gates and diagnostic
delta, **not** as the sole training reward: all paired deltas would be zero at
initialization and prevent learning.

The first implementation makes **one on-policy replay pass and one optimizer
step per group**. Its pre-update ratios should be one; the PPO clipping term is
present but does not provide multi-epoch PPO reuse. This is an intentionally
small starting protocol, not a reproduction of every Flow-GRPO hyperparameter.
`logprob_reduction=mean` is the official-style dimension-normalized surrogate,
not a joint high-dimensional importance ratio. `sum` is supported but changes
scaling and needs retuning. Never mix these in one comparison.

CFG is fixed to 1. ACE's default APG guidance has inter-step momentum and cannot
be substituted without storing/replaying that state. No Heun, velocity clipping,
EMA, repaint, LM prompt rewriting or hidden sample postprocessing is enabled.

## Reward admission and interpretation

Default GPU templates use **F + SC** as a lower-cost pilot reward: phase residual
and stereo statistics. **F is not the thesis's high-frequency vocal statistic**;
that is S. To study that hypothesis, use a separate config with, for example,
`"families": ["S", "F", "SC"]`, install the exact released Demucs/runtime assets,
and use vocal-eligible prompts. S/D/R/P require their original frozen neural
extractors; no cheaper DSP substitute is silently used. H is the chroma-path
feature family. BC remains a research-only diagnostic and is rejected as a
predictive reward family.

All selected features must be complete, finite and eligible. Bundle, extractor
and asset hashes pin the reward; changes require an explicit new experiment.
Gates check native stereo, duration, finite samples, silence, clipping, bandwidth,
stereo collapse and paired RMS/activity drift. Thresholds are **engineering
starting values**, not validated judgments of artistic quality. Do not normalize,
duplicate channels or low-pass audio to pass these checks. All-invalid groups
stop with diagnostics; zero-advantage groups skip the update and are counted.

These guards cannot prove prompt/lyric adherence or prevent every reward hack.
Inspect paired saved audio and monitor non-reward feature families, independent
quality/adherence metrics, output diversity, VAE reconstruction error and blinded
listening. A lower AI-source detector score is not proof of a human origin or
improved perceived music. Human reference recordings belong in evaluation, not
automatically in the online prompt optimization set.

## LoRA versus full comparison

```bash
python -m music_detector.rl.cli ablation-plan \
  --config /path/to/local_lora.json --output .rl-runs/ablation-plan
```

This writes **45 planned configs, launches none**: all-linear LoRA ranks 8/32/128,
attention-only rank 32, and full decoder; three LR candidates and three seeds each.
All-linear includes the decoder's other linear projections as well as attention
and MLP; exact module paths and parameter counts are saved. Alpha/r stays at 2.
It does not adapt Conv1d/ConvTranspose1d or free scale/shift parameters. Full
decoder trains those too: this is a coverage/capacity comparison, not a pure
rank-only comparison. Those differences are deliberate and must be reported.

Keep data, solver, groups, steps, stochastic seeds and reward fixed; tune each
arm's LR on validation only. Test only after selection. Compare at matched
candidate rollout/group budgets, and separately at measured GPU-hours. The
100-group default is a pilot (400 candidates), not one pass over 500 prompts and
not a convergence claim. Base-reference rollouts double generated clip count.

`model.precision` specifies compute precision. Full mode uses fp32 trainable
tensors/Adam states with autocast compute; LoRA has fp32 adapters and frozen
base tensors. Actual trainable/frozen dtypes are recorded in the manifest.
Full mode stores a frozen CPU reference and caches a matching-dtype reference
copy on the GPU (roughly another 8 GB for a 2B fp32 decoder), in addition to the
trainable model. Its memory cost differs from LoRA;
do not infer a throughput multiplier from trainable parameter count alone.
Profile before choosing rental hardware. Full decoder cannot eliminate a frozen
VAE's intrinsic reconstruction ceiling.

## Outputs and evidence to preserve

Each training directory contains config/manifest hashes, exact parameter/module
counts, per-sample reward diagnostics/seeds, group metrics, timing and peak-memory
measurements, periodic paired FLOAT WAVs, checkpoints and summary/failure files.
The failure log preserves the last completed group; resume only from a completed
checkpoint. CUDA allocated/reserved figures are PyTorch process measurements;
separate device used/free/total figures include other processes and CUDA state.
Process peaks reset at the start of each training invocation, including resume.
TensorBoard logging is enabled in ACE templates; detailed setup, exact scalar
definitions and a no-GPU preview are in [rl_tensorboard.md](rl_tensorboard.md).

Evaluation defaults to same-seed paired **ODE** generation on validation/test,
with optional `--stochastic` to examine the training sampler. It rejects training
caption/source overlap, retains paired audio, and reports valid counts and mean
raw-score delta. This is a minimal evaluation harness: bootstrap confidence
intervals, source-stratified analysis and blinded listening remain required for
publication-grade conclusions. No experiment result is added to the thesis until
those real runs are completed and reviewed.

## Additional frozen-policy offline cache (2026-10-03)

The originally approved cache was **300 unused training prompts × 4 candidates**,
not 1,200 independent prompts and not a replacement for the active online run.
Select sorted train records `[100:400]`; reject overlap with the online first
100 records and with validation/test prompt IDs, normalized captions or sources.
Freeze the online v2 group-10 behavior-policy adapter. No optimizer is created.
Keep the existing base model, S+R+F reward, sampler and admission thresholds.
The standalone worker uses a copied runtime and does not mutate online source
files. Its CUDA allocator ceiling is 25% of device capacity, with GPU/disk reserve
checks. These controls do not themselves guarantee contention-free execution.

Each candidate stores 51 unique FP32 latent states for the 50-step solver,
timesteps, transition variances, dimension-mean old log densities, a likelihood
mask, cached conditioning, caption/seed/provenance, reward/feature diagnostics,
and candidate/base native FLOAT WAVs. Deterministic transitions are masked, not
represented as valid Gaussian policy actions. Invalid samples and all-invalid
groups remain present. First-two-group replay checks use unchanged behavior
weights and a 1e-6 log-density tolerance.

The two-group real-GPU probe completed: eight candidates, eight paired bases,
all eight admitted; the replayed log-density maximum discrepancy was zero.
That is an implementation check, not evidence of perceptual improvement.
That original full collection did not start: a verified direct Drive upload
route was unavailable. The user subsequently replaced it with the smaller
persistent-disk collection described below; a proxy is explicitly not authorized.
At 30 seconds/48 kHz/stereo, 1,200 paired candidates with complete trajectories
contain approximately 39.4 GB of raw audio/latent payload. Budget 49–55 GB,
excluding a second archive copy and environment snapshots. Actual probe files
were 11,520,088 bytes/WAV and 9,796,301 bytes/full trajectory.

Components: `offline_collect.py`, `offline_trajectory.py`, `offline_io.py` for
collection; `offline_archive.py`, `offline_drive_store.py`, `offline_upload.py`
for task-rooted immutable backup. Use a dedicated Drive profile rooted at the
approved new dataset folder, mode 0600, never commit its OAuth token. The uploader
packs only checksum-verified closed groups into 10-group shards; it never syncs
or deletes other Drive contents. Payload transfers verify Drive size and MD5,
while retaining source/member SHA-256 manifests. Tiny route and final receipt
files additionally undergo actual remote SHA-256 readback. This distinction is
explicit: full-payload remote SHA-256 readback is **not** claimed. Temporary
credentials and large staging files must be excluded from the environment snap.

```bash
PYTHONPATH=src python -m music_detector.rl.offline_collect \
  --config /path/to/unchanged_online_config.json \
  --train-data /path/to/train.jsonl \
  --validation-data /path/to/validation.jsonl --test-data /path/to/test.jsonl \
  --checkpoint /path/to/group_000010.pt --checkpoint-sha256 EXACT_SHA256 \
  --offset 100 --count 300 --max-groups 2 --output /path/to/new_cache
```

After a real upload-route round trip, continue with identical arguments and
`--resume` instead of `--max-groups 2`; complete-prefix file checks must pass.
Do not repeatedly reuse a fixed-policy cache and call it standard online GRPO:
the official [Flow-GRPO loop](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3.py)
recollects trajectories across outer iterations. This cache can support replay
checks and separately declared off-policy experiments; no offline RL training
has yet been run.

For final online results, `scripts/download_rl_pilot_results.py` can copy the
exact completed-package member allowlist using resumable rsync and verify every
local file's bytes/SHA-256 without holding both a local archive and extraction.
Its receipt explicitly does not claim that the archive was locally downloaded.
Backup, completed snap and exact-node stopped-billing verification are separate
release gates; an upload attempt or a checkpoint alone is not completion.

### Revised persistent-storage scope (2026-10-03, 16:31 China)

The user declined a temporary local VPN tunnel and requested a smaller cache on
the existing persistent disk. The active replacement is **24 new unused training
prompts × 4 candidates**: 96 full FP32 trajectories and 192 paired native FLOAT
WAVs, using sorted train[102:126]. The original two probe groups (train[100:102])
remain separately preserved; they are not counted as newly generated samples.
Online train[0:100], validation and test remain excluded. The original 300-group
plan must not be reported as completed.

[`scripts/run_matpool_offline_persistent.py`](../scripts/run_matpool_offline_persistent.py)
is a dated, task-specific controller with explicit `/mnt` paths, not a generic
launcher. It retains the same frozen online-v2 group-10 behavior policy, online
configuration, sampler, reward, guards and collector implementation. It has no
offline optimizer. No Drive upload or proxy is needed. Collection started;
completion is established only by its final persistent verification receipt.

The launch gate reserved 4,104,339,435 bytes for new data (including a 20% allowance),
7,469,425,556 bytes for worst-case remaining online outputs at group 65,
26,000,000,000 bytes for the environment snapshot, and a 4 GiB safety reserve.
The disk reported 43,578,818,560 bytes free against 41,868,732,287 bytes required.
These are launch observations and capacity estimates, not measured final payload
or snapshot sizes. The controller validates closed-group identity, exact member
coverage, sizes and SHA-256 after the worker exits successfully; all-invalid
groups remain in the data. Final local backup and confirmed environment snapshot
are still required before stopping the exact rented node.

### Completed first shard and latest 50-group target (2026-10-03)

The 24-group shard completed with 96 full FP32 trajectories, 192 WAVs,
59 admitted and 37 retained invalid candidates. All 347 files (3,341,891,948
bytes) passed size/SHA-256 verification and a second full read. The first two
new groups recorded 32 replay checks with zero maximum log-probability error.
First/last trajectory spot checks have shape `[51,1,750,64]`, 50 solver steps,
and 31 masked-in stochastic likelihood steps. This is a frozen-policy cache,
not offline optimizer training or evidence of improved perceptual quality.

The user's latest instruction supersedes a proposed 100-group expansion:
**50 cumulative prompt groups, without expanding storage**. Preserve the first
24 groups and append 26 distinct groups from sorted train[126:152] in a separate
directory. The main cache target is 200 trajectories/400 WAVs. Including the
separate original two-group probe, retention will total 52 groups, 208
trajectories and 416 WAVs; those totals are not completion claims.

[`scripts/run_matpool_offline_extension.py`](../scripts/run_matpool_offline_extension.py)
pins the original controller source hash, validates the measured serialization
contract, never overwrites the first shard, and verifies both shards again
before writing the cumulative 50-group receipt. The additional data budget is
3,740,106,327 bytes: measured fixed-size audio/FP32 trajectories, 3 MB condition
and 400 KB group-metadata allowances, a complete adapter copy and 64 MiB global
metadata. This replaces the pre-probe blanket 20% data allowance; the separate
26 GB snapshot estimate and 4 GiB overall safety reserve are unchanged. Actual
closed test WAVs already occupying the disk are not counted a second time as
future output. Snapshot size remains an estimate until the provider confirms it.

Forty collection, backup, controller and extension tests passed locally and on
the GPU host using CPU fixtures. The append worker started only after the live
capacity gate passed. Completion still requires the cumulative verification,
full exact-member local backup, snapshot confirmation and exact-node stop.
