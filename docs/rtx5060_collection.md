# RTX 5060: collect and deliver the next offline GRPO cache

This is the current delivery priority (2026-10-04): **prepare the separate
RTX 5060 for data collection and cloud-disk upload**. Cloud disk expansion and
the later training session are deferred. No machine is rented or restarted by
these commands, and no local VPN/Clash proxy is enabled.

The code has synthetic CPU regression coverage. A real RTX 5060, its VRAM fit,
throughput, and 5060-to-A6000 likelihood replay have **not** been tested yet.
Do not launch 1,000 groups before the explicit native two-group probe succeeds.

## What is collected

| Quantity | Default |
|---|---|
| Behavior policy | Online **v2 group 100**, frozen LoRA; not failed v1 |
| Model | ACE-Step 1.5 non-Turbo SFT 2B; frozen VAE/text/condition encoders |
| Prompt budget | 1,000 unique, previously unused training prompts |
| Candidates | Four sequential candidates per prompt: 4,000 trajectories |
| Audio | 4,000 candidate + 4,000 same-seed original-SFT base WAVs |
| Sampling | Native 30 s, 48 kHz stereo; 50 flow steps; noise 0.7, stochastic t=0.2–0.8, shift 1, CFG 1 |
| Trajectories | All 51 FP32 latent states, original transition log densities/variances, FP64 schedule, deterministic-step mask |
| Local optimizer updates | **Zero** |
| Local reward analysis | None; immutable native WAVs are scored later by frozen cloud S+R+F |
| Storage estimate | About 140–150 decimal GB of cache; models/environment/export staging are additional |

1,000 prompt groups are **not a promise of 1,000 optimizer updates**. Invalid,
zero-advantage or stale groups may be skipped after cloud scoring. The later
training preflight counts signal-bearing groups and refuses an insufficient
budget. Do not fabricate lyrics, normalize/clip WAV peaks, discard bad outputs,
quantize weights or change the sampler to make a probe pass.

Component staging puts the text encoder, condition/flow model and VAE on CUDA
in separate phases. Inactive modules reside on host RAM; their forward passes
are still CUDA. BF16 frozen weights and FP32 LoRA parameters retain their dtype.
No optimizer or neural artifact analyzer shares the 5060. Linux x86-64 / WSL2
with Python 3.12 is the supported collection target; native Windows is not
validated (the existing trainer imports the Unix `resource` module).

## 1. Restore inputs and install the collector

Clone the delivered branch on the **5060 machine**, not this Mac:

```sh
git clone --branch codex/acestep-online-rl https://github.com/ezmonyi/ai-music-detection.git
cd ai-music-detection
python3.12 -m venv .venv-collector
.venv-collector/bin/python -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv-collector/bin/python -m pip install -r requirements/rl_collector_5060.txt
.venv-collector/bin/python -m pip install -e '.[test]'
```

These runtime pins match the observed A6000 collector stack. The pinned
[ACE source dependencies](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/pyproject.toml)
also distinguish Linux and Windows CUDA wheels. This requirements subset is
not a fully resolved historical lockfile. Record the actual `pip freeze`,
driver and GPU model on the new machine; the doctor rejects different core
versions. Do not install the unrelated full ACE UI/LM stack over this environment.

Use a clean ACE source checkout at
`ca1e85fe9430179831e6bc6be790c332190a3866`. Importing that source does not download
weights or use `trust_remote_code=True`.

Download these existing inputs from **MatPool My Disk → Region 1 →
ai-music-rl-20261003**, using the account's web/client access:

- The **whole** `checkpoints/` directory, including hidden/auxiliary files in
  `acestep-v15-sft/`, `vae/` and `Qwen3-Embedding-0.6B/`.
- `runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt` (508,991,837 bytes).
- The original validation/test JSONL files under
  `data/musiccaps_a6000_curated_500_20261003/`.
- Optionally the old offline shards' `PROMPTS.json` exclusion lists. Do not
  download their large trajectories to the 5060 merely to exclude their prompts.

The mandatory group-100 SHA-256 is:

```text
c28a4502247cb4736ec1f5b20322381192baef877d3a377188e7ee9a35b2ccd2
```

The collector verifies the base/VAE/text **original full directory identities**.
The recorded SFT tree had seven files while the public core-file receipt lists
four SFT files. Therefore downloading only core weights is not equivalent to
restoring the original directory. A mismatch must be investigated, not bypassed
by editing the checkpoint/manifest. The optional helper verifies 15 public
core files totaling 6,336,577,076 bytes, via hf-mirror with no proxy or official-HF
fallback; it does **not** claim to reconstruct the full historical tree:

```sh
.venv-collector/bin/python scripts/prepare_rtx5060_assets.py --directory /data/music-rl/checkpoints
# Explicit download, only if you need the core files; prefer restoring the whole original directories.
.venv-collector/bin/python scripts/prepare_rtx5060_assets.py --directory /data/music-rl/checkpoints --execute
```

No weights, checkpoint binaries, licensed prompt text or credentials are in Git.

## 2. Prepare more training prompts without touching the fixed held-out files

The old checkpoint's **entire historical training pool** is conservatively
excluded, as are normalized caption/source matches and validation/test prompts.
Use provenance-complete local metadata, or pinned MusicCaps text metadata.
MusicCaps does not supply lyrics; keep them empty. Its caption licensing is
not a license to redistribute the underlying YouTube audio.

Example metadata-only preparation through a direct hf-mirror request:

```sh
curl --noproxy '*' --fail --location --output /data/music-rl/musiccaps-public.csv \
  https://hf-mirror.com/datasets/google/MusicCaps/resolve/0a51889b340037bb75a9a0858af2e4ece21f7f89/musiccaps-public.csv
.venv-collector/bin/python -m music_detector.rl.data \
  --input /data/music-rl/musiccaps-public.csv \
  --source-dataset google/MusicCaps --source-revision 0a51889b340037bb75a9a0858af2e4ece21f7f89 \
  --license cc-by-sa-4.0 --total 2000 --train-count 2000 --validation-count 0 --test-count 0 \
  --seed 1104 --output-dir /data/music-rl/new-training-pool
```

This makes a larger candidate training pool, **not replacement held-out data**.
Supply the restored original JSONL paths to the plan command below. Selection
and exclusion counts are validated; if fewer than 1,000 unused prompts remain,
prepare additional metadata rather than weakening exclusions.

```sh
.venv-collector/bin/python -m music_detector.rl.offline_cli plan \
  --checkpoint /data/music-rl/runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt \
  --checkpoint-sha256 c28a4502247cb4736ec1f5b20322381192baef877d3a377188e7ee9a35b2ccd2 \
  --train-data /data/music-rl/new-training-pool/train.jsonl \
  --validation-data /data/music-rl/data/musiccaps_a6000_curated_500_20261003/validation.jsonl \
  --test-data /data/music-rl/data/musiccaps_a6000_curated_500_20261003/test.jsonl \
  --count 1000 --output /data/music-rl/plan-5060-1000
```

Repeated `--exclude-prompts /path/to/old-shard/PROMPTS.json` is available. The
entire old pool exclusion already covers the original cloud 50-group selection.
Keep the plan and original held-out files alongside the cache for the cloud session.

## 3. Probe first, then collect with the unchanged policy

Copy `configs/rl/rtx5060_runtime.example.json` and
`configs/rl/rtx5060_collection.example.json` into your private experiment folder.
Replace every `/REPLACE/` path. The session's `runtime` path is resolved relative
to the session JSON. Keep `state_dir` and export staging outside the raw cache.
Leave `upload.remote_base` empty until an actual upload route is configured.

```sh
.venv-collector/bin/python -m music_detector.rl.collector_cli session \
  --config /data/music-rl/collector-session.json --phase probe
.venv-collector/bin/python -m music_detector.rl.collector_cli session \
  --config /data/music-rl/collector-session.json --phase doctor --execute
.venv-collector/bin/python -m music_detector.rl.collector_cli session \
  --config /data/music-rl/collector-session.json --phase probe --execute
```

The first command is a no-write/no-network/no-GPU dry run. The doctor checks
software, CUDA/BF16, paths and disk estimates, but is not a full-model fit test.
The probe collects two **real native** groups (eight trajectories/16 WAVs),
checks recorded likelihood replay, and logs actual bytes, elapsed time and
peak allocated/reserved GPU memory in `GPU_ADMISSION.json`. Reserved VRAM must
remain below 95% of device total. OOM or replay failure preserves the failed
prefix; do not relabel it as an admission pass.

If an A6000/cloud machine is available later, transfer the two-group probe and
run `music-offline-rl replay` there before scaling. Cross-GPU kernel differences
may exceed the frozen 1e-6 tolerance. This has not been measured. Do not rent a
machine solely for this optional check without authorization.

After admission, the full command resumes the existing two groups; it does not
regenerate them:

```sh
.venv-collector/bin/python -m music_detector.rl.collector_cli session \
  --config /data/music-rl/collector-session.json --phase collect --execute
```

The same command resumes a stopped batch and refuses concurrent writers.
Closed members are rehashed before reuse. Partial groups are moved to an
`incomplete/` quarantine instead of overwriting successful groups. Changing
code, core runtime, hardware receipt, plan or sampler requires a separate cache.
Changing runtime paths does not permit changed weights or precision. For an
interrupted initial probe, preserve the folder and review the failure before
using the lower-level explicit `collect --resume` entry point.

Monitor `collection_state.json` and `collector-state/SESSION_STATE.json`, plus
`nvidia-smi`. ETA should be based on the actual probe/batch timings, not the old
A6000 throughput. This process has no train loss: it performs zero updates.
The two probe groups count toward the 1,000-group budget.

## 4. Verify and upload without restarting the old GPU node

```sh
.venv-collector/bin/python -m music_detector.rl.collector_cli verify \
  --dataset /data/music-rl/rtx5060-g100-1000x4
```

Only a finalized `FILES_SHA256.json` plus all 1,000 closed groups is a complete
cache. Pending reward/validity fields are null, **not zero or a validity pass**.
The original FLOAT WAV values, including invalid outputs, are preserved.

### MatPool web/client, no running rented node

The [MatPool client](https://matpool.com/supports/doc-use-matbox-on-matpool/)
supports folder transfer. Upload the complete cache folder and collection plan
into a **new dedicated Region 1 subfolder**, not over the old 20261003 dataset.
Client transfer completion is not a SHA-256 verification receipt.

For web upload or manageable batches, export verified TARs (25 groups is about
3.5–3.75 GB, subject to actual contents):

```sh
.venv-collector/bin/python -m music_detector.rl.collector_cli export \
  --dataset /data/music-rl/rtx5060-g100-1000x4 \
  --output /data/music-rl/upload-export --groups-per-shard 25
```

This intentionally creates another copy; reserve extra staging space. You can
export closed ranges with `--first 0 --last 25`, then `--first 25 --last 50`, etc.
Use a fixed, non-overlapping partition; retain every archive, its receipt and
the final `EXPORT_INDEX.json`. The metadata/completion TARs carry identity and
the original global file manifest. An early probe can use a separate export
directory with `--last 2 --groups-per-shard 2`. Do not mix overlapping probe and
production shard layouts in one export directory.

On a future cloud machine, verify and restore a full export into a **fresh** path:

```sh
python -m music_detector.rl.collector_cli restore \
  --source /mnt/new-5060-upload/export --output /mnt/new-5060-upload/restored-cache
```

For a two-group replay-only import, add `--allow-prefix`. It deliberately remains
an incomplete cache and cannot be merged/scored for training as a completed set.
Full restore hashes every TAR before extraction, hashes every member as written,
then checks the original dataset manifest. Unsafe paths/symlinks/duplicates are
rejected. Never delete 5060 originals based only on the browser's progress bar;
retain them until a real destination verification has been recorded.

### Optional configured rclone remote

This is a generic direct upload route, **not a MatPool web-disk API**. MatPool
[SFTP/SCP access requires a running instance](https://matpool.com/learn/article/matbox-connect-to-matpool-server/).
The old qKEVx1 address is inactive. An already configured direct remote to another
authorized storage service also works. Securely configure it outside Git;
for SFTP, verify its host key and use a known-hosts file. Do not put passwords,
keys or OAuth tokens into session JSON, shell commands or this repository.

```sh
.venv-collector/bin/python -m music_detector.rl.collector_cli upload \
  --dataset /data/music-rl/rtx5060-g100-1000x4 \
  --remote configured:dedicated/ai-music-offline \
  --receipts /data/music-rl/upload-receipts
# Add --execute only after reviewing the dry run and configuring the route.
```

The uploader uses non-deleting [rclone copy](https://rclone.org/commands/rclone_copy/)
with `--immutable`, an exact member list and isolated receipts. It removes proxy
environment variables for subprocesses; it neither toggles global Clash settings
nor defeats an OS-level transparent tunnel. Connectivity must actually be direct.

Default verification streams back **every** remote member and compares source
bytes/SHA-256 without saving a second local copy. This adds roughly one full
cache's download traffic. Explicit `--verification server_sha256` instead
requires a complete server-provided SHA-256 listing; unsupported hashes fail
without a size-only fallback. That receipt clearly states no full-payload
readback. Failed copying/checking never deletes raw files or certifies a backup.
Rerunning the same destination/receipt directory resumes immutable file copies;
the final verification is repeated. No archive copy is required for rclone.

## Acceptance and next handoff

Collection is delivered when 1,000 distinct groups, 4,000 trajectories and
8,000 WAVs are closed, their full source SHA-256 manifest verifies, and the
upload route has either a real full-member verification receipt or a clearly
identified **pending cloud verification** status. Export success alone is not
cloud upload success. The cloud disk will be expanded manually by the human.

The later merge/offline-training code is described in
[offline continuation](rl_offline_continuation.md). The current conversation
handoff is [here](SESSION_HANDOFF_OFFLINE_GRPO_20261004.md). Do not restart the
old node, old heartbeat, 300-group collector, Drive/VPN tunnel or online run.
