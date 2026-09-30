# Interpretable Music Artifact Analysis

A local audio-upload application and reproducible research code for a thesis
on source-labelled AI-generated versus human music. Acoustic evidence is an
association with a research dataset, not proof of a musician's authorship.

## Start the application

Use Python 3.12 (the verified host uses 3.12.14):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/music-detector serve --port 8765
```

Open http://127.0.0.1:8765. Choose another port if one is already occupied.
The service binds only to loopback, rejects cross-origin uploads, limits
uploads to 100 MiB, deletes each temporary audio file after the job, and retains
results in memory for at most one hour. It does not upload your audio elsewhere.

Input contract: native stereo WAV/FLAC/MP3/OGG, 30–3600 seconds. The detector
uses the floor-centred 30 seconds at the native sample rate, then 44.1 kHz
stereo FLOAT32. Mono duplication, short-file padding and loudness normalization
are deliberately excluded. Historical selected-region replay additionally
requires the pinned region hash. MP3 decoding may differ across platforms.

F, H, SC and optional research-only BC work without neural weights.
S/D/R/P use the original analysis models, not DSP substitutes:

| ID | Measurement | Required frontend |
|---|---|---|
| S | Bias-corrected vocal high-frequency spectrum and sibilance | Demucs + frozen MUSDB response correction |
| D | Nonvocal dynamics | Demucs bass/drums/other stems |
| R | Beat-interval and tempo variation | Beat This! final0 |
| P | Section durations and lengths in bars | All-In-One boundaries + Beat This downbeats |
| F | Phase residual and group-delay evolution | Fixed STFT descriptors |
| H | Chroma/pitch-class path | Fixed chroma descriptors, not harmonic overtones |
| SC | Bandwise stereo level/side-energy statistics | Preserved stereo mix |
| BC | Fixed 500/750/1250 Hz bicoherence | Research measurement only; never affects probability |

Analysis dependencies and weights are installed only through an explicit setup:

```sh
.venv/bin/python -m music_detector.neural setup --install-dependencies --download-weights
.venv/bin/python -m music_detector.neural status
```

This restores analysis frontends, not music-generation models. The status check
and runtime verify pinned files. Apple Silicon uses CPU; CUDA can be selected
with `serve --device cuda`. No silent MPS/CPU fallback. CPU and historical CUDA
precision differ and must not be described as bit-identical neural inference.
Real CPU S/D/R/P inference has been checked on one AI-labelled ACE-Step and
one human-labelled MAESTRO development recording. The upload UI also completed
all seven predictive families plus optional BC on a real recording. These are
implementation checks, not a new external accuracy benchmark; archived and
fresh neural values differ under hardware/precision and stochastic shifts.

The interface exposes actual values, raw-score contributions, missingness
contributions, probability calibration scope, warnings and downloadable JSON.
A missing measurement is null, not zero. Invalid, silent or constant-DC audio
is rejected; when every selected predictive metric is missing, the system
abstains instead of producing a missingness-only probability.

## What the probability means

The historical classifier is weighted ridge (alpha 10), with an unbounded
identity-link score and a fixed 0.5 threshold. It was not a probability.

The deployment bundle is a separate, retrospective experiment: 4,228
development recordings, a metadata-first dependency-closure split into
2,549 train / 839 calibration / 840 internal-test recordings, and monotonic
Platt calibration fitted only on the calibration partition. All 127 nonempty
combinations of S/D/R/P/F/H/SC have their own model. BC is excluded.

| Predeclared combination | Internal weighted BA | AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| F+H+SC | 0.746070 | 0.812212 | 0.177601 | 0.044323 |
| S+D+R+P | 0.752909 | 0.829170 | 0.167656 | 0.050872 |
| All seven | 0.772188 | 0.864156 | 0.149388 | 0.063683 |

These are **internal reference-mixture results, not unknown-generator accuracy**.
No combination was chosen on this test table. P is fully observed on only
524/4,228 recordings; its missingness is a serious interpretability limitation.
The probability is calibrated to a balanced class/source/group research
mixture, not the prevalence of AI music on a streaming service.

The browser check deliberately retains a counterexample:
human-labelled MAESTRO sample 0124 receives p(AI)=0.716227 with F+H+SC.
Reproducibility does not imply a correct classification. Do not use this
application as sole evidence to accuse or penalize a musician.

## Reproduction from Google Drive

Authoritative project folder:
[ai_music_artifact](https://drive.google.com/drive/u/0/folders/1ow8f0RzHLYEpOaBaa1hH1bIoJ8dHDQcU).

Access is private and rights remain source-specific. Manifests identify exact
Drive objects, TAR member offsets and hashes; audio and credentials are not
distributed in this repository. Use your existing configured rclone remote
(`--rclone-remote NAME`, no colon), or securely set
`MUSIC_DETECTOR_DRIVE_TOKEN`. Never commit OAuth tokens or use them in URLs.

### 1. Fixed audio-sample replay

```sh
.venv/bin/python -m music_detector.reproduce \
  --audio-dir .reproduction/audio --download --rclone-remote YOUR_REMOTE \
  --report .reproduction/sample_replay.json
```

This downloads only eight known members, approximately 22 MB, using authenticated
HTTP byte ranges rather than whole 16–20 GB archives. Current archive metadata,
source SHA-256, waveform hashes and all 27 F/H/SC values are checked.

Observed validation: all eight source hashes match; four FLAC waveform/feature
replays pass. Four MP3 replays have 8–11 metrics outside the predeclared strict
tolerance (rtol=1e-7, atol=1e-8). Across seven combinations and eight samples,
56 decisions are unchanged; maximum probability difference is 2.886e-6.
These development examples test infrastructure, not classifier accuracy.

### 2. Refit every deployment combination from pinned Drive features

```sh
.venv/bin/python -m music_detector.fetch \
  --manifest manifests/reproduction_inputs.json \
  --destination .reproduction/inputs --rclone-remote YOUR_REMOTE

.venv/bin/python -m music_detector.calibration train \
  --metadata .reproduction/inputs/metadata.json \
  --features .reproduction/inputs/features.json \
  --screen .reproduction/inputs/screen.json \
  --planner current/audio_phenomena_expansion_20260907/code/plan_native30_evaluation_schedule_v1.py \
  --protocol manifests/calibration_protocol.json \
  --bundle .reproduction/refit/deployment_models.json \
  --report-dir .reproduction/refit/report

.venv/bin/python scripts/verify_refit.py \
  --original src/music_detector/assets/deployment_models.json \
  --refit .reproduction/refit/deployment_models.json \
  --protocol manifests/calibration_protocol.json \
  --features .reproduction/inputs/features.json \
  --output .reproduction/refit_comparison.json
```

The verified fresh Drive refit preserved all 106,680 internal-test decisions.
Maximum raw-score error was 1.183e-14 and probability error 2.647e-9; the model
files were not byte-identical. The protocol refuses changed code/input hashes
and never reads locked YuE2 labels. This is feature-level refitting, not a
claim of fresh whole-corpus neural inference.

### 3. Historical report and model replay

- `validation/scoring_replay.json`: two archived models, 828 test recordings
  each; maximum error below 9e-16, no changed historical decisions.
- `validation/native60_replay/run_replay.sh`: exact historical Native60
  prediction-table aggregation, 876,300 predictions / 9,525 summaries /
  1,905 report cells. CSV and LaTeX outputs match historical bytes.
- `validation/native60_replay/ACCEPTANCE.json`: complete scope and hashes.
  This re-aggregates frozen predictions; it does not rerun audio inference.
- Historical source-holdout and generator-holdout conclusions are separate
  from the new deployment table above.

Commands refuse to overwrite existing verification reports. Choose a fresh
output path on rerun. Temporary data under `.reproduction/` can be removed
after validation; keep the compact receipts.

## Code and evidence

- `src/music_detector/`: the only application package, CLI and upload UI.
- `manifests/`: pinned Drive sources and frozen calibration protocol.
- `tests/`: regression, numerical, protocol and upload-safety tests.
- `validation/`: observed replay, refit, browser and storage-retention evidence.
- `scripts/verify_refit.py`: explicit refit-to-release comparison.
- `current/`: pinned historical algorithms and report drivers still needed
  by reproduction. They are evidence, not alternative application entrypoints.
- `historical_snapshots/`: original scientific implementations cited by the
  thesis, not alternative application entrypoints.
- `results/`: compact historical aggregate tables referenced by the thesis.

Superseded thesis versions, bulk dataset catalogues and dated publication
manifests are recoverable from Git commit
`b12aad5e2d3ffe1e76b5add7a049e187105ab858`; their exact checkout-pruning hashes
are in `validation/publication_prune_manifest.json`. Git history is preserved.
Private local storage inventories are deliberately excluded from public Git.
Legacy LaTeX templates are likewise restored from that baseline if an obsolete
presentation script needs them; they are not bundled as additional manuscripts.

```sh
.venv/bin/python -m pytest -q
```

No raw music, generation weights, API keys or private recovery material belong
in the public repository. Restoring private Drive audio does not grant a
redistribution license. The single final English manuscript is in the sibling
`thesis/` directory (`thesis.tex` and its compiled `thesis.pdf`, 42 pages).
Its source and PDF hashes are bound by `validation/thesis-qa/QA.json`.
No audio is needed to start the service; reference audio is downloaded only
when an explicit reproduction command requests it.

## Experimental ACE-Step online RL

An isolated training extension now supports frozen-VAE ACE-Step 1.5 SFT 2B
FM/DiT LoRA **and full-decoder** policy modes, text-only prompt preparation,
stochastic flow-policy replay, frozen artifact rewards, checkpoint/resume and
paired held-out evaluation. This is research code, not evidence that RL has
improved music quality or that it fits a particular GPU.

See [the training runbook](docs/rl_training.md) and
[the LoRA-versus-full research review](docs/rl_lora_vs_full.md).
