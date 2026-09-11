# YuE2 500-prompt extension — 2026-09-11

Status: preparation; no completed generation or classification claimed.

Operational checkpoint: frozen-manifest validation passed (500 unique IDs,
250/250 languages and 400/100 splits). SHA256:
`bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b`.
An isolated runtime is being installed on 5090-2; retry installer PID 2622229,
detached staged launcher PID 2622365. The launcher waits for installation,
checks the runtime, generates the first manifest row as a smoke check, then
continues the 500-row loop with that same output retained. Any technical
exception stops the run rather than silently skipping/replacing a prompt.
Server code root: `/mnt/nfs-code/users/yi/yue2_500_extension_20260911`;
output root: `/mnt/nfs-data/users/yi/yue2_500_extension_20260911`.
Logs: `install.log`, `generation.log` (failed first attempts),
`install_r2.log`, `generation_r2.log` (current attempt).
Initial venv bootstrap lacked ensurepip; the existing runtime's pip is used
solely to install into the new venv. First dependency download exhausted
system temporary space; retry uses task-specific shared-storage TMPDIR,
no pip cache, and shared-storage HF model cache. No user data was deleted.
Downstream analysis is specified below but NOT yet launched automatically.

## Frozen generation design

Add YuE2 as a distinct generator, not YuE v1. Repository commit:
`92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`.
Model `m-a-p/YuE2-3B` revision
`1a96eca688d6ae5d7f0feb88573fec89920fcd19`;
listening decoder `m-a-p/YuE2-Vae` revision
`95535e72a97bc0f09b8ada125d26b4009428c0e8`.
Do not substitute the legacy benchmark decoder.

Reuse exactly the existing 500-row Muse prompt manifest shared by HeartMuLa
and ACE-Step, including prompt_common, lyrics_30s, seeds, IDs and 400/100
development/locked-test assignments (250 Chinese, 250 English).
One candidate per prompt, cot=full, other released defaults unchanged.
No best-of-N, source-song cover, custom score, prompt rewrite, listening-based
selection, or detector-guided regeneration. A same-seed retry of a technical
failure is retained as a separate attempt. Do not silently replace failed rows.
Run only on the available RTX 5090 servers in an isolated Python environment.
The initial compatibility check uses the first manifest entry, not a chosen song.

Preserve full native 48 kHz stereo output and all save_artifacts products,
including symbolic plans, effective configuration, tokens, latents, timings,
weights identity and truncation flags. The request uses the existing 30-second
lyric window, but YuE2 native duration is NOT claimed to be forced to 30 seconds.
Truncated generation is marked explicitly, not silently treated as complete.

## Analysis extension and leakage controls

Retain currently running Native30/BC jobs and their frozen results unchanged.
Create versioned expanded cohorts after YuE2 generation and validation.
First evaluate YuE2 as a held-out generator using the old frozen detector;
then run expanded-development experiments with YuE2, keeping locked tests
separate. Shared prompt/source-song identities across generators must stay
in the same split/group, not become independent train/test examples.

Apply the existing standardization and htdemucs vocal separation on 5090;
reuse the frozen Demucs bias correction, not re-estimate it on YuE2 test audio.
Repeat applicable full-mix/vocal raw/corrected spectral analyses, STFT mean
differences, frequency-band ablations, D/R/P and the added phenomenon families,
single and combined-family evaluations, training-size ablations and
source/generator-held-out comparisons. BC admission still requires its audit.
Native-duration eligibility must be measured from original audio: never count
zero-padding as native 30/60-second support. Report ineligible strata rather
than fabricating them or changing duration thresholds.

The symbolic plan can contextualize dynamics/rhythm/form findings, but is not
an input to an audio-only classifier (no corresponding human-side plan).
Use the same fixed parameters and predeclared metrics, not new threshold
optimization on YuE2 locked data. Report nulls and negative increments.

## Delivery

Include tested YuE2 audio and provenance in the user's new HF dataset after
license review; include project scripts in the new GitHub repository. Preserve
all intermediate/final reports and results locally alongside this protocol.
Code license is Apache-2.0; model weights are CC BY-NC 4.0. This distinction
does not establish blanket redistribution permission for every output asset.

Sources accessed 2026-09-11:
- https://github.com/multimodal-art-projection/YuE
- https://github.com/multimodal-art-projection/YuE/blob/main/docs/generation.md
- https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE
