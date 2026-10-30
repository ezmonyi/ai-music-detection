# Archived unattended delivery and monitoring implementation

These are the actual project-owned support sources used for the 2026-10-31
RTX 3090 receiver experiment. They complement the frozen scientific core and
receiver wrapper; they are **an execution archive**, with the original task
paths and single owned node identity, rather than a general rental launcher.
No credential or connection secret is included. The working originals live in
`/Users/yi/Documents/projects/music/operations/numerical_calibration_20261031`.

The user explicitly authorized unattended completion, fixed 50-prompt audio
comparison download and protected release of the current instance. Release
cannot precede the exact output manifest, local per-file SHA-256 checks and
preservation of the unchanged completed environment snapshot. A failed or
insufficient-data experiment is recorded honestly and protected before release.

- `checkpoint_backup.py` transports only closed atomic checkpoints through
  official `matcli node exec`, with a real binary-route probe and full SHA-256.
- `live_event_mirror.py` appends authenticated event deltas to private local
  TensorBoard without replacing file prefixes or inventing scalar events.
- `protect_outputs.py` builds lossless split archives, verifies every original
  archive member on readback, manifests all outputs, and protects the latest
  checkpoint on `/mnt`. The 132 GB source dataset is excluded from local backup.
- `closeout_supervisor.py` reconciles all checkpoint and archive checksums,
  downloads the exact 100 official/new-RL test WAVs to `~/Downloads`, then uses
  the official release preview and explicit confirmation flag for the one owned
  node. Released status and two stable official billing reads are required.
- `inspect_closeout_safe.mjs` reads only whitelisted identity, snapshot and
  current-node billing fields, without displaying SSH/Jupyter credentials.
- `scoring_upper_bound_watch.py` and `quality_upper_bound_watch.py` stop an
  impossible single-epoch target only after a conservative mathematical proof.
  The latter evaluates the original pre-analyzer quality guards on CPU; a pass
  means only possible admission and never becomes a fabricated artifact score.
  Independent invalid-proof sets are unioned without double-counting.

The MatPool CLI always runs in its `matpool-machine-provider` Conda environment.
The separate project-only `music-rl-monitor` environment serves TensorBoard;
`monitor-environment.yml` and `monitor-requirements-lock.txt` record its runtime.
No local model inference is installed as part of this monitoring setup.

The meaningful standard-library regressions cover binary round-trip and corrupt
payload rejection, allowlist checks, split-archive member verification, live
event prefix preservation and conservative feasibility bounds. They are separate
from the scientific CPU/GPU admission tests. Replaying an archived script does
not itself grant permission to create, modify or release any future instance.
