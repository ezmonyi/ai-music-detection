# Dependency source recovery verification

On 2026-09-13, anonymous GitHub API requests resolved both server checkout
commits below. Their codeload tar archives were streamed without extraction,
and relevant Python/shell member bytes were compared to the server inventory
using size and SHA-256.

| Dependency | Exact commit | Verification |
|---|---|---|
| [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5/commit/ca1e85fe9430179831e6bc6be790c332190a3866) | ca1e85fe9430179831e6bc6be790c332190a3866 | 587 source paths matched exactly; no mismatches |
| [HeartLib](https://github.com/HeartMuLa/heartlib/commit/3783bdb8441f2c298b1e64c8651173aac200361c) | 3783bdb8441f2c298b1e64c8651173aac200361c | All 10 inventoried source paths matched exactly |

ACE-Step's other 22 inventoried paths are under the nano-vllm build/lib tree.
Every one is a byte-identical duplicate of a non-build source object in the
same verified inventory. Thus all 619 dependency file objects are recoverable
by content from these commits, with build copies distinguished from original
source paths. Both server Git working trees reported clean status.

This verifies source recovery for the inventoried Python/shell files only.
It does not verify model weights, environments, ignored non-source files,
successful installation, or reproduced audio. No inference was run. Upstream
availability was observed at this date, not guaranteed indefinitely.

The full inventory is retained locally as
`current_results/HISTORICAL_SERVER_CODE_AUDIT_V2_20260913.json` in the report
archive. Nine non-dependency variants from the same audit are preserved under
`historical_snapshots/historical_server_variants_v2/`.
