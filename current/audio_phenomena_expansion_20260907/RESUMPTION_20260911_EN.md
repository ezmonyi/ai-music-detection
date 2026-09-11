# Existing experiment resumption — 2026-09-11

Scope: finish previously planned experiments; no new datasets, phenomena,
thresholds, or split selection. Raw audio separation is not being rerun.

Verified access: 5090-2 (yi@117.50.223.120); project root
`/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907` (RC), data root
`/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907` (RD).
Credentials are excluded from project files.

Completed inputs remain available: Native30 SDRP, FHSC, and BC measurements.
SDRP COMMIT: `8cc844c789724fadfbb6344b4576d6964377bd268ace8946b5074e5e34c84ee2`.
BC COMMIT: `573293da2c4159a5a30c204b08881eea86c2137bfce352fd6db3acf8765f398e`.
Measurement completion does not imply classifier performance or final audit acceptance.

## Work dispatched

- Seven-family input package assembly: RD/native30_evaluation_package_v3_20260911.
  Log: RC/logs/native30_package_v3_20260911.log. No duplicate assembly.
- Parent regression check: 47 tests passed in 9.958 seconds in the frozen
  remote runtime; RC/logs/native30_v3_resume_tests_20260911.log. Failure lines
  inside the adversarial fixture output are intentional; suite exit was zero.
- Seven-family evaluator preflight follows only a successful package COMMIT.
  Separate parent fit authorization remains required before production fitting.
- BC independent numerical audit implementation and actual-runtime tests are
  under review before any full-cohort audit acceptance.
- Latest missing LaTeX/report code is being recovered from the server-retained
  Mac results archive into RC/recovery_20260911, without overwriting frozen code.

## Remaining deliverables

1. Seven-family 127-subset, five-cap evaluation (51,562 eligible model cells).
2. BC full-cohort independent numerical audit, matched eight-family package,
   evaluation (103,530 eligible cells) and X+BC versus X comparisons.
3. Verification of reported metrics, missingness, negative increments and
   source/generator holdout boundaries; final results tables and English report.

These are planned or running tasks, not claims of successful completion.

## Seven-family package completion

Assembly finished successfully (exit 0). Package COMMIT SHA:
`361d499824cf1340ef5b65073685bbf2b981002545ee80485fe07374e1607863`.
Parent independently rehashed this file and inspected metadata/features:
3,830 rows, 1,664 human and 2,166 AI, across 11 source groups. Counts of rows
with every feature finite within each family are S=3,830, D=3,830, R=3,739,
P=465, F=3,819, H=3,789, SC=3,829. These are complete-feature counts, not
the producer eligibility flags (for example P eligibility is not identical
to completeness of every section-duration and section-bar feature).
No row is dropped merely because a family contains missing values.

## Additional verified recovery

Recovered the original BC report adapter and tests from the retained archive:
code SHA `c8cca9237826b7ee7a8b4d4a75d39ebd3998b6a71532355b69f32de408a0a2db`,
test SHA `f7c13e71022f358f5f6612dd7714934e1c456cd3cb5f65947da9a6848054b7bf`.
Seven report-adapter tests passed in 11.618 seconds in the remote runtime
(log: RC/logs/native30_bc_reporter_resume_tests_r3_20260911.log).
The first two retained test attempts failed because the migrated environment
lacked the test-only mirrors of the two frozen schedule documents. Restored
those documents byte-for-byte from the authoritative RD copies; no numerical
code or thresholds were changed. Latest archived LaTeX sources are restored
locally, but no new final thesis compilation has been claimed.

Root authorizes the existing seven-family fold-only evaluation conditional
on actual deep preflight success, exact roster/catalogue/accounting and
unchanged pinned evaluator/assembler code. A separately hashed fit freeze
must capture the actual preflight contract before any model fitting. This
authorization is not BC approval or an independent publication-level audit.

The actual seven-family preflight has now passed and was inspected by root:
contract SHA `5db5c305e9ed75c697dc6e21ce7e73134e51409d4cb88b2ab18d103106e958eb`,
54 features, 36,830 primary and 14,732 diagnostic model cells; ridge 10,
threshold 0.5. Python 3.11.15, NumPy 1.26.4, pandas 3.0.5, thread caps 2.
Root approved this exact contract for the separate fit freeze and detached
execution. This is not a claim that any model or classifier report is complete.

Detached execution is now independently verified by root on 5090-2:
Python PID 2590371 (launcher PID 2590370), output
RD/native30_evaluation_v3_20260911, log
RC/logs/native30_evaluation_v3_20260911.log. Separate parent freeze SHA is
`767669be8beae5c880b197a446d1c9e393649c04746019220f945cebba9a379a`.
The process initially performs NFS input verification; no completed fitting
or performance result is inferred from the launch alone.

BC independent replay is running as PID 2585780 on 5090-2. Its retained log is
RC/audit/native30_bc_v1_independent_audit_20260911_r6.log; expected receipt is
RC/audit/native30_bc_v1_independent_receipt_20260911_r6.json. Parent reran the
14 auditor tests successfully in 1.666 seconds. Auditor SHA
`b25fe15e39f29cf44302c20cda6e051d072da3cd34a4993bb6e0ef61f5d6f33f`;
tests SHA `02bfff0588c18d51dacd357367fb0f5d365994fa417c40aefde7b97b8ea20133`.
The earlier r5 launch failed because its thread environment differed from the
measurement freeze; this operational failure remains logged, with no pass receipt.
