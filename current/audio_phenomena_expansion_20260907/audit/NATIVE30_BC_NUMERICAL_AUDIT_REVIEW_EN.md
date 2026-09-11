# Native30 BC numerical-audit review

Status: complete. The caller-pinned Native30 BC v2 COMMIT is terminal and the
independent numerical/integrity audit passed all 3,830 rows. The success
receipt is an audit result only; it does not admit BC or authorize fitting.

## Scope and entry contract

`code/audit_native30_bc_v1.py` is a read-only, fail-closed auditor. Its CLI
requires all of the following caller-pinned inputs:

```text
--result-dir <complete Native30 BC result root>
--freeze <native30-bc-parent-freeze-v1 JSON> --freeze-sha256 <64-hex SHA>
--commit <result-root/COMMIT.json> --commit-sha256 <64-hex SHA>
--output <new receipt path outside result/source/code trees>
```

The implementation accepts only the fixed v2 producer contract, exact 3,830
sorted rows, a complete terminal `COMMIT.json`, and the frozen producer,
cohort, gate, code, and runtime bindings. A missing/partial COMMIT, retained
failure, orphan product, source/product mutation, symlink, wrong-order view,
or changed authority prevents publication. A successful receipt is an audit
result only; it is not BC admission, a classifier fit, or model scoring.

Implementation bindings for this review:

```text
audit_native30_bc_v1.py:       b25fe15e39f29cf44302c20cda6e051d072da3cd34a4993bb6e0ef61f5d6f33f
test_audit_native30_bc_v1.py:  02bfff0588c18d51dacd357367fb0f5d365994fa417c40aefde7b97b8ea20133
```

## Runtime-schema correction

Frozen runtime manifests store `module_files`, `package_files`, and
`libsndfile_binary` as complete `{path, bytes, sha256}` dictionaries. The
auditor now validates that shape and passes `entry["path"]` to
`file_binding`, comparing the freshly computed binding to the complete entry.
This fixes the older draft's dict-to-path mistake without changing any
producer code, numerical tolerances, thresholds, or measurement decisions.
The synthetic suite includes a real-shaped runtime manifest and a malformed
binding rejection. The manual Kaiser-5 FIR operation order remains unchanged;
its cutoff-times-index construction is bit-exact to the pinned SciPy reference.

## Independent replay

For every retained row the auditor rehashes the materialized FLOAT WAV before
and after decoding, checks exact RIFF/SoundFile format, 44,100 Hz, stereo,
1,323,000 frames, finite full EOF, and reconstructs the float64 stereo hash.
It explicitly computes `(L + R) / 2` in float64, manually designs the fixed
Kaiser-5 rational resampler, applies whole-record resampling before the fixed
floor-centered `[176000:304000]` crop, and applies gain `0.25`.

It then independently replays both 64,000-sample pools: explicit 1,024-frame
means, periodic Hann, normalized RFFT, pooled energy/fraction masks, the fixed
target reducer, and all 228 grid cells. The primitive uses the corrected
F-contiguous normalized-column layout and binary-scaled sufficient sums. All
metadata fields, NPZ array keys/shapes/dtypes, masks, nulls, producer scalar,
manifest coverage, item lineage, and array/intermediate hashes are checked.

The inherited numeric comparison margins are absolute `2e-12` and relative
`2e-11`; masks and nulls are exact. No threshold, tolerance, condition, or
denominator is changed by this auditor. Scientific nulls retain their row;
producer failures are not converted into scientific nulls.

## Frozen authority pins

The auditor checks the caller freeze and the producer v2 bindings, including:

```text
parent freeze: native30-bc-parent-freeze-v1
producer:      run_native30_bc_v2.py
producer SHA:  822f51f3e961bb4edf16389283a91b898393084f34d1a87da0956d7a92cffe0d
producer tests:4d24a5e8689eb4d7599669e560ab504eb29a34b8409e9471b6a304c48ff5c7dd
prepared:      3509a425dda56ec32c12bd0a481d22e8006f5892edc744ef1de8b350c00f58c9
cohort:        730b9bb25d18715c5ed0191ec9eb7a921d473b9534c9e5bc9196c25f5ce44510
plan:          94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964
screen:        ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87
```

The full numerical dependency graph is rehashed from the freeze runtime,
including Python/NumPy/SciPy/SoundFile versions, imported module files,
package files, and the libsndfile binding. The auditor does not import the
producer extractor, primitive, scalar reducer, or frozen resampling routine.

## Verification performed

```text
python3 -m py_compile code/audit_native30_bc_v1.py code/test_audit_native30_bc_v1.py
<pinned Python 3.11.15> -m unittest test_audit_native30_bc_v1 -v
```

The pinned remote runtime (Python 3.11.15, NumPy 1.26.4, SciPy 1.17.1,
SoundFile 0.14.0) completed all 14 synthetic/integrity tests:

```text
ssh 117.50.223.120 'python -B -m unittest discover -s /mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code -p test_audit_native30_bc_v1.py -v'
Ran 14 tests in 1.680s
OK
```

## Completed cohort audit

The terminal result COMMIT and parent freeze were rechecked immediately before
launch. The caller supplied and the auditor rehashed:

```text
parent freeze SHA: c7fcf8cdf2d3bb0e4673a7aa6b2e3468a67e31cd0e7fc4d4ec30575c9d874ea0
result COMMIT SHA: 573293da2c4159a5a30c204b08881eea86c2137bfce352fd6db3acf8765f398e
```

The corrected full replay ran detached as PID `2585780` and completed with
`passed=true`, `rows_expected=3830`, `rows_replayed=3830`, nine scientific
null rows, 1,746,480 full-grid cells, 7,660 pools, 1,892,020 STFT frames, and
11,492 products. The receipt records `producer_extractor_primitive_scalar_imported=false`
and retains all grid evidence.

```text
log:     /mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_bc_v1_independent_audit_20260911_r6.log
receipt: /mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_bc_v1_independent_receipt_20260911_r6.json
receipt SHA-256: c2c2c7a1ba70b7afbe5c766386b872a54e1deb96b5c2463ff0ceb1107bc021cc
```

The receipt's runtime is Python 3.11.15 / NumPy 1.26.4 / SciPy 1.17.1 /
SoundFile 0.14.0, and its auditor/test bindings match the hashes above. The
full replay changed no source audio, result product, producer code, threshold,
classifier fit, model score, or admission state.

## Preserved operational failure

The first post-patch launch is preserved unchanged for auditability:

```text
PID:  2584138
log:  /mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_bc_v1_independent_audit_20260911_r5.log
```

It failed closed before replay because the launch environment added
`MKL_NUM_THREADS=2` while the frozen runtime requires that variable to be
unset. It produced no receipt and is an operational environment mismatch, not
a measurement failure. Prior failed-closed logs and the earlier successful
receipt remain separate and were not overwritten.

The audit remains outside classifier/admission work: it performs no fitting,
model scoring, threshold changes, or producer/result/source writes. It also
does not establish causal or statistical significance for the descriptive BC
statistic.
