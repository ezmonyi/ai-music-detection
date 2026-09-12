# YuE2 generation and duration catalogue

All 500 generated recording IDs, including two excluded from Native30.
Producer metadata and exclusion hashes were checked against the Native60 input
COMMIT. All 498 duration-eligible Native30 IDs, roles and groups were reconciled
with the published expanded Native30 catalogue. Native60 admits 276 recordings
and excludes 224; its exact crop offsets, lengths and input hashes are retained.
No padding is implied by either native-duration eligibility flag.

The two short recordings are ai_yue2_muse_cn_suno_cn_006375_0 (1,057,856 frames)
and ai_yue2_muse_en_suno_en_016463_0 (1,269,056 frames), both at 48 kHz.
Their exclusion from native-duration experiments does not erase the generated
source files or imply failed generation. Legacy padded-first30 analysis has a
different protocol and must not be treated as a native-duration measurement.

Source hashes identify original generated audio; Native60 input hashes identify
derived exact-length WAV files. Paths and private server details are omitted.
This metadata publication does not establish audio upload or redistribution
rights. Eligibility alone is not proof of measurement or classifier success;
refer to the separate committed experiment results. Preserve shared Muse prompt
groups across YuE2, ACE-Step and HeartMuLa when constructing splits.
