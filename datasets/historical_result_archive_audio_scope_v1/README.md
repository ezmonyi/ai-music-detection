# Audio scope of the 19 historical result archives

All 19 `historical_results/*.tar.gz` archives were read to completion using
Python `tarfile.open(..., 'r|gz')`, without extracting or modifying any file.
The scan counted regular-file members whose lower-case suffix was one of:
`.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.aif`, `.aiff`, `.mid`, `.midi`.

Every archive contained zero members matching these audio/MIDI suffixes.
Consequently these result archives add no demonstrated direct waveform
coverage for the unmatched historical stems. This is a filename-based member
inventory, not recursive inspection of nested containers or media sniffing of
arbitrarily named payloads.

| Archive stem | Regular-file members | Recognized audio/MIDI |
|---|---:|---:|
| audio_phenomena_expansion_20260907 | 38390 | 0 |
| demucs_artifacts_100x100_20260814 | 15 | 0 |
| demucs_bias_corrected_1000_20260901 | 46 | 0 |
| direct_band_500x500_20260815 | 1011 | 0 |
| dynamics_rhythm_change_detector_extension_20260903 | 2148 | 0 |
| dynamics_rhythm_external_benchmark_20260903 | 127 | 0 |
| dynamics_rhythm_method_research_20260903 | 1 | 0 |
| external_filter_test_20260902 | 40 | 0 |
| external_generator_500_heuristics_20260904 | 7173 | 0 |
| external_generator_500_testset_20260904 | 85 | 0 |
| four_family_diversity_ablation_20260904 | 35 | 0 |
| frequency_ablation_20260813 | 406 | 0 |
| interpretable_phenomena_literature_20260906 | 2 | 0 |
| musdb18_oracle_vocal_eval_20260901 | 16 | 0 |
| music_flamingo_midi_rhythm_probe_20260903 | 44 | 0 |
| open_models_spectral_500_20260901 | 175 | 0 |
| source_diversity_expansion_20260904 | 36 | 0 |
| source_diversity_expansion_20260905 | 339 | 0 |
| stem_spectral_pilot_20260813 | 13 | 0 |

This does not invalidate preservation of numerical results and source code in
these archives. It distinguishes result preservation from audio preservation.
Other private, encrypted, or source-specific archives are outside this scan.
