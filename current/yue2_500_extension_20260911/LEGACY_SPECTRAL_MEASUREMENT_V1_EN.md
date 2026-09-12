# Legacy spectral extension — measurement scope

The remaining YuE2 legacy spectral arm uses all 500 fixed generated recordings
and the original 500 human baseline rows, preserving their original split
labels. It reuses the existing 44.1 kHz, stereo PCM16 first-30-second views
and previously completed htdemucs stems. Two YuE2 recordings need padding in
this representation; this is not the center-30-second Native30 cohort.

The calibration SHA256 is
`bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e`.
Both historical calibration copies were checked to match before launch. The
curve is not estimated from YuE2 or the new AI/human difference maps.

`run_legacy_spectral_measurement_v1.py` hashes all 2,000 input mix/vocal files,
binds the imported legacy source files, extracts raw and corrected spectral
metrics and frequency vectors, and runs the existing descriptive class-average
spectrogram exporter. It rehashes audio before committing the output inventory.
No classifier is fitted, no neural inference is performed, and no threshold
is selected. Descriptive locked-cohort maps must not be reused for tuning.

The existing heatmap uses 4096-point STFT, 1024 hop, and 128 log-spaced bands
between 300 Hz and 20 kHz. It reports raw/corrected absolute and frame-centered
spectral shapes. Unaligned excerpt times are not musical phrase alignment.

The legacy classifier main function is deliberately not invoked in this
measurement step: it has its own historical evaluation scheme, which requires
a separate group/split audit before reuse. This does not complete the remaining
frequency-band classification comparisons or native60 extension.

Launch: server 5090-2, PID 3084177, observed live after launch on September 12.
Log: `/mnt/nfs-code/users/yi/yue2_500_extension_20260911/legacy_spectral_measurement_v1.log`.
Output: `/mnt/nfs-data/users/yi/yue2_500_extension_20260911/legacy_spectral_measurement_v1`.
Only a verified terminal COMMIT establishes measurement completion.
