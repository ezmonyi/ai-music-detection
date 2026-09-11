#!/usr/bin/env python3
"""Dependency-free canonical feature names for union and verification tools."""

S16_FEATURES = (
    "tilt_1_5k_db_oct", "hf_tilt_5_16k_db_oct", "hf_ratio_5_16_db",
    "air_ratio_12_20_db", "sibilance_ratio_5_10_db", "hf_flatness",
    "hf_entropy", "hf_crest_db", "fakeprint_peak_density",
    "fakeprint_periodicity", "hf_flux", "hf_frame_similarity",
    "hf_power_sd_db", "hf_mod_4_12_share", "sibilance_contrast_db",
    "sibilance_burst_rate_hz",
)

S8_FEATURES = (
    "tilt_1_5k_db_oct", "hf_tilt_5_7p5k_db_oct", "hf_ratio_5_7p5_db",
    "sibilance_ratio_5_7p5_db", "hf_flatness_5_7p5", "hf_entropy_5_7p5",
    "hf_crest_5_7p5_db", "fakeprint_peak_density_5_7p5_per_khz",
    "fakeprint_periodicity_5_7p5", "hf_flux_5_7p5",
    "hf_frame_similarity_5_7p5", "hf_power_sd_5_7p5_db",
    "hf_mod_4_12_share_5_7p5", "sibilance_contrast_5_7p5_db",
    "sibilance_burst_rate_5_7p5_hz",
)

D_FEATURES = ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")
R_FEATURES = ("ibi_cv", "tempo_tv", "tempo_entropy")
P_FEATURES = (
    "section_duration_cv", "section_duration_entropy", "section_bars_cv",
    "section_bars_offmode_fraction", "section_duration_median", "section_bars_median",
)
