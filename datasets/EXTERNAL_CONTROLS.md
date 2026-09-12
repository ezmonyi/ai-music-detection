# External measurement controls

External controls validate whether descriptors respond to an acoustic
phenomenon. They are not automatically members of the AI/human song classifier
cohort. In particular, synthetic instruments are not generative-AI authorship
labels.

## NSynth official test release

- Source: https://magenta.tensorflow.org/datasets/nsynth
- Attribution: Google Inc.; Engel et al. (2017), *Neural Audio Synthesis of
  Musical Notes with WaveNet Autoencoders*, https://arxiv.org/abs/1704.01279
- License: https://creativecommons.org/licenses/by/4.0/
- Content: unmodified official test archive, 4,096 four-second monophonic
  notes and source metadata. This archives the full source release, not a
  claim that every note was tested in every experiment.
- Publication: https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/tree/5d3ef14828ab329f844e4c4c53d8a32502d68543/external_controls/nsynth_test_source_v1
- Archive SHA256: `0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c`
- Archive bytes: 349,501,546.

The remote LFS hash and size were checked at the fixed revision above; the
attribution/license README was downloaded and checked. The source acquisition
receipt identifies the official archive and prior complete decoding. These
4,096 notes must not be added to the count of independent human or AI songs.

## GuitarSet v1.1.0

- Official source: https://doi.org/10.5281/zenodo.3371780
- Attribution: Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye,
  and Juan P. Bello; *GuitarSet: A Dataset for Guitar Transcription* (2018).
- License: https://creativecommons.org/licenses/by/4.0/
- Content: unmodified mono microphone archive (360 recordings) and corrected
  annotation archive. Other pickup variants are not included.
- Publication: https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/tree/4b649de5ebc78328b3b1e1df5ea1196c2646e143/external_controls/guitarset_source_v1
- Audio SHA256: `237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28`
- Annotation SHA256: `8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99`

The current official API license, archive sizes and MD5 checksums were checked
before publication; local SHA256 hashes match the frozen acquisition values.
All four published files (two archives, README, provenance) were checked against
remote content hashes and sizes at the fixed revision above. These source
recordings support external bicoherence controls; they do not add 360 songs to
the classifier cohort or change development/reserved/unused partitions.

Other external controls and remaining song-source uploads are still being
reconciled. This index does not claim complete dataset delivery.
