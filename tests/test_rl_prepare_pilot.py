import pytest

from music_detector.rl.prepare_pilot import exclusion_reasons, prepare


def test_screening_preserves_ordinary_music_and_does_not_guess_vocals():
    assert exclusion_reasons("Warm piano, acoustic bass and expressive female singing.") == []
    assert exclusion_reasons("A monophonic synthesizer melody with a stereo drum kit.") == []


@pytest.mark.parametrize("caption,reason", [
    ("A MONO recording", "explicit_mono"),
    ("A low-quality recording", "degraded_recording"),
    ("A very noisy guitar performance", "noise_instruction"),
    ("A person speaking over drums", "speech_instruction"),
])
def test_screening_reports_explicit_conflicts(caption, reason):
    assert reason in exclusion_reasons(caption)


def test_wrong_source_is_rejected_without_output(tmp_path):
    source = tmp_path / "different.csv"
    source.write_text("caption,ytid\nA piano,x\n")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare(source, output)
    assert not output.exists()
