"""Focused tests for the optional TensorBoard run monitor."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_detector.rl import monitoring
from music_detector.rl.monitoring import RunMonitor


class FakeWriter:
    instances: list["FakeWriter"] = []

    def __init__(self, *, log_dir: str):
        self.log_dir = Path(log_dir)
        self.calls: list[tuple] = []
        self.closed = False
        type(self).instances.append(self)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.calls.append(("scalar", tag, value, step))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.calls.append(("text", tag, text, step))

    def flush(self) -> None:
        self.calls.append(("flush",))

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_writers():
    FakeWriter.instances.clear()


def test_disabled_monitor_is_a_noop_without_importing_tensorboard(tmp_path: Path, monkeypatch):
    def unexpected_dependency_resolution():
        raise AssertionError("disabled monitoring must not resolve TensorBoard")

    monkeypatch.setattr(monitoring, "_resolve_summary_writer", unexpected_dependency_resolution)
    log_dir = tmp_path / "disabled"
    monitor = RunMonitor(log_dir, enabled=False)

    monitor.add_scalars({"bad": float("nan"), "missing": None}, step=4)
    monitor.add_text("status", "ignored")
    monitor.flush()
    monitor.close()

    assert monitor.event_dir is None
    assert not log_dir.exists()


def test_enabled_monitor_uses_unique_local_event_directories(tmp_path: Path):
    first = RunMonitor(tmp_path, writer_factory=FakeWriter)
    second = RunMonitor(tmp_path, writer_factory=FakeWriter)

    assert first.event_dir is not None
    assert second.event_dir is not None
    assert first.event_dir.parent == tmp_path
    assert second.event_dir.parent == tmp_path
    assert first.event_dir != second.event_dir
    assert FakeWriter.instances[0].log_dir == first.event_dir
    assert FakeWriter.instances[1].log_dir == second.event_dir

    first.close()
    second.close()


def test_scalars_skip_none_and_flush_immediately(tmp_path: Path):
    monitor = RunMonitor(tmp_path, writer_factory=FakeWriter)
    writer = FakeWriter.instances[0]

    monitor.add_scalars({"reward": 1.25, "optional": None}, step=7)
    monitor.add_text("phase", "group complete", step=7)

    assert writer.calls == [
        ("scalar", "reward", 1.25, 7),
        ("flush",),
        ("text", "phase", "group complete", 7),
        ("flush",),
    ]
    monitor.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scalars_are_rejected_before_any_write(tmp_path: Path, value: float):
    monitor = RunMonitor(tmp_path, writer_factory=FakeWriter)
    writer = FakeWriter.instances[0]

    with pytest.raises(ValueError, match="finite"):
        monitor.add_scalars({"good": 1.0, "bad": value}, step=1)

    assert writer.calls == []
    monitor.close()


def test_missing_tensorboard_fails_clearly_when_enabled(tmp_path: Path, monkeypatch):
    def missing_dependency():
        raise RuntimeError(
            "TensorBoard monitoring is enabled, but the optional TensorBoard "
            "dependency is unavailable."
        )

    monkeypatch.setattr(monitoring, "_resolve_summary_writer", missing_dependency)
    with pytest.raises(RuntimeError, match="TensorBoard monitoring is enabled"):
        RunMonitor(tmp_path)
    assert not list(tmp_path.iterdir())


def test_context_closes_writer_and_preserves_body_failure(tmp_path: Path):
    writer = None
    with pytest.raises(RuntimeError, match="body failed"):
        with RunMonitor(tmp_path, writer_factory=FakeWriter):
            writer = FakeWriter.instances[0]
            raise RuntimeError("body failed")

    assert writer is not None
    assert writer.closed
    assert writer.calls[-2:] == [("flush",), ("close",)]


def test_close_is_idempotent(tmp_path: Path):
    monitor = RunMonitor(tmp_path, writer_factory=FakeWriter)
    writer = FakeWriter.instances[0]

    monitor.close()
    monitor.close()

    assert writer.calls == [("flush",), ("close",)]
    with pytest.raises(RuntimeError, match="closed"):
        monitor.add_text("late", "event")
