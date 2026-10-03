"""Small, optional TensorBoard monitoring wrapper for RL runs.

The RL package deliberately keeps TensorBoard optional.  A disabled
``RunMonitor`` is a true no-op, while an enabled monitor fails at
construction time when the optional dependency is unavailable instead of
silently dropping metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from numbers import Real
from pathlib import Path
import tempfile
from typing import Any, Protocol


class _SummaryWriter(Protocol):
    """The small part of SummaryWriter used by :class:`RunMonitor`."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> Any:
        ...

    def add_text(self, tag: str, text_string: str, global_step: int) -> Any:
        ...

    def flush(self) -> Any:
        ...

    def close(self) -> Any:
        ...


# Keeping this name at module scope makes the dependency easy to replace in
# focused tests, while the actual import remains lazy for disabled monitors.
SummaryWriter: Any | None = None


def _resolve_summary_writer() -> Any:
    """Load the optional PyTorch TensorBoard writer with an actionable error."""

    global SummaryWriter
    if SummaryWriter is not None:
        return SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter as torch_summary_writer
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "TensorBoard monitoring is enabled, but the optional TensorBoard "
            "dependency is unavailable. Install `tensorboard` (or disable "
            "monitoring with enabled=False)."
        ) from error
    SummaryWriter = torch_summary_writer
    return torch_summary_writer


class RunMonitor:
    """Write optional per-run scalar and text events to TensorBoard.

    ``log_dir`` is treated as a run-log root.  Each enabled monitor gets a
    unique ``run-*`` child directory so event files from separate runs cannot
    collide.  ``add_scalars`` and ``add_text`` flush immediately; this keeps
    group/evaluation metrics visible even if a long-running process stops
    before its next periodic flush.

    ``writer_factory`` is an intentionally small injection point for tests.
    Production callers should omit it and use the standard PyTorch
    ``SummaryWriter``.
    """

    def __init__(
        self,
        log_dir: Path,
        enabled: bool = True,
        *,
        writer_factory: Callable[..., _SummaryWriter] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self.event_dir: Path | None = None
        self._writer: _SummaryWriter | None = None
        self._closed = False

        if not self.enabled:
            return

        factory = writer_factory or _resolve_summary_writer()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.event_dir = Path(tempfile.mkdtemp(prefix="run-", dir=self.log_dir))
        try:
            self._writer = factory(log_dir=str(self.event_dir))
            if self._writer is None:
                raise TypeError("SummaryWriter factory returned None")
        except Exception:
            # Only remove the just-created empty directory.  If a writer has
            # already emitted files, preserving them is safer for diagnosis.
            try:
                self.event_dir.rmdir()
            except OSError:
                pass
            self.event_dir = None
            raise

    def __enter__(self) -> "RunMonitor":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close()
        except Exception:
            # Never hide the exception that caused a monitored block to fail.
            if exc_type is None:
                raise
        return False

    def _ensure_open(self) -> None:
        if self.enabled and self._closed:
            raise RuntimeError("RunMonitor is closed")

    def add_scalars(self, values: Mapping[str, float | None], step: int) -> None:
        """Write finite, non-``None`` scalar values and flush immediately.

        All values are validated before any writer call, so a nonfinite value
        cannot leave a partially written metric batch behind.
        """

        if not self.enabled:
            return
        self._ensure_open()
        assert self._writer is not None

        finite_values: list[tuple[str, float]] = []
        for tag, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Scalar {tag!r} must be a real number or None")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Scalar {tag!r} must be finite")
            finite_values.append((tag, numeric))

        for tag, value in finite_values:
            self._writer.add_scalar(tag, value, step)
        self._writer.flush()

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        """Write a text event and flush immediately."""

        if not self.enabled:
            return
        self._ensure_open()
        assert self._writer is not None
        self._writer.add_text(tag, text, step)
        self._writer.flush()

    def flush(self) -> None:
        """Flush pending events, if monitoring is enabled and open."""

        if not self.enabled:
            return
        self._ensure_open()
        assert self._writer is not None
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the writer; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        if not self.enabled or self._writer is None:
            return

        writer = self._writer
        flush_error: Exception | None = None
        try:
            writer.flush()
        except Exception as error:
            flush_error = error
        try:
            writer.close()
        except Exception:
            if flush_error is None:
                raise
        if flush_error is not None:
            raise flush_error


__all__ = ["RunMonitor"]
