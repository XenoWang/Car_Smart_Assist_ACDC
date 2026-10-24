"""Latch manual training-stop requests until an epoch boundary."""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

STOP_EXIT_CODE = 75


class StopController:
    """A persistent STOP marker or a signal requests a graceful pause."""

    def __init__(self, stop_file=None, *, default_directory: Path):
        marker = stop_file or os.environ.get("CAR_SMART_ASSIST_STOP_FILE")
        self.path = Path(marker) if marker else Path(default_directory) / "STOP"
        self._paths = {self.path, Path(default_directory) / "STOP"}
        self._requested = False
        self._handlers = {}
        self._depth = 0

    def watch_directory(self, directory: Path) -> None:
        """Also honor a nested trainer's own output-directory marker."""
        self._paths.add(Path(directory) / "STOP")

    @property
    def requested(self) -> bool:
        if not self._requested:
            self._requested = any(path.exists() for path in self._paths)
        return self._requested

    def request(self) -> None:
        self._requested = True

    def request_stop_file(self) -> None:
        """Create this controller's marker without starting training."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.request()

    def resume(self) -> None:
        """Explicit continue/fresh clears only this controller's marker files."""
        self._requested = False
        shared = os.environ.get("CAR_SMART_ASSIST_STOP_FILE")
        shared_path = Path(shared).resolve() if (
            os.environ.get("CAR_SMART_ASSIST_PIPELINE_CHILD") == "1" and shared
        ) else None
        for path in self._paths:
            if shared_path is None or path.resolve() != shared_path:
                path.unlink(missing_ok=True)
        if shared_path is not None and shared_path.exists():
            self.request()

    def require_running(self) -> None:
        """Refuse new work before archives, training, or another model starts."""
        if self.requested:
            raise SystemExit(STOP_EXIT_CODE)

    def _signal(self, _signum, _frame) -> None:
        self.request()

    def __enter__(self):
        if self._depth == 0 and threading.current_thread() is threading.main_thread():
            try:
                for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
                    signum = getattr(signal, name, None)
                    if signum is not None:
                        self._handlers[signum] = signal.signal(signum, self._signal)
            except BaseException:
                for signum, handler in self._handlers.items():
                    signal.signal(signum, handler)
                self._handlers.clear()
                raise
        self._depth += 1
        return self

    def __exit__(self, _type, _value, _traceback):
        self._depth -= 1
        if self._depth == 0:
            for signum, handler in self._handlers.items():
                signal.signal(signum, handler)
            self._handlers.clear()
