"""File-change monitoring for the loaded structure (GUI).

Wraps QFileSystemWatcher with a debounce timer and an mtime+size
baseline. QFileSystemWatcher fires per write event — editors often
write in bursts, and some save by delete+recreate — so raw signals are
too noisy to prompt the user with. The monitor only emits
``file_changed(path)`` when the file on disk genuinely differs from the
last-seen baseline after the churn settles.

Baseline discipline (settled §7.12):

- ``watch(path)`` stores a fresh baseline — call after open/save/reload
  (the GUI's own writes therefore never prompt);
- ``rebase()`` adopts the current on-disk state — call after the user
  chooses to ignore a change, so only FUTURE changes fire again;
- ``clear()`` detaches (derived structure — the model no longer has a
  filepath to watch).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal

logger = logging.getLogger(__name__)

#: Writes settle fast; this only needs to cover write bursts, not user
#: think-time (each new event during the window restarts the timer).
DEBOUNCE_MS = 400

#: (mtime_ns, size) snapshot used as the change baseline.
_Stat = tuple[int, int] | None


class FileChangeMonitor(QObject):
    """Debounced on-disk change detection for one file path."""

    file_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._watcher = QFileSystemWatcher(self)
        self._path: str | None = None
        self._baseline: _Stat = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._evaluate)
        self._watcher.fileChanged.connect(self._on_file_changed)

    @property
    def watched_path(self) -> str | None:
        """The currently watched path, or None."""
        return self._path

    def watch(self, path: str | Path) -> None:
        """Watch ``path`` with a fresh baseline (open/save/reload)."""
        p = str(Path(path))
        if self._path is not None and self._path != p:
            self._watcher.removePath(self._path)
        if self._path != p:
            self._watcher.addPath(p)
        self._path = p
        self._baseline = self._stat(p)
        self._debounce.stop()

    def clear(self) -> None:
        """Detach from any file (model has no filepath)."""
        if self._path is not None:
            self._watcher.removePath(self._path)
        self._path = None
        self._baseline = None
        self._debounce.stop()

    def rebase(self) -> None:
        """Adopt the current on-disk state as the baseline.

        Used after the user chooses to ignore a change (only future
        changes should fire again).
        """
        if self._path is not None:
            self._baseline = self._stat(self._path)
        self._debounce.stop()

    # -- internals -----------------------------------------------------

    @staticmethod
    def _stat(path: str) -> _Stat:
        try:
            st = Path(path).stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _on_file_changed(self, _path: str) -> None:
        if self._path is None:
            return
        # Delete+recreate writers drop the path from the watcher —
        # re-add so the recreated file is still observed. If the file
        # is gone for good, _evaluate() stays quiet (nothing to load).
        if Path(self._path).exists():
            self._watcher.addPath(self._path)
        self._debounce.start()

    def _evaluate(self) -> None:
        if self._path is None:
            return
        current = self._stat(self._path)
        if current is None or current == self._baseline:
            # Missing = transient mid-write state; the recreated file
            # fires fileChanged again and restarts the debounce.
            return
        logger.info("File changed on disk: %s", self._path)
        self.file_changed.emit(self._path)
