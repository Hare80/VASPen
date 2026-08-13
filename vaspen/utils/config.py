"""Application configuration via QSettings.

Persists user preferences (window geometry, recent files, language,
pseudopotential library path, default INCAR overrides, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings


class AppConfig:
    """Singleton wrapper around QSettings for VASPen preferences.

    Usage:
        config = AppConfig()
        config.set("language", "zh")
        lang = config.get("language", "en")
    """

    _instance: AppConfig | None = None

    def __new__(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = QSettings("VASPen", "VASPen")
        return cls._instance

    # ------------------------------------------------------------------
    # Generic get / set
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Read a setting value."""
        return self._settings.value(key, default)

    def set(self, key: str, value: Any) -> None:
        """Write a setting value."""
        self._settings.setValue(key, value)

    def remove(self, key: str) -> None:
        """Remove a setting."""
        self._settings.remove(key)

    def sync(self) -> None:
        """Flush pending writes to disk."""
        self._settings.sync()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        return self.get("language", "en")

    @language.setter
    def language(self, value: str) -> None:
        self.set("language", value)

    @property
    def potcar_library_path(self) -> str:
        return self.get("potcar_library_path", "")

    @potcar_library_path.setter
    def potcar_library_path(self, value: str) -> None:
        self.set("potcar_library_path", value)

    @property
    def wrap_padding(self) -> float:
        """Remembered vacuum padding for the periodic-wrap dialog (Angstrom)."""
        try:
            return float(self.get("periodic_wrap_padding", 5.0))
        except (TypeError, ValueError):
            return 10.0

    @wrap_padding.setter
    def wrap_padding(self, value: float) -> None:
        self.set("periodic_wrap_padding", float(value))

    @property
    def remember_wrap_padding(self) -> bool:
        """Whether the wrap dialog should default to the remembered padding."""
        # QSettings on Windows returns bools as the string "false" —
        # bool("false") is True, so compare the string form.
        return str(self.get("remember_wrap_padding", False)).lower() in ("true", "1")

    @remember_wrap_padding.setter
    def remember_wrap_padding(self, value: bool) -> None:
        self.set("remember_wrap_padding", bool(value))

    @property
    def recent_files(self) -> list[str]:
        return self.get("recent_files", []) or []

    @recent_files.setter
    def recent_files(self, files: list[str]) -> None:
        self.set("recent_files", files[-10:])  # keep last 10

    def add_recent_file(self, filepath: str | Path) -> None:
        """Add a file to the recent-files list (deduplicates, max 10)."""
        files = self.recent_files
        path_str = str(Path(filepath).resolve())
        if path_str in files:
            files.remove(path_str)
        files.insert(0, path_str)
        self.recent_files = files

    @property
    def default_calc_type(self) -> str:
        return self.get("default_calc_type", "scf")

    @default_calc_type.setter
    def default_calc_type(self, value: str) -> None:
        self.set("default_calc_type", value)

    @property
    def window_geometry(self) -> bytes | None:
        return self.get("window_geometry")

    @window_geometry.setter
    def window_geometry(self, value: bytes) -> None:
        self.set("window_geometry", value)

    @property
    def window_state(self) -> bytes | None:
        return self.get("window_state")

    @window_state.setter
    def window_state(self, value: bytes) -> None:
        self.set("window_state", value)

    @property
    def last_directory(self) -> str:
        return self.get("last_directory", str(Path.home()))

    @last_directory.setter
    def last_directory(self, value: str) -> None:
        self.set("last_directory", value)

    @property
    def render_settings(self):
        """Persisted 3D render settings (JSON string under QSettings).

        Corrupt or missing JSON falls back to defaults. Lazy import
        keeps utils free of a core dependency at module load time.
        """
        import json

        from vaspen.core.render_settings import RenderSettings

        raw = self.get("render_settings", "")
        if not raw:
            return RenderSettings.default()
        try:
            return RenderSettings.from_dict(json.loads(str(raw)))
        except (ValueError, TypeError):
            return RenderSettings.default()

    @render_settings.setter
    def render_settings(self, value) -> None:
        import json

        self.set("render_settings", json.dumps(value.to_dict()))
