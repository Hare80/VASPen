"""Unified file I/O with a format registry.

Supports registering read/write handlers per file extension so that
adding a new format only requires registration — no code changes
elsewhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ase import Atoms
from PySide6.QtCore import QCoreApplication

# Type aliases
Reader = Callable[[str | Path], Atoms]
Writer = Callable[[str | Path, Atoms], None]


def _tr(text: str) -> str:
    """Translate a user-visible string (translation context: FileIO).

    The core layer cannot use QObject.tr(); QCoreApplication.translate
    resolves against the installed translators at call time, so file
    dialog labels follow the active UI language.
    """
    return QCoreApplication.translate("FileIO", text)


# Map extensions → display name for file dialogs
EXTENSION_DISPLAY_NAMES: dict[str, str] = {
    ".cif": "CIF — Crystallographic Information File",
    ".xyz": "XYZ — Extended XYZ",
    ".vasp": "POSCAR / CONTCAR (VASP)",
    ".poscar": "POSCAR (VASP)",
    ".contcar": "CONTCAR (VASP)",
    ".xsf": "XCrySDen Structure File",
    ".pdb": "Protein Data Bank",
    ".json": "ASE JSON",
    ".cube": "Gaussian Cube",
}


class FileIO:
    """Registry-based file reader/writer for structure files.

    Built-in handlers leverage ASE's ``read`` / ``write`` functions.
    Additional formats can be registered at runtime.

    Usage:
        FileIO.register(".myfmt", my_reader, my_writer)
        atoms = FileIO.read("structure.myfmt")
        FileIO.write("output.myfmt", atoms)
    """

    _readers: dict[str, Reader] = {}
    _writers: dict[str, Writer] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, extension: str, reader: Reader, writer: Writer) -> None:
        """Register a reader and writer for a file extension.

        Args:
            extension: File extension including the dot (e.g. ``".xyz"``).
            reader: Callable that takes a path and returns ASE Atoms.
            writer: Callable that takes (path, Atoms).
        """
        ext = extension.lower()
        cls._readers[ext] = reader
        cls._writers[ext] = writer

    @classmethod
    def supported_read_formats(cls) -> list[str]:
        """Return list of registered readable extensions."""
        return sorted(cls._readers.keys())

    @classmethod
    def supported_write_formats(cls) -> list[str]:
        """Return list of registered writable extensions."""
        return sorted(cls._writers.keys())

    @classmethod
    def file_filter(cls, for_writing: bool = False) -> str:
        """Build a Qt file-dialog filter string.

        Args:
            for_writing: If True, only include writable formats.

        Returns:
            Filter string like ``"Structure files (*.cif *.xyz);;All files (*)"``.
        """
        exts = cls.supported_write_formats() if for_writing else cls.supported_read_formats()
        patterns = " ".join(f"*{e}" for e in exts)
        filters = [_tr("Structure files") + f" ({patterns})"]
        for ext in exts:
            label = _tr(EXTENSION_DISPLAY_NAMES.get(ext, ext.upper()))
            filters.append(f"{label} (*{ext})")
        filters.append(_tr("All files") + " (*)")
        return ";;".join(filters)

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    @classmethod
    def read(cls, filepath: str | Path) -> Atoms:
        """Read a structure file.

        Format is detected from the file extension.

        Args:
            filepath: Path to the structure file.

        Returns:
            ASE Atoms object.

        Raises:
            ValueError: If the file extension is not registered.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        if ext not in cls._readers:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: {cls.supported_read_formats()}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return cls._readers[ext](str(path))

    @classmethod
    def write(cls, filepath: str | Path, atoms: Atoms, fmt: str | None = None) -> None:
        """Write a structure file.

        Args:
            filepath: Destination path.
            atoms: ASE Atoms to write.
            fmt: ASE format string. Auto-detected from extension if None.

        Raises:
            ValueError: If the file extension is not registered.
        """
        path = Path(filepath)
        if fmt is not None:
            # Explicit format request bypasses the extension registry
            from ase.io import write as ase_write
            ase_write(str(path), atoms, format=fmt)
            return
        ext = path.suffix.lower()
        if ext not in cls._writers:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: {cls.supported_write_formats()}"
            )
        self_ = cls._writers[ext]
        self_(str(path), atoms)


# ------------------------------------------------------------------
# Built-in format registration using ASE
# ------------------------------------------------------------------

def _ase_reader(path: str | Path) -> Atoms:
    """Generic ASE reader — format auto-detected."""
    from ase.io import read as ase_read
    return ase_read(str(path))


def _ase_writer(path: str | Path, atoms: Atoms) -> None:
    """Generic ASE writer — format from extension."""
    from ase.io import write as ase_write
    ase_write(str(path), atoms)


# Register all common structure formats
_BUILTIN_EXTENSIONS = [
    ".cif", ".xyz", ".vasp", ".poscar", ".contcar",
    ".xsf", ".pdb", ".json", ".cube", ".extxyz",
    ".traj", ".db",
]

for _ext in _BUILTIN_EXTENSIONS:
    FileIO.register(_ext, _ase_reader, _ase_writer)
