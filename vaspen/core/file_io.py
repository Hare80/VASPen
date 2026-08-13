"""Unified file I/O with a format registry.

Supports registering read/write handlers per file extension so that
adding a new format only requires registration — no code changes
elsewhere in the codebase.
"""

from __future__ import annotations

import re
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

# Formats whose file content inherently represents a periodic structure
PERIODIC_FORMATS: frozenset[str] = frozenset({"vasp", "cif"})

# Canonical ASE format name per extension. Explicit mapping makes
# .poscar/.contcar work on all platforms — ASE's own glob matching
# (*POSCAR*/*CONTCAR*) is case-sensitive on Linux.
FORMAT_BY_SUFFIX: dict[str, str] = {
    ".vasp": "vasp", ".poscar": "vasp", ".contcar": "vasp",
    ".cif": "cif", ".xyz": "xyz", ".extxyz": "extxyz",
    ".xsf": "xsf", ".pdb": "pdb", ".json": "json",
    ".cube": "cube", ".traj": "traj", ".db": "db",
}

# Extensionless basenames recognized as VASP structure files
VASP_BASENAMES: frozenset[str] = frozenset({"poscar", "contcar"})


def resolve_format(filepath: str | Path) -> str | None:
    """Return the canonical ASE format name for a path, or None.

    Extensionless basenames POSCAR/CONTCAR (any case) resolve to "vasp",
    so the canonical VASP filenames open and save on every platform.
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext:
        return FORMAT_BY_SUFFIX.get(ext)
    if path.stem.lower() in VASP_BASENAMES:
        return "vasp"
    return None


def _normalize_pbc(atoms: Atoms, fmt: str | None) -> Atoms:
    """Mark structures with a full 3D cell as periodic (mutates in place).

    The ASE CIF reader never sets pbc (ase/io/cif.py), so VASPen rendered
    opened CIF crystals as molecules without a cell frame. The vasp reader
    already sets pbc=True. Scoped to cif only: for xyz a stored cell with
    explicit pbc=False (e.g. a molecule in a vacuum box) must round-trip
    unchanged.
    """
    if fmt == "cif" and atoms.get_cell().rank == 3 and not atoms.pbc.any():
        atoms.pbc = True
    return atoms


# Numeric prefix of an occupancy token: accepts "0.5", "1.0", "0.5(2)"
# (error-bar notation) — the "(2)" suffix is dropped.
_OCC_NUM_PREFIX = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)")


def _sanitize_cif_occupancy(text: str) -> str:
    """Replace non-numeric occupancy tokens in CIF text with ``1.0``.

    ASE's CIF reader treats occupancy values as floats, but real-world
    CIFs (e.g. ICSD entries) mark unknown occupancy with ``?`` or ``.``.
    ASE then crashes with a str/float comparison when merging
    partially-occupied sites (ase/spacegroup/xtal.py). Unknown occupancy
    is normalized to full occupation (1.0); error-bar values like
    ``0.5(2)`` keep their numeric prefix. Only the ``_atom_site_occupancy``
    column inside ``loop_`` blocks is touched.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.lower().startswith("loop_"):
            out.append(line)
            i += 1
            headers: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip().lstrip("_"))
                out.append(lines[i])
                i += 1
            if "atom_site_occupancy" in headers:
                occ_col = headers.index("atom_site_occupancy")
                # Data rows run until the next tag / loop / data block.
                while (i < len(lines)
                       and lines[i].strip()
                       and not lines[i].strip().startswith(
                           ("_", "loop_", "data_", "#"))):
                    toks = lines[i].split()
                    if len(toks) > occ_col:
                        tok = toks[occ_col].strip("\"'")
                        m = _OCC_NUM_PREFIX.match(tok)
                        toks[occ_col] = m.group(0) if m else "1.0"
                        out.append(" ".join(toks))
                    else:
                        out.append(lines[i])
                    i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def extract_occupancy(atoms: Atoms) -> list[dict[str, float]] | None:
    """Return per-atom site compositions from ASE occupancy info, or None.

    ASE's CIF reader merges co-located partially-occupied species into
    one atom per site (the symbol is the dominant species) and stores the
    full composition in ``atoms.info['occupancy']`` — a dict mapping
    ``str(spacegroup_kinds)`` (or the raw site index when the kinds array
    is absent) to ``{symbol: occupancy}``. Returns one dict per atom, in
    atom order, or None when the structure carries no occupancy data —
    including uniform single-species 1.0 occupancy, which ASE also
    reports and which does not constitute disorder.
    """
    occ_info = atoms.info.get("occupancy")
    if not isinstance(occ_info, dict) or not occ_info:
        return None

    def _parse_comp(comp: object) -> dict[str, float]:
        if not isinstance(comp, dict):
            return {}
        parsed: dict[str, float] = {}
        for sym, occ in comp.items():
            try:
                parsed[str(sym)] = float(occ)
            except (TypeError, ValueError):
                continue
        return parsed

    by_key = {str(k): _parse_comp(v) for k, v in occ_info.items()}
    kinds = atoms.arrays.get("spacegroup_kinds")
    if kinds is None and len(by_key) == len(atoms):
        # CIF path without symmetry expansion: keys are site indices in
        # atom order.
        kinds = list(range(len(atoms)))
    if kinds is None:
        return None
    result: list[dict[str, float]] = []
    for i, kind in enumerate(kinds):
        comp = by_key.get(str(kind))
        if not comp:
            # Site has no occupancy entry — full occupation of its symbol.
            comp = {str(atoms.symbols[i]): 1.0}
        result.append(comp)
    if all(len(comp) == 1 and next(iter(comp.values())) == 1.0
           for comp in result):
        return None
    return result


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
            for_writing: If True, only include writable formats. The write
                filter is categorized per type (VESTA-style) with no
                combined "Structure files" entry; the read filter keeps
                the combined entry first for convenience.

        Returns:
            Filter string like ``"Structure files (*.cif *.xyz);;All files (*)"``.
        """
        exts = cls.supported_write_formats() if for_writing else cls.supported_read_formats()
        filters: list[str] = []
        if not for_writing:
            patterns = " ".join(f"*{e}" for e in exts) + " POSCAR CONTCAR"
            filters.append(_tr("Structure files") + f" ({patterns})")
        for ext in exts:
            if ext in (".poscar", ".contcar"):
                continue  # covered by the combined VASP entry below
            if ext == ".vasp":
                # Canonical VASP filenames are extensionless
                filters.append(_tr("POSCAR / CONTCAR (VASP)") + " (*.vasp *.poscar *.contcar POSCAR CONTCAR)")
                continue
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
        fmt = resolve_format(path)
        if ext not in cls._readers and fmt != "vasp":
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: {cls.supported_read_formats()}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if fmt == "vasp":
            # Explicit format: works for extensionless POSCAR/CONTCAR and
            # .poscar/.contcar on every platform (do NOT force "xyz" —
            # ASE's extxyz reader preserves Lattice/pbc keys).
            from ase.io import read as ase_read
            atoms = ase_read(str(path), format="vasp")
        else:
            atoms = cls._readers[ext](str(path))
        return _normalize_pbc(atoms, fmt)

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
        ext = path.suffix.lower()
        if fmt is not None:
            target = fmt
        else:
            resolved = resolve_format(path)
            if ext not in cls._writers and resolved != "vasp":
                raise ValueError(
                    f"Unsupported file extension: {ext}. "
                    f"Supported: {cls.supported_write_formats()}"
                )
            target = resolved
        if target in PERIODIC_FORMATS and atoms.get_cell().rank < 3:
            # Guard BEFORE any file is created: ASE's vasp writer raises a
            # raw RuntimeError here, and its cif writer silently writes a
            # cell-less CIF. The UI wraps molecules first (see
            # MainWindow._ensure_periodic_for).
            raise ValueError(_tr(
                "Cannot save a non-periodic structure in {} format: a unit cell is required."
            ).format(target))
        if target == "vasp":
            from ase.io import write as ase_write
            ase_write(str(path), atoms, format="vasp")
        elif fmt is not None:
            # Explicit format request bypasses the extension registry
            from ase.io import write as ase_write
            ase_write(str(path), atoms, format=fmt)
        else:
            cls._writers[ext](str(path), atoms)


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


def _cif_reader(path: str | Path) -> Atoms:
    """CIF reader — sanitizes invalid occupancy tokens before ASE reads.

    Real-world CIFs (ICSD) often mark unknown occupancy with ``?`` or
    ``.``; ASE crashes on those when merging partially-occupied sites
    (see ``_sanitize_cif_occupancy``). Unknown occupancy is read as full
    occupation (1.0).
    """
    import io

    from ase.io import read as ase_read

    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return ase_read(io.StringIO(_sanitize_cif_occupancy(text)), format="cif")


# Register all common structure formats
_BUILTIN_EXTENSIONS = [
    ".cif", ".xyz", ".vasp", ".poscar", ".contcar",
    ".xsf", ".pdb", ".json", ".cube", ".extxyz",
    ".traj", ".db",
]

for _ext in _BUILTIN_EXTENSIONS:
    FileIO.register(_ext, _ase_reader, _ase_writer)

# CIF needs occupancy sanitation (the generic ASE reader above crashes on
# '?'-valued occupancies) — override with the dedicated reader.
FileIO.register(".cif", _cif_reader, _ase_writer)
