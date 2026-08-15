"""Build script for creating Windows executables.

Usage:
    python scripts/build.py              # PyInstaller (quick dev build)
    python scripts/build.py --smoke      # PyInstaller + launch smoke test
    python scripts/build.py --nuitka     # Nuitka (optimized, release)
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESOURCES = ROOT / "vaspen" / "resources"
BUILD_DIR = ROOT / "build"
DIST_APP = ROOT / "dist" / "VASPen"

#: Files copied into dist/VASPen/ after the build so the release zip is
#: self-contained (MIT requires the license text alongside the binary;
#: the guides make the zip self-documenting).
BUNDLED_DOCS = [
    "LICENSE",
    "README.md",
    "docs/user-guide-en.md",
    "docs/user-guide-zh.md",
]

SMOKE_TIMEOUT_S = 90


def _read_version() -> str:
    """Read ``__version__`` from vaspen/__init__.py — the single source
    of truth (the file is a docstring + constant, safe to exec)."""
    spec = importlib.util.spec_from_file_location(
        "vaspen_pkg", ROOT / "vaspen" / "__init__.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__


def version_tuple(version: str) -> tuple[int, int, int, int]:
    """Split "0.3.0" → (0, 3, 0, 0); tolerates pre-release suffixes
    ("0.3.0.dev0" → (0, 3, 0, 0))."""
    parts: list[int] = []
    for part in version.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0, 0, 0])[:4])  # type: ignore[return-value]


def _write_version_file(version: str) -> Path:
    """Render the Windows VERSIONINFO from ``__version__``.

    Generated at build time into build/ (gitignored) — no second
    hand-maintained version file to drift out of sync.
    """
    BUILD_DIR.mkdir(exist_ok=True)
    path = BUILD_DIR / "version_info.txt"
    path.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version_tuple(version)},
    prodvers={version_tuple(version)},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [
            StringStruct('CompanyName', 'VASPen'),
            StringStruct('FileDescription', 'VASPen - GUI for VASP first-principles calculations'),
            StringStruct('FileVersion', '{version}'),
            StringStruct('InternalName', 'VASPen'),
            StringStruct('OriginalFilename', 'VASPen.exe'),
            StringStruct('ProductName', 'VASPen'),
            StringStruct('ProductVersion', '{version}'),
            StringStruct('LegalCopyright', 'MIT License')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
        encoding="utf-8",
    )
    return path


def build_pyinstaller(smoke: bool = False) -> None:
    """Build with PyInstaller — fast, good for development. The release
    pipeline (version resource, bundled docs/examples, zip, smoke test)
    runs here too."""
    # --add-data takes "source<os.pathsep>dest" (';' on Windows, ':' on Linux).
    # Dest "vaspen/resources" mirrors the source layout so the frozen code
    # finds i18n/icons next to the package (Path(__file__).parent).
    # Fail loudly instead of silently building without the exe icon —
    # the .ico is a committed generated asset.
    icon = RESOURCES / "icons" / "app.ico"
    if not icon.exists():
        raise SystemExit(
            f"app.ico missing ({icon}) — run "
            f"`python scripts/make_icon.py` first")
    version = _read_version()
    version_file = _write_version_file(version)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "VASPen",
        "--windowed",
        "--onedir",
        # Rebuild in place: overwrite the previous dist/VASPen instead of
        # aborting with "output directory is not empty".
        "--noconfirm",
        "--icon", str(icon),
        # Windows file-version resource, generated from vaspen/__init__.py.
        "--version-file", str(version_file),
        # Keep the generated VASPen.spec inside build/ (it is gitignored
        # either way — this keeps the repo root tidy).
        "--specpath", str(BUILD_DIR),
        "--add-data", f"{RESOURCES}{os.pathsep}vaspen{os.sep}resources",
        # ASE's format plugins (extxyz/vasp/cif/…) are imported lazily via
        # the format registry — PyInstaller's static analysis cannot see
        # them, and the exe fails with "No module named 'ase.io.extxyz'".
        # Collect all ase.io submodules (plus ase.geometry, also lazy).
        "--collect-submodules", "ase.io",
        "--collect-submodules", "ase.geometry",
        # ASE and pymatgen load data files at runtime (ase/spacegroup/
        # spacegroup.dat for CIF symmetry, ase/collections/*.json for the
        # molecule builder, pymatgen's periodic_table.json etc.) — static
        # analysis misses them and the exe fails with "No such file or
        # directory". Collect all package data for both.
        "--collect-data", "ase",
        "--collect-data", "pymatgen",
        str(ROOT / "vaspen" / "main.py"),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)

    _bundle_release_files()
    zip_path = _make_zip(version)
    _verify_dist(zip_path)
    if smoke:
        _smoke_test()


def build_nuitka() -> None:
    """Build with Nuitka — compiles to C, better performance and code protection."""
    icon = RESOURCES / "icons" / "app.ico"
    if not icon.exists():
        raise SystemExit(
            f"app.ico missing ({icon}) — run "
            f"`python scripts/make_icon.py` first")
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=disable",
        "--enable-plugin=pyside6",
        "--windows-icon-from-ico", str(icon),
        # Dest mirrors the PyInstaller layout (vaspen/resources next to
        # the module) — the code resolves resources via __file__.
        f"--include-data-dir={ROOT / 'vaspen' / 'resources'}=vaspen{os.sep}resources",
        f"--output-dir={ROOT / 'dist'}",
        str(ROOT / "vaspen" / "main.py"),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("Build complete → dist/main.dist/")


def _bundle_release_files() -> None:
    """Copy LICENSE/docs/examples into dist/VASPen/.

    The pseudopotential library is never bundled — nothing to exclude
    there; the resources/potcars dir ships empty by design.
    """
    for rel in BUNDLED_DOCS:
        src = ROOT / rel
        dst = DIST_APP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copytree(ROOT / "examples", DIST_APP / "examples",
                    dirs_exist_ok=True)


def _make_zip(version: str) -> Path:
    """Zip dist/VASPen → dist/VASPen-v<version>-win64.zip with a single
    top-level VASPen/ folder (safe "Extract here" behavior)."""
    zip_base = DIST_APP.parent / f"VASPen-v{version}-win64"
    stale = Path(str(zip_base) + ".zip")
    if stale.exists():
        stale.unlink()
    zip_path = Path(shutil.make_archive(
        str(zip_base), "zip",
        root_dir=str(DIST_APP.parent), base_dir="VASPen"))
    print(f"Release zip → {zip_path}")
    return zip_path


def _verify_dist(zip_path: Path) -> None:
    """Hard-fail if the dist is missing a key piece — silent breakage
    here ships a broken release."""
    internal = DIST_APP / "_internal"
    expected = [
        DIST_APP / "VASPen.exe",
        internal / "vaspen" / "resources" / "i18n" / "vaspen_en.qm",
        internal / "vaspen" / "resources" / "i18n" / "vaspen_zh.qm",
        internal / "vaspen" / "resources" / "themes" / "light.qss",
        internal / "vaspen" / "resources" / "themes" / "dark.qss",
        internal / "vaspen" / "resources" / "icons" / "app.ico",
        DIST_APP / "LICENSE",
        zip_path,
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise SystemExit(
            "Build verification failed — missing:\n  "
            + "\n  ".join(str(p) for p in missing))
    bundle_mb = sum(f.stat().st_size for f in DIST_APP.rglob("*")
                    if f.is_file()) / 1e6
    print(f"Verified dist: {bundle_mb:.0f} MB bundle, "
          f"zip {zip_path.stat().st_size / 1e6:.0f} MB")


def _frozen_log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", "") or Path.home())
    return base / "VASPen" / "vaspen.log"


def _smoke_test() -> None:
    """Launch the built exe and wait for the frozen log to grow — proof
    the app actually started (main.py logs "VASPen started" at launch).

    Never kills a VASPen.exe that was already running before the test.
    """
    exe = DIST_APP / "VASPen.exe"
    running = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq VASPen.exe", "/NH"],
        capture_output=True, text=True)
    if "VASPen.exe" in running.stdout:
        raise SystemExit(
            "VASPen.exe is already running — close it before --smoke "
            "(the test never kills a live session).")
    log_path = _frozen_log_path()
    log_size = log_path.stat().st_size if log_path.exists() else 0

    proc = subprocess.Popen([str(exe)])
    try:
        deadline = time.time() + SMOKE_TIMEOUT_S
        while time.time() < deadline:
            if log_path.exists() and log_path.stat().st_size > log_size:
                break
            if proc.poll() is not None:
                raise SystemExit(
                    f"VASPen.exe exited early with code {proc.returncode}")
            time.sleep(1)
        else:
            raise SystemExit(
                f"No startup log within {SMOKE_TIMEOUT_S}s")
    finally:
        # /T kills the bootloader's child tree too.
        subprocess.run(["taskkill", "/T", "/F", "/IM", "VASPen.exe"],
                       capture_output=True)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise SystemExit("VASPen.exe did not terminate after taskkill")

    tail = log_path.read_text(
        encoding="utf-8", errors="replace").splitlines()[-5:]
    print("Smoke test OK — log tail:\n  " + "\n  ".join(tail))


if __name__ == "__main__":
    if "--nuitka" in sys.argv:
        build_nuitka()
    else:
        build_pyinstaller(smoke="--smoke" in sys.argv)
