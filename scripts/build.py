"""Build script for creating Windows executables.

Usage:
    python scripts/build.py              # PyInstaller (quick)
    python scripts/build.py --nuitka     # Nuitka (optimized, release)
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESOURCES = ROOT / "vaspen" / "resources"


def build_pyinstaller() -> None:
    """Build with PyInstaller — fast, good for development."""
    # --add-data takes "source<os.pathsep>dest" (';' on Windows, ':' on Linux).
    # Dest "vaspen/resources" mirrors the source layout so the frozen code
    # finds i18n/icons next to the package (Path(__file__).parent).
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "VASPen",
        "--windowed",
        "--onedir",
        "--icon", str(ROOT / "vaspen" / "resources" / "icons" / "app.ico"),
        "--add-data", f"{RESOURCES}{os.pathsep}vaspen{os.sep}resources",
        str(ROOT / "vaspen" / "main.py"),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("Build complete → dist/VASPen/")


def build_nuitka() -> None:
    """Build with Nuitka — compiles to C, better performance and code protection."""
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=disable",
        "--enable-plugin=pyside6",
        f"--include-data-dir={ROOT / 'vaspen' / 'resources'}=resources",
        f"--output-dir={ROOT / 'dist'}",
        str(ROOT / "vaspen" / "main.py"),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("Build complete → dist/main.dist/")


if __name__ == "__main__":
    if "--nuitka" in sys.argv:
        build_nuitka()
    else:
        build_pyinstaller()
