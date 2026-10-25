"""Tests for the release-build file selection (publish isolation,
§7.16): the zip may only contain COMMITTED repository files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.build import _tracked_examples

pytest.importorskip("pytest")
_GIT = subprocess.run(["git", "--version"], capture_output=True)
HAS_GIT = _GIT.returncode == 0


def _init_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    ex = root / "examples"
    ex.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root,
                   check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root,
                   check=True)
    return root


@pytest.mark.skipif(not HAS_GIT, reason="git not available")
def test_only_committed_examples_bundle(tmp_path):
    root = _init_repo(tmp_path)
    (root / "examples" / "Cu.vasp").write_text("committed",
                                               encoding="utf-8")
    subprocess.run(["git", "add", "examples/Cu.vasp"], cwd=root,
                   check=True)
    subprocess.run(["git", "commit", "-qm", "demo"], cwd=root,
                   check=True)
    # personal file on disk, and a merely STAGED one — neither ships
    (root / "examples" / "personal.vasp").write_text("mine",
                                                     encoding="utf-8")
    (root / "examples" / "staged.vasp").write_text("staged",
                                                   encoding="utf-8")
    subprocess.run(["git", "add", "examples/staged.vasp"], cwd=root,
                   check=True)
    files = [f.name for f in _tracked_examples(root)]
    assert files == ["Cu.vasp"]


@pytest.mark.skipif(not HAS_GIT, reason="git not available")
def test_fresh_repo_falls_back_to_index(tmp_path):
    """No commits yet: ls-tree fails, the index fallback applies."""
    root = _init_repo(tmp_path)
    (root / "examples" / "Cu.vasp").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "examples/Cu.vasp"], cwd=root,
                   check=True)
    files = [f.name for f in _tracked_examples(root)]
    assert files == ["Cu.vasp"]


def test_no_git_at_all_falls_back_to_disk(tmp_path, monkeypatch):
    root = tmp_path / "plain"
    (root / "examples").mkdir(parents=True)
    (root / "examples" / "a.vasp").write_text("x", encoding="utf-8")
    (root / "examples" / "sub").mkdir()
    (root / "examples" / "sub" / "b.vasp").write_text("y",
                                                      encoding="utf-8")

    def _no_git(*a, **k):
        raise OSError("git not found")

    monkeypatch.setattr("scripts.build.subprocess.run", _no_git)
    files = sorted(f.name for f in _tracked_examples(root)
                   if f.is_file())
    assert files == ["a.vasp", "b.vasp"]
