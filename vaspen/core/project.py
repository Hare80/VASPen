"""Calculation projects — folder-as-project with a JSON manifest.

A project is an ORDINARY directory (git-friendly, AI-navigable, no
binary database — settled §7.15)::

    my-project/
    ├── project.json          # the manifest (schema-versioned)
    ├── structures/           # source structures ("materials")
    └── runs/                 # one directory per calculation (standard
        │                     # VASP layout inside, outputs land there)
        ├── 001-scf-fe3o4/
        │   ├── run.json      # per-run provenance card
        │   └── INCAR KPOINTS POSCAR POTCAR
        └── 002-neb-o-diff/

Design boundaries (settled):

- VASP output files are NEVER parsed here — run status is a label set
  by the user or the AI (whitelisted values); at most a caller may
  display file-existence hints.
- Plain-English ValueError messages (transform.py precedent — no _tr
  contexts; this module is AI-facing as much as GUI-facing).
- The manifest is written atomically (temp file + os.replace) after
  every mutation — projects autosave; there is no "save project" step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from vaspen import __version__

SCHEMA_VERSION = 1

#: Whitelisted run statuses (draft = directory exists but no inputs —
#: create_run always writes inputs, so runs start at inputs-ready).
RUN_STATUSES = ("inputs-ready", "submitted", "running", "done", "failed")

MANIFEST_NAME = "project.json"
STRUCTURES_DIR = "structures"
RUNS_DIR = "runs"


def slugify(text: str) -> str:
    """Windows-safe lowercase slug: [a-z0-9-], collapsed separators."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def sha256_file(path: str | Path) -> str:
    """`sha256:<hex>` digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class Project:
    """One on-disk project directory bound to its manifest."""

    def __init__(self, root: Path, manifest: dict) -> None:
        self._root = root
        self._manifest = manifest

    # -- lifecycle ------------------------------------------------------

    @classmethod
    def create(cls, directory: str | Path, name: str) -> "Project":
        """Create the project skeleton in ``directory`` (must not
        already contain a manifest)."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if (root / MANIFEST_NAME).exists():
            raise ValueError(
                f"{root / MANIFEST_NAME} already exists — open the "
                "project instead of creating it.")
        (root / STRUCTURES_DIR).mkdir()
        (root / RUNS_DIR).mkdir()
        manifest = {
            "schema": SCHEMA_VERSION,
            "name": name,
            "created": _now(),
            "structures": [],
            "runs": [],
        }
        project = cls(root, manifest)
        project._write_manifest()
        return project

    @classmethod
    def load(cls, directory: str | Path) -> "Project":
        """Load an existing project (manifest required)."""
        root = Path(directory)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError(
                f"No {MANIFEST_NAME} in {root} — not a VASPen project.")
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or "schema" not in manifest:
            raise ValueError(
                f"{manifest_path} is not a valid project manifest.")
        if manifest["schema"] != SCHEMA_VERSION:
            raise ValueError(
                f"Project schema {manifest['schema']} is newer/different "
                f"than supported {SCHEMA_VERSION}.")
        # Dirs may be missing in a hand-trimmed project — recreate.
        for sub in (STRUCTURES_DIR, RUNS_DIR):
            (root / sub).mkdir(exist_ok=True)
        manifest.setdefault("structures", [])
        manifest.setdefault("runs", [])
        return cls(root, manifest)

    # -- queries --------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def name(self) -> str:
        return str(self._manifest.get("name", self._root.name))

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def structures(self) -> list[dict]:
        return list(self._manifest["structures"])

    @property
    def runs(self) -> list[dict]:
        return list(self._manifest["runs"])

    def structure_path(self, structure_id: str) -> Path:
        for entry in self._manifest["structures"]:
            if entry["id"] == structure_id:
                path = self._root / entry["file"]
                if not path.exists():
                    raise ValueError(
                        f"Structure file missing: {path}")
                return path
        raise ValueError(f"Unknown structure id: {structure_id}")

    def run_dir(self, run_id: str) -> Path:
        for entry in self._manifest["runs"]:
            if entry["id"] == run_id:
                return self._root / RUNS_DIR / entry["dir"]
        raise ValueError(f"Unknown run id: {run_id}")

    def run_entry(self, run_id: str) -> dict:
        for entry in self._manifest["runs"]:
            if entry["id"] == run_id:
                return entry
        raise ValueError(f"Unknown run id: {run_id}")

    def next_run_number(self) -> int:
        """One past the highest existing run number (never recycled)."""
        highest = 0
        for entry in self._manifest["runs"]:
            match = re.match(r"^(\d+)-", entry["id"])
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    # -- mutations (each autosaves the manifest) ------------------------

    def add_structure(self, src_path: str | Path, name: str | None = None,
                      metadata: dict | None = None) -> dict:
        """Copy a structure file into structures/ and register it.

        IDs are slugs of the stem; a collision appends -2, -3, … (the
        FILE is still copied — same name, different content, is a
        distinct material)."""
        src = Path(src_path)
        if not src.exists():
            raise ValueError(f"File not found: {src}")
        stem = name or src.stem
        base = slugify(stem)
        existing = {entry["id"] for entry in self._manifest["structures"]}
        struct_id = base
        counter = 2
        while struct_id in existing:
            struct_id = f"{base}-{counter}"
            counter += 1
        dest = self._root / STRUCTURES_DIR / src.name
        if dest.exists():  # same filename, different id — keep both
            dest = self._root / STRUCTURES_DIR / f"{struct_id}{src.suffix}"
        shutil.copy2(src, dest)
        entry = {
            "id": struct_id,
            "file": str(dest.relative_to(self._root)),
            "digest": sha256_file(dest),
            "added": _now(),
        }
        if metadata:
            entry.update(metadata)  # e.g. formula, space_group
        self._manifest["structures"].append(entry)
        self._write_manifest()
        return entry

    def add_run(self, slug: str, rtype: str, params: dict,
                structure_id: str, structure_digest: str,
                via: str) -> dict:
        """Register a run directory (the caller writes the VASP files
        into it) with its provenance card."""
        number = self.next_run_number()
        run_id = f"{number:03d}-{slugify(slug)}"
        run_dirname = run_id
        run_dir = self._root / RUNS_DIR / run_dirname
        if run_dir.exists():
            raise ValueError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        card = {
            "id": run_id,
            "type": rtype,
            "structure_id": structure_id,
            "structure_digest": structure_digest,
            "parameters": params,
            "generated_by": {
                "app": "VASPen",
                "version": __version__,
                "via": via,
            },
            "status": "inputs-ready",
            "created": _now(),
        }
        (run_dir / "run.json").write_text(
            json.dumps(card, indent=2) + "\n", encoding="utf-8")
        entry = {
            "id": run_id,
            "dir": run_dirname,
            "type": rtype,
            "structure": structure_id,
            "status": "inputs-ready",
            "created": card["created"],
        }
        self._manifest["runs"].append(entry)
        self._write_manifest()
        return {**entry, "card": card}

    def set_run_status(self, run_id: str, status: str) -> dict:
        if status not in RUN_STATUSES:
            raise ValueError(
                f"Status must be one of {', '.join(RUN_STATUSES)} "
                f"(got {status!r}).")
        entry = self.run_entry(run_id)
        entry["status"] = status
        entry["status_changed"] = _now()
        # keep the provenance card in sync
        card_path = self.run_dir(run_id) / "run.json"
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text("utf-8"))
                card["status"] = status
                card_path.write_text(
                    json.dumps(card, indent=2) + "\n", encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                pass  # panel metadata is authoritative; card is best-effort
        self._write_manifest()
        return dict(entry)

    def run_parameters(self, run_id: str) -> dict:
        """The run's parameter dict from its provenance card."""
        card_path = self.run_dir(run_id) / "run.json"
        if not card_path.exists():
            return {}
        try:
            card = json.loads(card_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(card.get("parameters", {}))

    def update_run_parameters(self, run_id: str, parameters: dict,
                              replace: bool = False) -> dict:
        """Set the run's provenance-card parameters. Merges by default;
        ``replace=True`` swaps the whole parameter dict (input
        regeneration uses replace so stale overrides never survive).
        Manifest runs entries carry no parameters — the card is the
        single source."""
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dict.")
        self.run_entry(run_id)  # validate the id
        card_path = self.run_dir(run_id) / "run.json"
        try:
            card = json.loads(card_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read run card: {exc}") from exc
        merged = parameters if replace else {
            **card.get("parameters", {}), **parameters}
        card["parameters"] = merged
        card["parameters_updated"] = _now()
        card_path.write_text(
            json.dumps(card, indent=2) + "\n", encoding="utf-8")
        return dict(merged)

    def write_run_inputs(self, run_id: str, model, preset: str = "scf",
                         incar_overrides: dict | None = None,
                         kpoints_mode: str = "automatic",
                         kpoints_params: dict | None = None,
                         potcar_library: str = "",
                         functional: str = "PBE",
                         poscar_direct: bool = False) -> dict:
        """(Re)generate the four VASP input files in the run directory
        from ``model`` and record the parameters on the provenance
        card. ``model`` is the caller's StructureModel — typically
        loaded from the registered structure file."""
        from vaspen.core.vasp_input import generate_all_inputs

        run_dir = self.run_dir(run_id)
        contents = generate_all_inputs(
            model,
            incar_preset=preset,
            incar_overrides=incar_overrides,
            kpoints_mode=kpoints_mode,
            kpoints_params=kpoints_params,
            potcar_library=potcar_library,
            functional=functional,
            poscar_direct=poscar_direct,
        )
        files: dict[str, str] = {}
        for name, content in contents.items():
            if not content:
                continue
            target = run_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            files[name] = str(target)
        # REPLACE: regeneration defines the full parameter set — stale
        # overrides from a previous generation must not survive.
        self.update_run_parameters(run_id, {
            "preset": preset,
            "kpoints_mode": kpoints_mode,
            "kpoints_params": kpoints_params or {},
            "functional": functional,
            **(incar_overrides or {}),
        }, replace=True)
        return files

    # -- persistence ----------------------------------------------------

    def _write_manifest(self) -> None:
        """Atomic manifest write — a crash never leaves half a file."""
        target = self._root / MANIFEST_NAME
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
