"""i18n synchronization guards (code review 2026-08-15, extended
2026-08-15 to every tr()-using module).

pyside6-lupdate is broken in this environment (extracts 0 strings), so
the .ts files are hand-maintained — drift is invisible at runtime until
a user switches language and meets an untranslated string (e.g. the
three MainWindow rebox messages that shipped in English in the Chinese
UI). These tests pin the failure modes:

- a tr() literal missing from a .ts file, or a .ts entry no code emits
  (ast parity check, both directions, over EVERY UI module that uses
  tr()/translate() and the core ``_tr()`` contexts Neb / StructureModel
  / VaspInput — the IncarEditorPanel presets that shipped in English
  in the Chinese UI were invisible until this extension),
- the FileIO context, whose strings are translated dynamically
  (``_tr(dict_value)`` — statically invisible; kept hand-maintained,
  its .ts sources must all come from EXTENSION_DISPLAY_NAMES or a
  static ``_tr`` literal),
- en.ts being a full mirror of zh.ts (translation == source — en IS
  the source language; regenerate via scripts/sync_en_ts.py, never
  hand-edit both), and
- a compiled .qm that predates the .ts (runtime QTranslator check).
"""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

from PySide6.QtCore import QTranslator

I18N = Path(__file__).parent.parent / "vaspen" / "resources" / "i18n"

#: UI modules whose tr() strings must be present in both .ts files
#: under their enclosing class's context. tools.py carries no tr()
#: calls — its shared confirm helper uses QCoreApplication.translate
#: with an explicit "MainWindow" context, which is checked too.
CHECKED_MODULES = (
    ("vaspen/ui/main_window.py", "MainWindow"),
    # bridge_server borrows the window's tr() (MainWindow context) for
    # its status strings — same convention as ui/tools.py.
    ("vaspen/ui/bridge_server.py", "MainWindow"),
    ("vaspen/ui/surface_dialog.py", "SurfaceDialog"),
    ("vaspen/ui/rebox_dialog.py", "ReBoxDialog"),
    ("vaspen/ui/symmetry_dialog.py", "SymmetryDialog"),
    ("vaspen/ui/tools.py", "MainWindow"),
    ("vaspen/ui/incar_editor.py", "IncarEditorPanel"),
    ("vaspen/ui/incar_editor.py", "IncarEditorDialog"),
    ("vaspen/ui/kpoints_editor.py", "KpointsEditorPanel"),
    ("vaspen/ui/kpoints_editor.py", "KpointsEditorDialog"),
    ("vaspen/ui/potcar_dialog.py", "PotcarPanel"),
    ("vaspen/ui/potcar_dialog.py", "PotcarDialog"),
    ("vaspen/ui/periodic_wrap_dialog.py", "PeriodicWrapDialog"),
    ("vaspen/ui/display_options_dialog.py", "DisplayOptionsDialog"),
    ("vaspen/ui/generate_all_dialog.py", "PoscarPanel"),
    ("vaspen/ui/generate_all_dialog.py", "GenerateAllDialog"),
    ("vaspen/ui/settings_dialog.py", "SettingsDialog"),
    ("vaspen/ui/structure_tree.py", "StructureTreePanel"),
    ("vaspen/ui/viewport3d.py", "Viewport3D"),
    ("vaspen/ui/measurement.py", "MeasurementPanel"),
    ("vaspen/ui/atom_properties.py", "AtomPropertiesPanel"),
    ("vaspen/ui/lattice_dialog.py", "LatticeDialog"),
    ("vaspen/ui/supercell_dialog.py", "SupercellDialog"),
    ("vaspen/ui/periodic_table_dialog.py", "PeriodicTableDialog"),
    ("vaspen/ui/transform_dialog.py", "TransformDialog"),
    ("vaspen/ui/welcome_page.py", "WelcomePage"),
)

#: Core modules translate via a module-level ``_tr(text)`` helper —
#: the context is fixed per module (the helper wraps
#: QCoreApplication.translate with it as a constant). FileIO is
#: deliberately absent: its strings are translated dynamically
#: (``_tr(dict_value)``) and pinned by test_fileio_context instead.
CORE_CONTEXTS = {
    "vaspen/core/neb.py": "Neb",
    "vaspen/core/structure.py": "StructureModel",
    "vaspen/core/vasp_input.py": "VaspInput",
}

#: Dynamically-translated strings — the tr()/translate() call site
#: carries no constant literal (``self.tr(text)`` over a module-level
#: data tuple, or a translate callable handed to a shared helper), so
#: the ast walk cannot see them. They are real runtime callers: the
#: orphan direction is relaxed for exactly these, and their presence
#: in both .ts files is pinned by test_dynamic_literals below.
DYNAMIC_LITERALS = {
    "DisplayOptionsDialog": {
        # _make_choice_button: QAction(self.tr(text), btn) over
        # _STYLE_ITEMS / _SCHEME_ITEMS.
        "Ball & Stick", "Space Filling (CPK)", "Wireframe",
        "Jmol (default)", "Metal / Non-metal",
        "Periodic table blocks (s/p/d/f)",
    },
    # structure_tree.composition_text(..., self.tr) — "Vacancy" is
    # resolved through the passed translate callable (both panels
    # share the helper, each with its own context).
    "StructureTreePanel": {"Vacancy"},
    "AtomPropertiesPanel": {"Vacancy"},
}


def _norm(text: str) -> str:
    """The hand-written .ts files store newlines as ``\\n`` escapes in
    some entries and as real newlines in others — normalize both sides
    before comparing."""
    return text.replace("\\n", "\n")


def _literals(module_rel: str) -> dict[str, set[str]]:
    """{context: string literals} — ``tr()`` calls (context = enclosing
    class), ``QCoreApplication.translate("ctx", "literal")`` calls
    (context = the explicit first argument), and core ``_tr("literal")``
    calls (context = CORE_CONTEXTS[module])."""
    tree = ast.parse(
        (Path(__file__).parent.parent / module_rel).read_text("utf-8"))
    core_context = CORE_CONTEXTS.get(module_rel)
    found: dict[str, set[str]] = {}
    stack: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "tr"
                    and stack
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.setdefault(stack[-1], set()).add(node.args[0].value)
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "translate"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                found.setdefault(node.args[0].value, set()).add(
                    node.args[1].value)
            if (core_context
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_tr"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.setdefault(core_context, set()).add(node.args[0].value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return found


def _ts_messages(ts_name: str) -> dict[str, set[str]]:
    """{context: {normalized source strings}} of a .ts file."""
    root = ET.parse(I18N / ts_name).getroot()
    out: dict[str, set[str]] = {}
    for ctx in root.findall("context"):
        name = ctx.find("name").text
        for msg in ctx.findall("message"):
            src = msg.find("source").text
            if src:
                out.setdefault(name, set()).add(_norm(src))
    return out


ZH = _ts_messages("vaspen_zh.ts")
EN = _ts_messages("vaspen_en.ts")


def test_all_tr_literals_present_in_both_ts_files():
    """Every tr()/translate() literal of the checked UI modules exists
    in zh.ts AND en.ts under its context — a missing zh entry ships
    English text in the Chinese UI; a missing en entry breaks the
    hand-maintained mirror."""
    missing: list[str] = []
    for module, context in CHECKED_MODULES:
        for literal in _literals(module).get(context, set()):
            if _norm(literal) not in ZH.get(context, set()):
                missing.append(f"zh.ts {context}: {literal!r}")
            if _norm(literal) not in EN.get(context, set()):
                missing.append(f"en.ts {context}: {literal!r}")
    assert not missing, "\n".join(missing)


def test_no_orphan_entries_in_checked_contexts():
    """No .ts source of the checked contexts may exist without a
    matching tr()/translate() literal — orphans accumulate silently
    (the review pruned 14: stale Transform/POTCAR-era messages,
    wrong-context duplicates, a stale tooltip split). Dynamically-
    translated strings (DYNAMIC_LITERALS) are exempt here and pinned
    by test_dynamic_literals_present_in_both_ts_files."""
    orphan: list[str] = []
    by_context: dict[str, set[str]] = {}
    for module, context in CHECKED_MODULES:
        by_context.setdefault(context, set()).update(_literals(module).get(
            context, set()))
    for context, literals in by_context.items():
        literals = {_norm(s) for s in literals}
        for ts_name, messages in (("zh.ts", ZH), ("en.ts", EN)):
            for src in messages.get(context, set()):
                if src in DYNAMIC_LITERALS.get(context, set()):
                    continue
                if src not in literals:
                    orphan.append(f"{ts_name} {context}: {src!r}")
    assert not orphan, "\n".join(orphan)


def test_dynamic_literals_present_in_both_ts_files():
    """The ast-invisible strings still ship in both .ts files — a
    hand-prune here would silently revert them to English."""
    missing = []
    for context, literals in DYNAMIC_LITERALS.items():
        for literal in literals:
            for ts_name, messages in (("zh.ts", ZH), ("en.ts", EN)):
                if _norm(literal) not in messages.get(context, set()):
                    missing.append(f"{ts_name} {context}: {literal!r}")
    assert not missing, "\n".join(missing)


def test_compiled_zh_qm_resolves_rebox_strings():
    """The compiled .qm must not predate the .ts — the three rebox
    MainWindow strings were missing from the Chinese UI (context trap:
    they existed under ReBoxDialog, but tr() context is the enclosing
    class)."""
    translator = QTranslator()
    assert translator.load(str(I18N / "vaspen_zh.qm"))
    for context, source in (
        ("MainWindow", "Re-box Slab"),
        ("MainWindow", "Re-boxing requires a periodic structure "
                       "(a full-rank cell with periodic boundary "
                       "conditions)."),
        ("MainWindow", "Slab re-boxed (vacuum along c)."),
        ("SurfaceDialog", "Cleave Surface / Slab"),
    ):
        translated = translator.translate(context, source)
        assert translated and translated != source, source


def test_en_ts_is_a_full_mirror_of_zh_ts():
    """en.ts is GENERATED from zh.ts (scripts/sync_en_ts.py): every
    message's translation equals its source — en IS the source
    language, so the file must never be hand-edited or drift. Both
    directions: a hand-added zh entry without re-running the
    regenerator must also fail."""
    en_root = ET.parse(I18N / "vaspen_en.ts").getroot()
    zh_root = ET.parse(I18N / "vaspen_zh.ts").getroot()
    broken = []
    for ctx in en_root.findall("context"):
        name = ctx.find("name").text
        for msg in ctx.findall("message"):
            source = msg.find("source")
            translation = msg.find("translation")
            if source is None or source.text is None:
                continue
            if (translation is None
                    or _norm(translation.text or "") != _norm(source.text)):
                broken.append(
                    f"{name}: {source.text!r} -> "
                    f"{translation.text if translation is not None else None!r}")
    missing_in_en = []
    for ctx in zh_root.findall("context"):
        name = ctx.find("name").text
        zh_sources = {_norm(m.find("source").text)
                      for m in ctx.findall("message")
                      if m.find("source") is not None
                      and m.find("source").text is not None}
        en_sources = {_norm(m.find("source").text)
                      for m in en_root.findall(f".//context[name='{name}']/message")
                      if m.find("source") is not None
                      and m.find("source").text is not None}
        for src in zh_sources - en_sources:
            missing_in_en.append(f"{name}: {src!r}")
    assert not broken, "\n".join(broken)
    assert not missing_in_en, "\n".join(missing_in_en)


def test_no_literal_backslash_n_in_ts_sources():
    """A .ts source that stores a literal backslash-n instead of a real
    newline never matches the code's tr("...\\n...") literal at runtime
    (QTranslator exact-matches strings) — the string ships English in
    the zh UI while the parity tests (which normalize both sides) stay
    green. Three SurfaceDialog sources shipped with this bug."""
    broken = []
    for ts_name in ("vaspen_zh.ts", "vaspen_en.ts"):
        root = ET.parse(I18N / ts_name).getroot()
        for ctx in root.findall("context"):
            for msg in ctx.findall("message"):
                source = msg.find("source")
                if source is not None and source.text and "\\n" in source.text:
                    broken.append(
                        f"{ts_name} {ctx.find('name').text}: {source.text!r}")
    assert not broken, "\n".join(broken)


def test_compiled_zh_qm_resolves_every_zh_entry():
    """The compiled .qm must cover EVERY zh.ts source — the old
    4-string spot check let stale .qm files slip through."""
    translator = QTranslator()
    assert translator.load(str(I18N / "vaspen_zh.qm"))
    unresolved = []
    root = ET.parse(I18N / "vaspen_zh.ts").getroot()
    for ctx in root.findall("context"):
        name = ctx.find("name").text
        for msg in ctx.findall("message"):
            source = msg.find("source")
            if source is None or source.text is None:
                continue
            translated = translator.translate(name, source.text)
            if not translated:
                unresolved.append(f"{name}: {source.text!r}")
    assert not unresolved, "\n".join(unresolved)


def test_fileio_context_sources_are_known_values():
    """FileIO strings are translated via ``_tr(dict_value)`` — the ast
    walk cannot see them, so the context is hand-maintained. Pin BOTH
    directions: every .ts FileIO source comes from
    EXTENSION_DISPLAY_NAMES or a static ``_tr`` literal (a stale entry
    here would silently ship an obsolete format name), and every
    display name + static literal exists in both .ts files (a new
    format without an entry renders its raw extension in the dialog)."""
    import vaspen.core.file_io as file_io

    static = set()
    tree = ast.parse(Path(file_io.__file__).read_text("utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_tr"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            static.add(node.args[0].value)
    known = {_norm(s) for s in set(file_io.EXTENSION_DISPLAY_NAMES.values())
             | static}
    for ts_name, messages in (("zh.ts", ZH), ("en.ts", EN)):
        fileio_sources = messages.get("FileIO", set())
        unknown = {_norm(s) for s in fileio_sources} - known
        missing = known - {_norm(s) for s in fileio_sources}
        assert not unknown, f"{ts_name} FileIO unknown: {sorted(unknown)}"
        assert not missing, f"{ts_name} FileIO missing: {sorted(missing)}"


def test_sync_en_ts_is_idempotent_and_bootstraps(tmp_path):
    """The regenerator must be byte-stable on its own output and able
    to create en.ts from scratch (missing/empty file — a fresh clone
    can only regenerate via this script)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sync_en_ts",
        Path(__file__).parent.parent / "scripts" / "sync_en_ts.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    zh_text = (I18N / "vaspen_zh.ts").read_text("utf-8")

    # Bootstrap: empty en.ts input → full valid mirror document.
    generated = sync.mirror(zh_text, "")
    assert "<TS" in generated and "</TS>" in generated
    assert sync._verify_mirror(generated) == []

    # Idempotence: mirroring the generated output again is byte-equal.
    assert sync.mirror(zh_text, generated) == generated
