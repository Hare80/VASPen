"""i18n synchronization guards (code review 2026-08-15).

pyside6-lupdate is broken in this environment (extracts 0 strings), so
the .ts files are hand-maintained — drift is invisible at runtime until
a user switches language and meets an untranslated string (e.g. the
three MainWindow rebox messages that shipped in English in the Chinese
UI). These tests pin the failure modes:

- a tr() literal missing from a .ts file, or a .ts entry no code emits
  (ast parity check, both directions, over the four UI modules whose
  contexts are fully hand-maintained), and
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
    ("vaspen/ui/surface_dialog.py", "SurfaceDialog"),
    ("vaspen/ui/rebox_dialog.py", "ReBoxDialog"),
    ("vaspen/ui/symmetry_dialog.py", "SymmetryDialog"),
    ("vaspen/ui/tools.py", "MainWindow"),
)


def _norm(text: str) -> str:
    """The hand-written .ts files store newlines as ``\\n`` escapes in
    some entries and as real newlines in others — normalize both sides
    before comparing."""
    return text.replace("\\n", "\n")


def _literals(module_rel: str) -> dict[str, set[str]]:
    """{context: string literals} — ``tr()`` calls (context = enclosing
    class) and ``QCoreApplication.translate("ctx", "literal")`` calls
    (context = the explicit first argument)."""
    tree = ast.parse(
        (Path(__file__).parent.parent / module_rel).read_text("utf-8"))
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
    wrong-context duplicates, a stale tooltip split)."""
    orphan: list[str] = []
    by_context: dict[str, set[str]] = {}
    for module, context in CHECKED_MODULES:
        by_context.setdefault(context, set()).update(_literals(module).get(
            context, set()))
    for context, literals in by_context.items():
        literals = {_norm(s) for s in literals}
        for ts_name, messages in (("zh.ts", ZH), ("en.ts", EN)):
            for src in messages.get(context, set()):
                if src not in literals:
                    orphan.append(f"{ts_name} {context}: {src!r}")
    assert not orphan, "\n".join(orphan)


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
