"""Regenerate vaspen_en.ts as a full mirror of vaspen_zh.ts.

SETTLED POLICY (CLAUDE.md §7.8.1) — do not re-litigate: English IS the
source language, so vaspen_en.ts must never be hand-edited. Every
message's translation equals its source. This script is the ONLY
regeneration path:

    python scripts/sync_en_ts.py

zh.ts is the source of truth. The existing en.ts is updated IN PLACE —
its own formatting (DOCTYPE, indentation, line endings) is preserved
and only drifted messages change, so the diff stays reviewable.
Message blocks are rebuilt per context in zh.ts order: entries whose
translation already equals their source are kept byte-verbatim,
drifted translations are rewritten, new messages are appended and
messages absent from zh.ts are dropped. The result is parsed back to
verify the mirror property before it is written (pinned by
tests/test_i18n.py::test_en_ts_is_a_full_mirror_of_zh_ts).
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
I18N = ROOT / "vaspen" / "resources" / "i18n"
ZH = I18N / "vaspen_zh.ts"
EN = I18N / "vaspen_en.ts"

_CTX_RE = re.compile(r"<context>.*?</context>", re.S)
_NAME_RE = re.compile(r"<name>(.*?)</name>", re.S)
#: Minimal en.ts document used to bootstrap the mirror when en.ts is
#: missing/empty (a fresh clone must be able to regenerate it).
_BOOTSTRAP = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<TS version="2.1" language="en_US" sourcelanguage="en_US">\n'
    "</TS>\n"
)
#: Any-position match (zh.ts has single-line runs of several <message>
#: blocks on one line — a line-anchored regex would miss them).
_MSG_RE = re.compile(r"<message[^>]*>.*?</message>", re.S)
#: Line-anchored WITH the indent captured, for en.ts block extraction:
#: rebuilt contexts keep message lines byte-verbatim (en.ts uses
#: 4-space message indent / 2-space context indent — different from
#: zh.ts, and its messages are one per line).
_MSG_IND_RE = re.compile(r"^([ \t]*)<message[^>]*>.*?</message>", re.M | re.S)
_SRC_RE = re.compile(r"<source[^>]*>(.*?)</source>", re.S)
_TRANS_RE = re.compile(r"<translation[^>]*>.*?</translation>", re.S)


def _zh_sources(zh_text: str) -> dict[str, list[str]]:
    """{context: ordered source texts (raw, entity-escaped)} of zh.ts."""
    out: dict[str, list[str]] = {}
    for ctx in _CTX_RE.findall(zh_text):
        name = _NAME_RE.search(ctx)
        if not name:
            continue
        sources = []
        for match in _MSG_RE.finditer(ctx):
            src = _SRC_RE.search(match.group(0))
            if src and src.group(1) is not None:
                sources.append(src.group(1))
        out.setdefault(name.group(1), []).extend(sources)
    return out


def _fix_translation(block: str) -> str:
    """Rewrite a message's translation to its source (if drifted)."""
    src = _SRC_RE.search(block)
    if not src or src.group(1) is None:
        return block
    trans = _TRANS_RE.search(block)
    if trans and trans.group(0) == f"<translation>{src.group(1)}</translation>":
        return block  # already a mirror — keep byte-verbatim
    return _TRANS_RE.sub(
        lambda _: f"<translation>{src.group(1)}</translation>", block)


def _render_message(indent: str, source: str) -> str:
    """A new <message> block in the en.ts house style."""
    inner = indent + "  "
    return (f"{indent}<message>\n"
            f"{inner}<source>{source}</source>\n"
            f"{inner}<translation>{source}</translation>\n"
            f"{indent}</message>")


def _render_context(
    name: str, sources: list[str], ctx_indent: str = "  ",
) -> str:
    """A whole <context> block in the en.ts house style."""
    msgs = "\n".join(_render_message("    ", s) for s in sources)
    return (f"{ctx_indent}<context>\n"
            f"{ctx_indent}  <name>{name}</name>\n"
            f"{msgs}\n"
            f"{ctx_indent}</context>")


def mirror(zh_text: str, en_text: str) -> str:
    """en.ts updated in place: translation == source for every zh.ts
    message; contexts/messages absent from zh.ts are dropped."""
    zh_sources = _zh_sources(zh_text)
    out: list[str] = []
    pos = 0
    for ctx in _CTX_RE.finditer(en_text):
        out.append(en_text[pos:ctx.start()])
        ctx_text = ctx.group(0)
        name_m = _NAME_RE.search(ctx_text)
        if not name_m:
            out.append(ctx_text)
            pos = ctx.end()
            continue
        name = name_m.group(1)
        wanted = zh_sources.get(name)
        if wanted is None:
            pos = ctx.end()  # context no longer exists in zh.ts
            continue
        ind_m = re.search(r"^([ \t]*)<message", ctx_text, re.M)
        msg_indent = ind_m.group(1) if ind_m else "    "
        end_m = re.search(r"^([ \t]*)</context>", ctx_text, re.M)
        ctx_indent = end_m.group(1) if end_m else "  "
        existing: dict[str, str] = {}
        for match in _MSG_IND_RE.finditer(ctx_text):
            src = _SRC_RE.search(match.group(0))
            if src and src.group(1) is not None:
                existing[src.group(1)] = match.group(0)
        rebuilt = [
            _fix_translation(existing[src])
            if src in existing else _render_message(msg_indent, src)
            for src in wanted
        ]
        out.append(
            ctx_text[:name_m.end()]
            + "\n"
            + "\n".join(rebuilt)
            + "\n"
            + ctx_indent
            + "</context>"
        )
        pos = ctx.end()
    out.append(en_text[pos:])

    # Contexts present in zh.ts but missing from en.ts (e.g. a brand-new
    # context) — rendered in the en.ts house style and inserted before
    # the closing </TS> tag.
    en_names = {_NAME_RE.search(c).group(1) for c in _CTX_RE.findall(en_text)
                if _NAME_RE.search(c)}
    missing = [n for n in zh_sources if n not in en_names]
    if missing:
        tail = out[-1]
        ts_pos = tail.find("</TS>")
        if ts_pos == -1:
            # Empty/missing en.ts — synthesize the document skeleton.
            tail = _BOOTSTRAP
            ts_pos = tail.find("</TS>")
        rendered = "\n".join(
            _render_context(name, zh_sources[name]) for name in missing)
        out[-1] = tail[:ts_pos] + rendered + "\n" + tail[ts_pos:]
    return "".join(out)


def _verify_mirror(text: str) -> list[str]:
    """Parse the generated content and list any non-mirror entry."""
    broken = []
    root = ET.fromstring(text)
    for ctx in root.findall("context"):
        name = ctx.find("name").text
        for msg in ctx.findall("message"):
            source = msg.find("source")
            translation = msg.find("translation")
            if (source is not None and source.text is not None
                    and (translation is None
                         or translation.text != source.text)):
                broken.append(f"{name}: {source.text!r}")
    return broken


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 if zh.ts is missing or the mirror check fails.
    """
    if not ZH.exists():
        print(f"zh.ts not found: {ZH}", file=sys.stderr)
        return 1
    en_text = EN.read_text("utf-8") if EN.exists() else _BOOTSTRAP
    generated = mirror(ZH.read_text("utf-8"), en_text)
    # Preserve en.ts's own line endings (CRLF on this machine's
    # checkout; regenerating on Linux must not flip the whole file).
    if "\r\n" in en_text:
        generated = generated.replace("\n", "\r\n")
    try:
        broken = _verify_mirror(generated)
    except ET.ParseError as exc:
        print(f"generated en.ts does not parse: {exc}", file=sys.stderr)
        return 1
    if broken:
        print("mirror verification failed:", file=sys.stderr)
        for entry in broken:
            print(f"  {entry}", file=sys.stderr)
        return 1
    # write_bytes: no newline translation (write_text would double the
    # CRs of CRLF content on Windows).
    EN.write_bytes(generated.encode("utf-8"))
    n_messages = len(_MSG_RE.findall(generated))
    print(f"en.ts regenerated from zh.ts ({n_messages} messages) -> {EN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
