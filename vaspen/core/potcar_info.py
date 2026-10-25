"""POTCAR inspection (read-only): parse the concatenated PAW dataset
headers into per-section summaries for display.

A POTCAR is a concatenation of dataset sections. Each section carries
a TITEL line (functional + potential variant + date), ZVAL (valence
electrons), ENMAX (recommended cutoff) and ONE VRHFIN line per atomic
species it describes.

Pseudo-hydrogen: a handful of 3d transition-metal potentials (the
``*_h`` variants — Ti_h, V_h, Cr_h, Mn_h, Fe_h, Co_h, Ni_h, ...) embed
a SECOND species inside the same section: a pseudo-hydrogen whose
electron fills the otherwise empty d-band. Such a section therefore
has two VRHFIN lines (e.g. ``V:`` and ``H:``) and two ZVAL numbers.
That H is NOT a hydrogen of the structure — element counting from a
POTCAR must not treat it as one, and displays should flag it.
"""

from __future__ import annotations


def parse_potcar_sections(text: str) -> list[dict]:
    """Parse POTCAR text into section dicts.

    Each dict: ``functional`` (e.g. "PAW_PBE"), ``variant`` (e.g.
    "Fe_pv"), ``date``, ``elements`` (VRHFIN order — a section with a
    pseudo-hydrogen lists it after the real element), ``zval`` (first
    valence count as float), ``enmax`` (float, eV), and ``pseudo_h``
    (True when the section embeds a pseudo-hydrogen).
    """
    sections: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("TITEL"):
            if current is not None:
                sections.append(current)
            # "TITEL  = PAW_PBE Fe_pv 06Sep2007"
            after = line.split("=", 1)[1].split()
            current = {
                "functional": after[0] if after else "",
                "variant": after[1] if len(after) > 1 else "",
                "date": after[2] if len(after) > 2 else "",
                "elements": [],
                "zval": None,
                "enmax": None,
            }
            continue
        if current is None:
            continue
        if line.startswith("VRHFIN"):
            # "VRHFIN =Fe: 3d64s2" — often no space after '='
            after = line.split("=", 1)[1] if "=" in line else ""
            element = after.split(":")[0].strip()
            if element:
                current["elements"].append(element)
        elif line.startswith("ZVAL"):
            current["zval"] = _first_float(line)
        elif line.startswith("ENMAX"):
            current["enmax"] = _first_float(line)
        elif line.startswith("End of Dataset"):
            sections.append(current)
            current = None
    if current is not None:
        sections.append(current)
    for section in sections:
        # a REAL hydrogen section is H alone (H, H_s, H_1.75, ...);
        # an H that follows another element inside one section is the
        # embedded pseudo-hydrogen of an *_h potential
        section["pseudo_h"] = "H" in section["elements"][1:]
    return sections


def _first_float(line: str) -> float | None:
    """First float after the '=' (e.g. "ENMAX = 267.983 eV; ..." or a
    multi-number ZVAL)."""
    after = line.split("=", 1)[1] if "=" in line else ""
    for token in after.replace(";", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None
