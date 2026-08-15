"""Interaction tools for the 3D viewport (mode state machines).

Tools receive mouse/key events from Viewport3D and drive its camera
state and signals. They never touch the data model or OpenGL — model
mutations happen in MainWindow, which listens to the viewport signals
the tools emit (Interaction → Command → Model → Renderer).

Navigation policy (settled): right/middle-drag pan and wheel zoom stay
viewport-global in every mode, and plain left-drag orbits unless the
active tool needs the drag for itself (move / shift+box-select /
anchored add-atom).
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from PySide6.QtCore import QCoreApplication, QPointF, QRectF, Qt
from PySide6.QtWidgets import QMessageBox, QWidget

from vaspen.core.transform import rotation_matrix


class ToolMode(Enum):
    """Exclusive interaction modes of the 3D viewport."""

    SELECT = auto()
    ADD_ATOM = auto()
    MOVE_ATOM = auto()
    ROTATE = auto()
    DELETE = auto()
    CREATE_BOND = auto()
    MEASURE_DISTANCE = auto()
    MEASURE_ANGLE = auto()
    MEASURE_TORSION = auto()


class Tool:
    """Base class for viewport interaction tools.

    The viewport calls mouse_press/move/release only for left-button
    events while this tool owns the press (mouse_press returned True).
    ``pos`` is in logical widget pixels.
    """

    def __init__(self, viewport) -> None:
        self._vp = viewport
        self._press_pos = None

    def mouse_press(self, event, pos) -> bool:
        """Handle a left-button press; True consumes it (tool owns the drag)."""
        return False

    def mouse_move(self, event, pos) -> None:
        """Handle mouse movement while the tool owns the press."""

    def mouse_release(self, event, pos) -> None:
        """Handle the left-button release (always called after an owned press)."""

    def key_press(self, event) -> bool:
        """Handle a key event; True consumes it."""
        return False

    def cancel(self) -> None:
        """Reset transient state (structure changed / mode left)."""

    def cursor(self) -> Qt.CursorShape:
        """Cursor shown while this tool is active."""
        return Qt.CursorShape.ArrowCursor

    def _target_blocked(self) -> bool:
        """True when the resolved drag target contains a frozen atom.

        Fixed flags live on the viewport (fed by set_structure(fixed=...)
        from the model). Any fixed direction counts (settled policy) —
        a mixed fixed/free group is blocked as a whole, matching the
        model-side commit guard. A shape mismatch (stale flags) reads
        as not blocked — the model guard remains the backstop.
        """
        flags = self._vp.fixed_flags
        if len(flags) != len(self._vp._atom_pos) or not self._indices:
            return False
        return bool(flags[list(self._indices)].any())


def _orbit_to(vp, pos) -> None:
    """Plain left-drag rotates the camera (viewport-wide policy)."""
    vp.orbit(pos.x() - vp._last_pos.x(), pos.y() - vp._last_pos.y())


class SelectTool(Tool):
    """Click = select; Ctrl+click = add; Shift+click = toggle;
    Shift+drag = box select; plain drag = orbit (navigation is always
    available in every mode — settled camera policy). Esc cancels an
    in-progress box selection."""

    def __init__(self, viewport) -> None:
        super().__init__(viewport)
        self._cancelled = False

    def mouse_press(self, event, pos) -> bool:
        self._press_pos = pos
        self._cancelled = False
        return True

    def mouse_move(self, event, pos) -> None:
        vp = self._vp
        if not vp._dragged or self._press_pos is None:
            return
        if event.modifiers() & Qt.ShiftModifier:
            vp.set_rubber_band(QRectF(self._press_pos, pos).normalized())
        else:
            _orbit_to(vp, pos)

    def mouse_release(self, event, pos) -> None:
        vp = self._vp
        if self._cancelled:
            self._cancelled = False
            return
        if vp._dragged:
            if event.modifiers() & Qt.ShiftModifier:
                vp.set_rubber_band(None)
                self._box_select(self._press_pos, pos)
            return
        hit = vp.pick(pos)
        if hit is None:
            vp.background_clicked.emit()
            return
        kind, index = hit
        mods = event.modifiers()
        if kind == "atom":
            mode = ("add" if mods & Qt.ControlModifier else
                    "toggle" if mods & Qt.ShiftModifier else
                    "replace")
            vp.atoms_selected.emit([index], mode)
        else:
            vp.bond_clicked.emit(index)

    def key_press(self, event) -> bool:
        if event.key() == Qt.Key_Escape and self._press_pos is not None:
            # Esc mid box-selection: cancel it (release then does nothing)
            self._cancelled = True
            self._vp.set_rubber_band(None)
            return True
        return False

    def cancel(self) -> None:
        self._cancelled = False
        self._vp.set_rubber_band(None)

    def _box_select(self, p1, p2) -> None:
        """Select atoms whose projected centers fall inside the rubber band."""
        vp = self._vp
        rect = QRectF(p1, p2).normalized()
        hits = []
        for i in range(len(vp._atom_pos)):
            s = vp.project_to_screen(vp._atom_pos[i])
            if s is not None and rect.contains(QPointF(s[0], s[1])):
                hits.append(i)
        vp.atoms_selected.emit(hits, "replace")


class AddAtomTool(Tool):
    """Click empty space → place an atom there (free placement).

    Press on an existing atom and drag → place a new atom snapped to the
    IDEAL BOND LENGTH (sum of covalent radii of the anchor and the
    toolbar element) along the cursor direction, with a ghost preview.
    Click on an atom without dragging → place at the ideal distance in
    the cursor direction. Plain drag on empty space orbits.
    """

    def __init__(self, viewport) -> None:
        super().__init__(viewport)
        self._anchor: int | None = None

    def mouse_press(self, event, pos) -> bool:
        self._press_pos = pos
        hit = self._vp.pick(pos)
        self._anchor = hit[1] if hit is not None and hit[0] == "atom" else None
        return True

    def mouse_move(self, event, pos) -> None:
        vp = self._vp
        if not vp._dragged:
            return
        if self._anchor is None:
            _orbit_to(vp, pos)
            return
        vp.preview_ghost_atom(self._snapped(pos))
        # ghost bond preview: the placed atom bonds to the anchor
        vp.preview_ghost_bond(vp._atom_pos[self._anchor])

    def mouse_release(self, event, pos) -> None:
        vp = self._vp
        anchor = self._anchor
        if vp._dragged:
            if anchor is not None:
                vp.atom_place_requested.emit(self._snapped(pos), anchor)
            self._clear_preview()
            return
        hit = vp.pick(pos)
        if hit is not None and hit[0] == "atom":
            # Click on an atom: grow a new bonded atom at the ideal
            # length, in the cursor direction (from the anchor toward
            # the camera-plane point under the cursor).
            vp.atom_place_requested.emit(self._snapped(pos), anchor)
            self._clear_preview()
            return
        if hit is not None:
            return  # click on a bond: ignore
        vp.atom_place_requested.emit(vp.screen_to_world(pos), None)

    def _clear_preview(self) -> None:
        self._anchor = None
        self._vp.preview_ghost_atom(None)
        self._vp.preview_ghost_bond(None)

    def cancel(self) -> None:
        self._clear_preview()

    def _ideal_distance(self) -> float:
        """Sum of covalent radii of the anchor atom and the new element."""
        vp = self._vp
        z_anchor = atomic_numbers.get(vp._atoms[self._anchor].symbol, 0)
        z_new = atomic_numbers.get(vp.current_element, 0)
        r_anchor = float(covalent_radii[z_anchor]) if 0 < z_anchor < len(covalent_radii) else 0.77
        r_new = float(covalent_radii[z_new]) if 0 < z_new < len(covalent_radii) else 0.77
        return r_anchor + r_new

    def _snapped(self, pos) -> np.ndarray:
        """Place the new atom at the ideal bond length from the anchor,
        preferring the SCREEN-OUTWARD direction (toward the viewer):
        the cursor controls the lateral angle in the camera plane, while
        the depth component always points toward the camera — the new
        atom never grows into the screen."""
        vp = self._vp
        anchor = np.asarray(vp._atom_pos[self._anchor], dtype=float)
        world = vp.screen_to_world(pos)
        _proj, view = vp._camera_matrices()
        outward = np.asarray(view[2, :3], dtype=float)  # toward the viewer
        rel = world - anchor
        lateral = rel - outward * float(rel @ outward)  # camera-plane part
        direction = lateral + outward
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = outward
        else:
            direction = direction / norm
        return anchor + direction * self._ideal_distance()

    def cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.CrossCursor


class MoveAtomTool(Tool):
    """Drag to move atoms. Move-set priority (settled with the user):

    1. ANY non-empty selection (even a single atom) → move exactly the
       selected atoms — the grabbed atom does not need to belong to it.
    2. Otherwise → move the bond-connected component of the grabbed
       atom (the whole molecule for molecular structures; the whole
       structure for a connected crystal; just the atom when isolated).

    The drag uses the fast preview path (no model calls); release
    commits as ONE undoable model call. Esc (or cancel) restores the
    start positions. Click without drag selects the primary atom.

    Axis constraint: pressing X/Y/Z during a drag locks the motion to
    that axis (toggle — pressing the same key again unlocks). For
    periodic structures the constraint follows the LATTICE vector
    direction so the fractional shift has a single component (pure
    fractional-axis semantics); otherwise it follows the world axis.
    """

    def __init__(self, viewport) -> None:
        super().__init__(viewport)
        self._indices: list[int] = []
        self._start_cart: np.ndarray | None = None   # K×3
        self._start_frac: np.ndarray | None = None   # K×3 (periodic only)
        self._press_world: np.ndarray | None = None
        self._moved = False
        self._axis: int | None = None  # 0/1/2 while X/Y/Z lock is active
        self._last_pos = None

    @staticmethod
    def _connected(index: int, bonds) -> list[int]:
        """Bond-connected component containing ``index`` (BFS)."""
        adjacency: dict[int, set[int]] = {}
        for b in bonds:
            adjacency.setdefault(b.i, set()).add(b.j)
            adjacency.setdefault(b.j, set()).add(b.i)
        seen = {index}
        frontier = [index]
        while frontier:
            atom = frontier.pop()
            for nb in adjacency.get(atom, ()):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        return sorted(seen)

    def mouse_press(self, event, pos) -> bool:
        vp = self._vp
        hit = vp.pick(pos)
        if hit is None or hit[0] != "atom":
            return False  # empty space → default navigation (orbit)
        index = hit[1]
        sel = vp._selected_indices
        if sel:
            # priority 1 (settled 2026-08-13): any selection — move
            # exactly the selected atoms, no matter which atom is grabbed
            self._indices = sorted(sel)
        else:
            self._indices = self._connected(index, vp._bonds)
        if self._target_blocked():
            # Frozen atoms must not move, not even in the drag preview —
            # swallow the press entirely (no preview state, no commit).
            vp.frozen_drag_blocked.emit()
            self._indices = []
            return True
        self._start_cart = np.asarray(vp._atom_pos[self._indices], dtype=float)
        atoms = vp._atoms
        if (atoms is not None and atoms.get_cell().rank == 3
                and atoms.get_pbc().any()):
            self._start_frac = atoms.get_scaled_positions()[self._indices].copy()
        else:
            self._start_frac = None
        self._press_world = vp.screen_to_world(pos)
        return True

    def mouse_move(self, event, pos) -> None:
        if not self._indices or not self._vp._dragged:
            return
        self._last_pos = pos
        self._vp.preview_atom_positions(self._indices, self._targets(pos))
        self._moved = True

    def mouse_release(self, event, pos) -> None:
        indices, moved = self._indices, self._moved
        # Compute the commit positions BEFORE _reset clears the drag state
        targets = self._targets(pos) if indices and moved else None
        self._reset()
        if not indices:
            return
        if moved:
            self._vp.atoms_moved.emit(indices, targets)
        else:
            self._vp.atoms_selected.emit([indices[0]], "replace")

    def key_press(self, event) -> bool:
        key = event.key()
        if key == Qt.Key_Escape and self._indices:
            self._vp.preview_atom_positions(self._indices, self._start_cart)
            self._reset()
            return True
        axis_map = {Qt.Key_X: 0, Qt.Key_Y: 1, Qt.Key_Z: 2}
        if key in axis_map and self._indices:
            axis = axis_map[key]
            self._axis = None if self._axis == axis else axis
            # Re-apply the constraint to the current drag immediately
            if self._moved and self._last_pos is not None:
                self._vp.preview_atom_positions(
                    self._indices, self._targets(self._last_pos))
            return True
        return False

    def cancel(self) -> None:
        if self._indices and self._start_cart is not None:
            self._vp.preview_atom_positions(self._indices, self._start_cart)
        self._reset()

    def _reset(self) -> None:
        self._indices = []
        self._start_cart = None
        self._start_frac = None
        self._press_world = None
        self._moved = False
        self._axis = None
        self._last_pos = None

    def _targets(self, pos) -> np.ndarray:
        """Start positions + cursor delta, wrapped per atom.

        Periodic: the whole GROUP is shifted by ONE fractional delta,
        then each atom wraps %1 — group-relative geometry is preserved
        across the cell boundary (bonds never tear).

        Axis lock (X/Y/Z during drag): the delta is projected onto the
        lattice-vector direction when periodic (single fractional
        component), onto the world axis otherwise.
        """
        world = self._vp.screen_to_world(pos)
        delta = world - self._press_world
        if self._axis is not None:
            if self._start_frac is not None:
                cell = np.asarray(self._vp._atoms.get_cell().array, dtype=float)
                axis_vec = cell[:, self._axis]
                n2 = float(axis_vec @ axis_vec)
                if n2 > 1e-12:
                    delta = axis_vec * (float(delta @ axis_vec) / n2)
            else:
                e = np.zeros(3)
                e[self._axis] = 1.0
                delta = e * float(delta @ e)
        if self._start_frac is not None:
            cell = np.asarray(self._vp._atoms.get_cell().array, dtype=float)
            frac_delta = np.linalg.solve(cell.T, delta)
            new_frac = self._start_frac + frac_delta
            for ax in range(3):
                if self._vp._atoms.get_pbc()[ax]:
                    new_frac[:, ax] %= 1.0
            return new_frac @ cell
        return self._start_cart + delta

    def cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.SizeAllCursor


class RotateAtomTool(Tool):
    """Drag to ROTATE a group of atoms about the group CENTROID (mouse-only).

    Group priority is identical to Move (settled 2026-08-13): any
    non-empty selection > bond-connected component of the grabbed atom.
    Horizontal drag rotates about the camera up axis, vertical drag
    about the camera right axis (0.5°/px, near side follows the
    cursor). Uses the fast preview path; release commits as ONE
    undoable model call (atoms_moved); Esc restores the start
    positions; click without drag selects the primary atom. Periodic
    axes wrap %1 per atom after rotation.
    """

    def __init__(self, viewport) -> None:
        super().__init__(viewport)
        self._indices: list[int] = []
        self._start_cart: np.ndarray | None = None   # K×3
        self._center: np.ndarray | None = None       # group centroid
        self._start_frac: np.ndarray | None = None   # K×3 (periodic only)
        self._press_pos = None
        self._moved = False

    @staticmethod
    def _connected(index: int, bonds) -> list[int]:
        return MoveAtomTool._connected(index, bonds)

    def mouse_press(self, event, pos) -> bool:
        vp = self._vp
        hit = vp.pick(pos)
        if hit is None or hit[0] != "atom":
            return False  # empty space → default navigation (orbit)
        index = hit[1]
        sel = vp._selected_indices
        if sel:
            self._indices = sorted(sel)   # priority 1: any selection
        else:
            self._indices = self._connected(index, vp._bonds)
        if self._target_blocked():
            # Frozen atoms must not rotate, not even in the drag preview.
            vp.frozen_drag_blocked.emit()
            self._indices = []
            return True
        self._start_cart = np.asarray(vp._atom_pos[self._indices], dtype=float)
        self._center = self._start_cart.mean(axis=0)
        atoms = vp._atoms
        if (atoms is not None and atoms.get_cell().rank == 3
                and atoms.get_pbc().any()):
            self._start_frac = atoms.get_scaled_positions()[self._indices].copy()
        else:
            self._start_frac = None
        self._press_pos = pos
        return True

    def mouse_move(self, event, pos) -> None:
        if not self._indices or not self._vp._dragged:
            return
        self._vp.preview_atom_positions(self._indices, self._targets(pos))
        self._moved = True

    def mouse_release(self, event, pos) -> None:
        indices, moved = self._indices, self._moved
        targets = self._targets(pos) if indices and moved else None
        self._reset()
        if not indices:
            return
        if moved:
            self._vp.atoms_moved.emit(indices, targets)
        else:
            self._vp.atoms_selected.emit([indices[0]], "replace")

    def key_press(self, event) -> bool:
        if event.key() == Qt.Key_Escape and self._indices:
            self._vp.preview_atom_positions(self._indices, self._start_cart)
            self._reset()
            return True
        return False

    def cancel(self) -> None:
        if self._indices and self._start_cart is not None:
            self._vp.preview_atom_positions(self._indices, self._start_cart)
        self._reset()

    def _reset(self) -> None:
        self._indices = []
        self._start_cart = None
        self._center = None
        self._start_frac = None
        self._press_pos = None
        self._moved = False

    def _targets(self, pos) -> np.ndarray:
        """Start positions rotated about the group CENTROID.

        Horizontal drag → rotation about the camera up axis (near side
        follows the cursor); vertical drag → about the camera right
        axis with the tilt convention the user chose (drag up tips the
        structure toward the viewer). Periodic: each atom wraps %1
        along pbc axes afterwards.
        """
        dx = pos.x() - self._press_pos.x()
        dy = pos.y() - self._press_pos.y()
        _proj, view = self._vp._camera_matrices()
        rot = view[:3, :3]
        up = np.asarray(rot[1, :], dtype=float)      # camera up
        right = np.asarray(rot[0, :], dtype=float)   # camera right
        angle_up = dx * 0.5
        angle_right = dy * 0.5
        R = rotation_matrix(up, angle_up) @ rotation_matrix(right, angle_right)
        rotated = (self._start_cart - self._center) @ R.T + self._center
        if self._start_frac is not None:
            cell = np.asarray(self._vp._atoms.get_cell().array, dtype=float)
            scaled = np.linalg.solve(cell.T, rotated.T).T
            for ax in range(3):
                if self._vp._atoms.get_pbc()[ax]:
                    scaled[:, ax] %= 1.0
            return scaled @ cell
        return rotated

    def cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.SizeAllCursor


class CreateBondTool(Tool):
    """Click atom A, click atom B → bond_created(A, B). If A belongs to
    a two-atom selection, the bond between the two selected atoms is
    created immediately. Click empty space or A again cancels. A is
    shown with the preview highlight. Plain drag orbits."""

    def __init__(self, viewport) -> None:
        super().__init__(viewport)
        self._first: int | None = None

    def mouse_press(self, event, pos) -> bool:
        self._press_pos = pos
        return True

    def mouse_move(self, event, pos) -> None:
        if self._vp._dragged:
            _orbit_to(self._vp, pos)

    def mouse_release(self, event, pos) -> None:
        vp = self._vp
        if vp._dragged:
            return
        hit = vp.pick(pos)
        if hit is None or hit[0] != "atom":
            self.cancel()
            return
        index = hit[1]
        if self._first is None:
            sel = vp._selected_indices
            if index in sel and len(sel) == 2:
                # two atoms selected → bond them directly
                other = next(iter(sel - {index}))
                vp.bond_created.emit(index, other)
                return
            self._first = index
            vp.set_preview_highlight({index})
        elif index == self._first:
            self.cancel()
        else:
            first = self._first
            self.cancel()  # clears preview + first (must run BEFORE emit)
            vp.bond_created.emit(first, index)

    def cancel(self) -> None:
        self._first = None
        self._vp.set_preview_highlight(None)

    def cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.PointingHandCursor


class DeleteTool(Tool):
    """Click to delete. Selection-aware: clicking an atom that belongs
    to a multi-atom selection (or clicking empty space while atoms are
    selected) deletes the whole selection; clicking an unselected atom
    deletes just that atom; clicking a bond deletes the bond."""

    def mouse_press(self, event, pos) -> bool:
        self._press_pos = pos
        return True

    def mouse_move(self, event, pos) -> None:
        if self._vp._dragged:
            _orbit_to(self._vp, pos)

    def mouse_release(self, event, pos) -> None:
        vp = self._vp
        if vp._dragged:
            return
        hit = vp.pick(pos)
        if hit is None:
            if vp._selected_indices:
                vp.delete_requested.emit("selection", 0)
            return
        kind, index = hit
        if kind == "atom":
            if index in vp._selected_indices and len(vp._selected_indices) > 1:
                vp.delete_requested.emit("selection", 0)
            else:
                vp.delete_requested.emit("atom", index)
        else:
            vp.delete_requested.emit("bond", index)

    def cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.ForbiddenCursor


class MeasureTool(Tool):
    """Click N atoms in sequence → measurement_added(kind, indices).

    Picked atoms are shown with the preview highlight; clicking empty
    space (or Esc) cancels the pending picks. Plain drag orbits.
    """

    def __init__(self, viewport, kind: str, need: int) -> None:
        super().__init__(viewport)
        self._kind = kind
        self._need = need
        self._picked: list[int] = []

    def mouse_press(self, event, pos) -> bool:
        self._press_pos = pos
        return True

    def mouse_move(self, event, pos) -> None:
        if self._vp._dragged:
            _orbit_to(self._vp, pos)

    def mouse_release(self, event, pos) -> None:
        vp = self._vp
        if vp._dragged:
            return
        hit = vp.pick(pos)
        if hit is None or hit[0] != "atom":
            self.cancel()
            return
        self._picked.append(hit[1])
        vp.set_preview_highlight(set(self._picked))
        if len(self._picked) >= self._need:
            kind, picked = self._kind, list(self._picked[:self._need])
            self.cancel()
            vp.measurement_added.emit(kind, picked)

    def key_press(self, event) -> bool:
        if event.key() == Qt.Key_Escape:
            self.cancel()
            return True
        return False

    def cancel(self) -> None:
        self._picked = []
        self._vp.set_preview_highlight(None)

    def cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.CrossCursor


TOOL_CLASSES = {
    ToolMode.SELECT: SelectTool,
    ToolMode.ADD_ATOM: AddAtomTool,
    ToolMode.MOVE_ATOM: MoveAtomTool,
    ToolMode.ROTATE: RotateAtomTool,
    ToolMode.CREATE_BOND: CreateBondTool,
    ToolMode.DELETE: DeleteTool,
}

_MEASURE_SPECS = {
    ToolMode.MEASURE_DISTANCE: ("distance", 2),
    ToolMode.MEASURE_ANGLE: ("angle", 3),
    ToolMode.MEASURE_TORSION: ("dihedral", 4),
}


def make_tool(mode: ToolMode, viewport) -> Tool:
    """Instantiate the tool for a mode."""
    if mode in _MEASURE_SPECS:
        kind, need = _MEASURE_SPECS[mode]
        return MeasureTool(viewport, kind, need)
    return TOOL_CLASSES[mode](viewport)


def confirm_disorder_loss(parent: QWidget, operation: str) -> bool:
    """Confirm before an operation that discards partial occupancy.

    The single implementation of the settled §7.8 message pattern
    (cleave / re-box / symmetrize / supercell share it — the previous
    four inline copies were byte-identical apart from the operation
    phrase). ``operation`` is an already-translated phrase ("Cleaving",
    "Re-boxing", ...). Returns True to continue.

    ``QCoreApplication.translate("MainWindow", ...)``: the strings
    live in the MainWindow context of the .ts files (hand-maintained,
    §11.1) — a bare tr() here would land in no context.
    """
    reply = QMessageBox.warning(
        parent,
        QCoreApplication.translate("MainWindow", "Partial Occupancy"),
        QCoreApplication.translate(
            "MainWindow",
            "This structure has partial occupancy (disorder).\n"
            "{} will discard the fractional occupancy information.\n\n"
            "Continue?").format(operation),
        QMessageBox.Yes | QMessageBox.Cancel,
        QMessageBox.Cancel,
    )
    return reply == QMessageBox.Yes
