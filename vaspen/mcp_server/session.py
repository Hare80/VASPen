"""Session state for the VASPen MCP server.

The server is a standalone process: one session holds the currently
loaded structure. :class:`StructureModel` is a QObject but works
without a QApplication (its signals simply have no observers headless
— verified against the whole core layer), so the GUI model is reused
as-is.

FastMCP runs sync tools on worker threads (anyio.to_thread), so every
tool body takes ``session.lock`` around session access.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ase import Atoms

from vaspen.core.structure import StructureModel


class ServerSession:
    """The currently loaded structure; the source path lives on the
    model itself (``StructureModel.save`` keeps it in sync)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._model: StructureModel | None = None

    # -- loading -------------------------------------------------------

    def load(self, path: str | Path) -> StructureModel:
        """Load a structure file and make it the session structure.

        Goes through ``StructureModel.load`` (the FileIO registry) so
        extension resolution, CIF space-group/occupancy sanitizing and
        pbc fixing behave exactly like the UI open path.
        """
        model = StructureModel()
        model.load(path)
        self._model = model
        return model

    # -- access --------------------------------------------------------

    @property
    def model(self) -> StructureModel | None:
        return self._model

    @property
    def filepath(self) -> str | None:
        """Source path of the current structure (None when absent)."""
        return self._model.filepath if self._model is not None else None

    def require_model(self) -> StructureModel:
        """The current model, or a clean error for the AI client."""
        if self._model is None or self._model.n_atoms == 0:
            raise ValueError(
                "No structure loaded — call open_structure(path) first."
            )
        return self._model

    def replace_atoms(
        self,
        atoms: Atoms,
        fixed_flags=None,
        magmoms=None,
    ) -> StructureModel:
        """Replace the current structure's atoms (undoable in-model).

        Used by transforms that rebuild the atom set (rotate,
        symmetrize, rebox): fixed flags and magmoms are carried over
        only when explicitly passed.
        """
        model = self.require_model()
        model.replace_atoms(atoms, fixed_flags=fixed_flags, magmoms=magmoms)
        return model
