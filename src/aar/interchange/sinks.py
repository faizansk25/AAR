"""Where a write goes, without an engine importing a connector to find out.

`engines/arrow_engine.py` used to answer "how do I write an Excel file" by
doing `from ..connectors.excel import write_excel` inside a method. That is
an import cycle by any static analysis - `connectors` is above `engines` in
the layering, and an engine reaching upward to ask a connector how to write
is the wrong direction. It was deferred inside a function to keep
`import aar` cheap, which hid the cycle without fixing it.

The fix is a table, not an import. This module names the target module and
the attribute; the import happens at the point of use, which is the only
moment a connector is actually wanted. Two consequences:

- `engines` no longer mentions `connectors` at all, so the edge is gone;
- nothing is registered at import time, so there is no ordering problem -
  a connector that is never imported is simply never needed.

`interchange` is the right home for it because it is the layer both sides
already depend on: engines take Arrow in and give Arrow out, connectors
read and write those tables, and neither knows about the other.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["sink_for", "SINKS", "SinkUnavailable"]

#: format -> (module, attribute) for formats whose writer is *connector*
#: code. Parquet and CSV are deliberately absent: `ArrowEngine.write` calls
#: `pyarrow.parquet.write_table` and `pyarrow.csv.write_csv` directly, because
#: those are two pyarrow calls rather than a connector that has to reason
#: about schemas, classification tags and header detection. Only Excel needs
#: the machinery, and only Excel is a real cycle.
#:
#: Listed by hand rather than discovered by scanning, so a rename is a loud
#: failure at lookup time with the format named, instead of a writer that
#: quietly stops being reachable. `test_every_declared_sink_really_imports`
#: checks every entry, which is what stops this table drifting into naming a
#: module that does not exist.
SINKS: dict[str, tuple[str, str]] = {
    "excel": ("aar.connectors.excel", "write_excel"),
}


class SinkUnavailable(RuntimeError):
    """A write format AAR knows the name of but cannot currently write.

    Raised rather than returned as ``None`` so a caller cannot mistake "this
    format has no writer" for "the write succeeded and produced nothing" -
    the same distinction the failure registry insists on everywhere else.
    """


def sink_for(fmt: str) -> Any:
    """Return the writer callable for ``fmt``, importing it on demand.

    Raises `SinkUnavailable` naming the format and what was needed, because a
    user who asked for a CSV write and got nothing needs to know which
    package to install or which format to use.
    """
    key = fmt.lower()
    if key in ("parquet", "csv"):
        raise SinkUnavailable(
            f"{key!r} is written inline by ArrowEngine, not through this "
            f"table; call the engine's write() rather than sink_for()")
    try:
        module_path, attr = SINKS[key]
    except KeyError as exc:
        raise SinkUnavailable(
            f"no writer for format {fmt!r}; the connector-backed formats "
            f"are {', '.join(sorted(SINKS))}"
        ) from exc
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SinkUnavailable(
            f"format {fmt!r} needs {module_path}, which could not be "
            f"imported: {exc}"
        ) from exc
    return getattr(module, attr)

