"""Arrow-native interchange: the only currency that crosses an engine boundary.

See :mod:`aar.interchange.table` for why this exists and what it carries.
"""

from .table import (  # noqa: F401
    Table, arrow_to_canonical, canonical_to_arrow, require_arrow,
)

__all__ = ["Table", "arrow_to_canonical", "canonical_to_arrow", "require_arrow"]
