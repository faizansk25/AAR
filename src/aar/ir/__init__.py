"""Internal Analytics IR - the stable internal representation of a pipeline.

The IR is intentionally *not* Substrait. Substrait is approaching 1.0 but is
not frozen, and it has no vocabulary for Excel ranges, Python UDFs, privacy
classifications, governance constraints or materialisation boundaries. AAR
keeps its own stable IR and treats Substrait as an interop format with
importers and exporters in :mod:`aar.ir.substrait`.
"""

from .nodes import (  # noqa: F401
    Agg, BinOp, CastExpr, Col, Expr, Func, JoinType, Lit, Node, NodeType,
    Privacy, ScanSpec, UnaryOp, WindowSpec, is_sink, is_source,
    topological_order,
)
from .identity import (  # noqa: F401
    SEMANTIC_ID_VERSION, UnstableSemanticIdentity, graph_node_id,
    operation_payload, resource_snapshot, semantic_operation_id, target_id,
)

__all__ = [
    "Agg", "BinOp", "CastExpr", "Col", "Expr", "Func", "JoinType", "Lit",
    "Node", "NodeType", "Privacy", "ScanSpec", "UnaryOp", "WindowSpec",
    "is_sink", "is_source", "topological_order",
    "SEMANTIC_ID_VERSION", "UnstableSemanticIdentity", "graph_node_id",
    "operation_payload", "resource_snapshot", "semantic_operation_id",
    "target_id",
]
