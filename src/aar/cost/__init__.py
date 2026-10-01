"""Cost model: what a node will cost on an engine, and what moving it costs.

The specification's central GPU insight lives in this package: a kernel can be
several times faster and still lose overall once host-to-device and
device-to-host transfers are counted. That outcome must fall out of
arithmetic, not be asserted, and it does.
"""

from .history import ExecutionHistory, ExecutionRecord  # noqa: F401
from .model import (  # noqa: F401
    CostBreakdown, CostModel, DEFAULT_PRIORS, EstimationError, EstimationLog,
    Priors, ResourceBudget, TransferProfile, default_cost_model,
    node_operation,
)

__all__ = [
    "CostBreakdown", "CostModel", "DEFAULT_PRIORS", "EstimationError",
    "EstimationLog", "ExecutionHistory", "ExecutionRecord", "Priors",
    "ResourceBudget", "TransferProfile", "default_cost_model",
    "node_operation",
]
