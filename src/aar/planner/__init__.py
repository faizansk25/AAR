"""Segment-based adaptive planner.

Optimising per operation is wrong; optimising per segment is right. See
:mod:`aar.planner.planner` for the objective and the reasoning.
"""

from .planner import (  # noqa: F401
    AdaptivePlanner, NodeTypeAffinity, Plan, Segment, SegmentPlan,
    decompose_into_segments, estimate_bytes, self_device,
)

__all__ = [
    "AdaptivePlanner", "NodeTypeAffinity", "Plan", "Segment", "SegmentPlan",
    "decompose_into_segments", "estimate_bytes", "self_device",
]
