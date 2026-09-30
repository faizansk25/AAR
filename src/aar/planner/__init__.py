"""Segment-based adaptive planner.

Optimising per operation is wrong; optimising per segment is right. See
:mod:`aar.planner.planner` for the objective and the reasoning.
"""

from __future__ import annotations


def implemented_engine_ids() -> set[str]:
    """Engine ids this build can actually construct, not merely declare.

    The join between the capability catalogue (which is aspirational: sixteen
    engines, so a distributed engine can be *described* before anyone has
    written it) and `aar.engines.factory` (which is real: an id appears only
    when a class exists behind it).

    The catalogue deliberately does not know about the factory - the factory
    imports the catalogue, so the dependency runs one way and this module is
    the only place the two can be compared without a cycle.

    Returns `{"arrow"}` if the engines package cannot be imported at all,
    because Arrow is the one engine with no third-party dependency and is
    therefore the one thing that can always be assumed to build.
    """
    try:
        from ..engines.factory import ENGINE_FACTORIES
    except Exception:  # noqa: BLE001 - a broken install must not crash a plan
        return {"arrow"}
    return set(ENGINE_FACTORIES)


from .planner import (  # noqa: E402,F401
    AdaptivePlanner, NodeTypeAffinity, Plan, Segment, SegmentPlan,
    decompose_into_segments, estimate_bytes, pushdown_report,
    segment_predecessors, self_device,
)

__all__ = [
    "AdaptivePlanner", "NodeTypeAffinity", "Plan", "Segment", "SegmentPlan",
    "decompose_into_segments", "estimate_bytes", "self_device",
    "implemented_engine_ids", "pushdown_report", "segment_predecessors",
]
