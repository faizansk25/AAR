"""Engine capability registry: which engine can do what, and what is installed.

Split into two answers that are deliberately kept apart:

* **Declared** - what an engine could do, from a static catalogue. This is
  what lets ``aar explain plan`` produce a plan on a machine that could not
  run it, which is how an analyst reviews a pipeline before scheduling it.
* **Available** - what is importable and hardware-compatible right now.

The planner uses the intersection. See :mod:`aar.capability.registry`.
"""

from .registry import (  # noqa: F401
    Capability, CapabilityRegistry, Device, EngineSpec, ENGINES, ENGINE_IDS,
    Tier, default_registry, probe_engine,
)

__all__ = [
    "Capability", "CapabilityRegistry", "Device", "EngineSpec", "ENGINES",
    "ENGINE_IDS", "Tier", "default_registry", "probe_engine",
]
