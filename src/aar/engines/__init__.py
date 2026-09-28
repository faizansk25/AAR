"""Execution engines.

Every engine takes Arrow in and returns Arrow out. The contract is in
:mod:`aar.engines.base`; construction and the recorded fallback are in
:mod:`aar.engines.factory`.
"""

from .base import Engine, EngineCapabilities, PredicateCompiler  # noqa: F401
from .factory import (  # noqa: F401
    ENGINE_FACTORIES, FALLBACK_ORDER, create_engine,
)

__all__ = [
    "Engine", "EngineCapabilities", "PredicateCompiler",
    "ENGINE_FACTORIES", "FALLBACK_ORDER", "create_engine",
]
