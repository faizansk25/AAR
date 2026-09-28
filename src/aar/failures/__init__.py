"""Failure taxonomy and the never-silently-fail contract."""

from .registry import (  # noqa: F401
    AARError, CapabilityError, ConcurrencyError, Degradation,
    DegradationLedger, FailureKind, FailureMode, FailureRegistry, PlanInfeasible,
    PolicyDenied, PrivacyViolation, QualityCheckFailed, ResourceExhausted,
    SchemaDriftError, Severity, SourceUnavailable, TypeMismatchError,
    UDFExecutionError, capture, process_ledger, register_modes,
)

__all__ = [
    "AARError", "CapabilityError", "ConcurrencyError", "Degradation",
    "DegradationLedger", "FailureKind", "FailureMode", "FailureRegistry",
    "PlanInfeasible", "PolicyDenied", "PrivacyViolation", "QualityCheckFailed",
    "ResourceExhausted", "SchemaDriftError", "Severity", "SourceUnavailable",
    "TypeMismatchError", "UDFExecutionError", "capture", "process_ledger",
    "register_modes",
]
