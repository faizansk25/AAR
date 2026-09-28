"""Failure registry and the never-silently-fail contract."""

from __future__ import annotations

import pytest

from aar.failures import (AARError, DegradationLedger, FailureKind,
                          FailureRegistry, PlanInfeasible, PrivacyViolation,
                          SchemaDriftError, Severity, SourceUnavailable,
                          TypeMismatchError, register_modes)


class TestFailureMatrix:
    def test_every_kind_has_an_entry(self):
        modes = register_modes()
        for kind in FailureKind:
            assert kind in modes, f"{kind} missing from the failure matrix"

    def test_matrix_covers_the_specification_modes(self):
        names = {m.kind.value for m in FailureRegistry.all()}
        # The specification's numbered list, plus the two AAR-specific ones.
        for required in (
            "GPU_UNAVAILABLE", "GPU_OOM_VRAM", "GPU_OP_UNSUPPORTED",
            "SCHEMA_DRIFT", "TYPE_MISMATCH", "NETWORK_PARTITION",
            "LICENSE_EXHAUSTED", "SOURCE_UNAVAILABLE", "CARDINALITY_ERROR",
            "EXCEL_FORMAT_CHANGE", "EXCEL_FORMULA_ERROR", "UDF_FAILURE",
            "ARROW_IPC_FAILURE", "WORKER_FAILURE", "MEMORY_SPILL",
            "CACHE_INVALIDATED", "CONCURRENT_ACCESS", "QUALITY_FAILURE",
            "PRIVACY_VIOLATION", "GPU_OOM_UVM",
        ):
            assert required in names, f"{required} missing"

    def test_every_mode_documents_detection_and_fallback(self):
        for mode in FailureRegistry.all():
            assert mode.detect and mode.handle and mode.fallback and mode.log

    def test_matrix_renders_as_table(self):
        text = FailureRegistry.render_matrix()
        assert "SEVERITY" in text
        assert "SCHEMA_DRIFT" in text

    def test_registration_is_idempotent(self):
        first = len(register_modes())
        second = len(register_modes())
        assert first == second


class TestErrorHierarchy:
    def test_errors_carry_failure_mode(self):
        assert SchemaDriftError("x").failure_mode is FailureKind.SCHEMA_DRIFT
        assert PrivacyViolation("x").failure_mode is FailureKind.PRIVACY_VIOLATION
        assert SourceUnavailable("x").failure_mode is FailureKind.SOURCE_UNAVAILABLE

    def test_blocking_errors_marked_blocking(self):
        assert SchemaDriftError("x").severity is Severity.BLOCKING
        assert PrivacyViolation("x").severity is Severity.BLOCKING

    def test_infeasible_plan_is_reported(self):
        assert PlanInfeasible("no engine").failure_mode is FailureKind.NO_FEASIBLE_PLAN

    def test_string_includes_code_and_context(self):
        err = AARError("boom", node="op1")
        assert "boom" in str(err)
        assert "node=op1" in str(err)

    def test_to_dict_is_serialisable(self):
        d = TypeMismatchError("bad", column="x").to_dict()
        assert d["failure_mode"] == "TYPE_MISMATCH"
        assert d["context"]["column"] == "x"


class TestDegradationLedger:
    def test_clean_ledger_passes_assertion(self):
        DegradationLedger().assert_clean()

    def test_degradation_is_recorded_with_detail(self):
        ledger = DegradationLedger()
        d = ledger.record(FailureKind.GPU_UNAVAILABLE, "join",
                          "no CUDA device", from_engine="polars_gpu",
                          to_engine="polars_cpu")
        assert len(ledger) == 1
        assert d.from_engine == "polars_gpu"
        assert "polars_cpu" in d.render()

    def test_blocking_degradation_fails_the_run(self):
        ledger = DegradationLedger()
        ledger.record(FailureKind.SCHEMA_DRIFT, "scan", "column removed")
        with pytest.raises(AARError, match="unresolved blocking"):
            ledger.assert_clean()

    def test_info_degradation_does_not_block(self):
        ledger = DegradationLedger()
        ledger.record(FailureKind.ENGINE_ABSENT, "plan", "duckdb not installed")
        ledger.assert_clean()  # must not raise

    def test_counts_by_kind(self):
        ledger = DegradationLedger()
        ledger.record(FailureKind.GPU_OOM_VRAM, "a", "x")
        ledger.record(FailureKind.GPU_OOM_VRAM, "b", "x")
        ledger.record(FailureKind.UDF_FAILURE, "c", "x")
        assert ledger.counts() == {"GPU_OOM_VRAM": 2, "UDF_FAILURE": 1}

    def test_sink_receives_every_event(self):
        seen = []
        ledger = DegradationLedger(seen.append)
        ledger.record(FailureKind.GPU_UNAVAILABLE, "a", "x")
        assert len(seen) == 1

    def test_broken_sink_does_not_mask_the_event(self):
        def bad(_d):
            raise RuntimeError("sink exploded")

        ledger = DegradationLedger(bad)
        ledger.record(FailureKind.GPU_UNAVAILABLE, "a", "x")
        assert len(ledger) == 1  # the degradation still stands

    def test_empty_ledger_renders_cleanly(self):
        assert "No degradations" in DegradationLedger().render()
