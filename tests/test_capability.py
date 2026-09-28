"""Capability registry: declared capability, availability, feasible sets."""

from __future__ import annotations

import pytest

from aar.capability import (Capability, CapabilityRegistry, Device, ENGINES,
                            EngineSpec, Tier, default_registry, probe_engine)
from aar.ir import Node, NodeType, ScanSpec


def _node(node_type: NodeType, **kw) -> Node:
    return Node(node_type, **kw)


class TestCatalogue:
    def test_catalogue_is_not_empty(self):
        assert len(ENGINES) >= 15

    def test_engine_ids_are_unique(self):
        ids = [e.id for e in ENGINES]
        assert len(ids) == len(set(ids))

    def test_every_engine_declares_at_least_one_operation(self):
        for spec in ENGINES:
            assert spec.ops, f"{spec.id} declares no operations"

    def test_every_operation_has_at_least_one_engine(self):
        registry = CapabilityRegistry()
        for nt in NodeType:
            assert registry.declared_engines_for(nt), f"no engine for {nt}"

    def test_specification_priority_order_is_preserved(self):
        """Tier order must follow the spec: pushdown first, compatibility last."""
        tiers = [s.tier for s in ENGINES if s.id in (
            "postgresql", "duckdb", "polars_gpu", "ray", "pandas", "python_worker")]
        assert tiers == sorted(tiers, key=int)

    def test_unknown_engine_raises_with_help(self):
        with pytest.raises(KeyError) as exc:
            CapabilityRegistry().spec("nope")
        assert "known:" in str(exc.value)

    def test_duplicate_engine_id_rejected(self):
        spec = ENGINES[0]
        with pytest.raises(ValueError, match="duplicate"):
            CapabilityRegistry([spec, spec])


class TestDeclaredCapability:
    def test_duckdb_supports_groupby(self):
        assert CapabilityRegistry().spec("duckdb").supports(NodeType.GROUPBY)

    def test_python_udf_is_not_a_gpu_capability(self):
        registry = CapabilityRegistry()
        # A hard capability fact, not a preference: arbitrary Python cannot
        # run on a GPU, so no GPU engine may claim PYTHON_UDF.
        for spec in registry.by_device(Device.GPU):
            assert not spec.supports(NodeType.PYTHON_UDF), spec.id

    def test_python_worker_only_does_python_udf(self):
        ops = CapabilityRegistry().spec("python_worker").ops
        assert ops == frozenset({NodeType.PYTHON_UDF})

    def test_excel_is_source_and_sink_only(self):
        ops = CapabilityRegistry().spec("excel").ops
        assert ops == frozenset({NodeType.SCAN_EXCEL, NodeType.WRITE})
        assert NodeType.JOIN not in ops

    def test_sql_engines_cannot_scan_parquet(self):
        for engine in ("postgresql", "mysql", "trino"):
            spec = CapabilityRegistry().spec(engine)
            assert not spec.supports(NodeType.SCAN_PARQUET), engine

    def test_mongodb_supports_aggregation_pipeline_ops(self):
        spec = CapabilityRegistry().spec("mongodb")
        for nt in (NodeType.FILTER, NodeType.PROJECT, NodeType.GROUPBY,
                   NodeType.SORT, NodeType.LIMIT):
            assert spec.supports(nt), nt

    def test_distributed_engines_are_marked_remote(self):
        for engine in ("ray", "dask", "spark_rapids"):
            assert CapabilityRegistry().spec(engine).remote, engine

    def test_arrow_cannot_write(self):
        assert not CapabilityRegistry().spec("arrow").supports(NodeType.WRITE)


class TestAvailabilityProbe:
    def test_probe_reports_every_engine(self):
        caps = CapabilityRegistry().probe()
        assert set(caps) == {s.id for s in ENGINES}

    def test_every_capability_has_a_reason(self):
        """An unavailable engine must always say why. No bare 'no'."""
        for cap in CapabilityRegistry().probe().values():
            assert cap.reason, f"{cap.engine} has no reason"

    def test_probe_is_memoised(self):
        r = CapabilityRegistry()
        assert r.probe() is r.probe()

    def test_gpu_engines_are_blocked_by_hardware_or_package(self):
        """A GPU engine is unavailable for a *stated* reason, never silently."""
        reasons = CapabilityRegistry().unavailable_reasons()
        caps = CapabilityRegistry().probe()
        for engine in ("polars_gpu", "cudf"):
            if engine in reasons:
                assert reasons[engine].strip(), engine
                # The reason names the actual blocker, whether that is the
                # package or the hardware.
                assert caps[engine].blocked_by in (
                    "missing_package", "no_gpu", "old_gpu")



    def test_missing_package_is_named(self):
        reg = CapabilityRegistry()
        for cap in reg.probe().values():
            if cap.blocked_by == "missing_package":
                assert "not installed" in cap.reason
                assert cap.version is None

    def test_every_availability_claim_is_backed_by_a_real_probe(self):
        """An engine must not claim availability without a way to know it.

        `postgresql` once carried `probe=None`, so it reported "available"
        while its driver (`psycopg`) was not installed. A plan that trusts
        such a claim fails at run time, on a machine that said yes.
        """
        reg = CapabilityRegistry()
        caps = reg.probe()
        for spec in reg.engines:
            if spec.probe is None and not spec.intrinsic:
                pytest.fail(
                    f"{spec.id} has neither a probe nor intrinsic=True, so its "
                    f"availability is unknowable")
            if caps[spec.id].available and spec.probe is None:
                assert spec.intrinsic, (
                    f"{spec.id} reports available with no probe module")

    def test_intrinsic_engines_are_always_available(self):
        """An intrinsic engine is the running interpreter; it cannot vanish."""
        reg = CapabilityRegistry()
        for spec in reg.engines:
            if spec.intrinsic:
                assert reg.is_available(spec.id), spec.id

    def test_remote_engines_have_a_driver_probe(self):
        """Anything remote needs a driver, and therefore a probe."""
        reg = CapabilityRegistry()
        for spec in reg.engines:
            if spec.remote:
                assert spec.probe, (
                    f"{spec.id} is remote but has no driver probe, so its "
                    f"availability is unknowable")

    def test_version_is_not_printed_twice(self):
        """Regression: the version appeared in both reason and suffix."""
        for cap in CapabilityRegistry().probe().values():
            text = cap.render()
            if cap.version:
                assert text.count(cap.version) == 1, text


    def test_default_registry_is_a_singleton(self):
        assert default_registry() is default_registry()



class TestFeasibleSet:
    def test_feasible_is_ordered_by_tier(self):
        reg = CapabilityRegistry()
        feas = reg.feasible(_node(NodeType.GROUPBY), available_only=False)
        tiers = [reg.spec(e).tier for e in feas]
        assert tiers == sorted(tiers, key=int)

    def test_node_declaration_narrows_the_set(self):
        reg = CapabilityRegistry()
        node = _node(NodeType.GROUPBY,
                     supported_engines=frozenset({"duckdb"}))
        assert "duckdb" in reg.feasible(node, available_only=False)
        assert "polars_cpu" not in reg.feasible(node, available_only=False)

    def test_absent_engine_is_excluded_when_availability_required(self):
        reg = CapabilityRegistry()
        reg.probe()
        node = _node(NodeType.GROUPBY,
                     supported_engines=frozenset({"duckdb", "spark_rapids"}))
        feas = reg.feasible(node, available_only=True)
        if not reg.is_available("spark_rapids"):
            assert "spark_rapids" not in feas
            assert "duckdb" in feas

    def test_empty_feasible_set_is_possible_and_reportable(self):
        """No feasible engine is a legitimate plan outcome, and it is reported."""
        reg = CapabilityRegistry()
        node = _node(NodeType.GROUPBY,
                     supported_engines=frozenset({"nonexistent"}))
        feas, rejected = reg.feasible_with_reasons(node)
        assert feas == ()
        # Every engine is rejected, and the unknown one is called out.
        assert rejected
        assert "not declared for this operation" in rejected["duckdb"]


    def test_rejections_are_all_explained(self):
        """Every rejected engine must carry a reason - the explain contract."""
        reg = CapabilityRegistry()
        for nt in (NodeType.GROUPBY, NodeType.WRITE, NodeType.PYTHON_UDF,
                   NodeType.SCAN_EXCEL):
            _, rejected = reg.feasible_with_reasons(_node(nt))
            for engine, reason in rejected.items():
                assert reason, f"{engine} rejected for {nt} with no reason"

    def test_python_udf_rejection_names_the_gpu_reason(self):
        reg = CapabilityRegistry()
        node = _node(NodeType.PYTHON_UDF)
        _, rejected = reg.feasible_with_reasons(node)
        gpus = [e for e in rejected if reg.spec(e).device is Device.GPU]
        for engine in gpus:
            assert "unsupported on GPU" in rejected[engine]

    def test_feasible_set_is_deterministic(self):
        reg = CapabilityRegistry()
        node = _node(NodeType.JOIN)
        assert reg.feasible(node) == reg.feasible(node)

    def test_available_only_false_ignores_install_state(self):
        reg = CapabilityRegistry()
        node = _node(NodeType.SCAN_CSV,
                     supported_engines=frozenset({"spark_rapids"}))
        # Declared, though almost certainly not installed.
        assert "spark_rapids" in reg.feasible(node, available_only=False)


class TestMemoryFitness:
    def _profile(self, ram: int, vram: int):
        class _P:
            memory_budget_bytes = ram
            vram_budget_bytes = vram
        return _P()

    def test_gpu_engine_is_bounded_by_vram_not_ram(self):
        """A 50 GB dataset does not fit an 8 GB card, however much RAM exists."""
        reg = CapabilityRegistry()
        profile = self._profile(ram=64_000_000_000, vram=8_000_000_000)
        assert not reg.fits_memory("polars_gpu", 50_000_000_000, profile)
        assert reg.fits_memory("duckdb", 50_000_000_000, profile)

    def test_cpu_engine_uses_ram_budget(self):
        reg = CapabilityRegistry()
        profile = self._profile(ram=1_000_000_000, vram=0)
        assert not reg.fits_memory("duckdb", 4_000_000_000, profile)
        assert reg.fits_memory("duckdb", 500_000_000, profile)

    def test_zero_vram_rejects_every_gpu_dataset(self):
        reg = CapabilityRegistry()
        profile = self._profile(ram=1 << 40, vram=0)
        assert not reg.fits_memory("cudf", 1, profile)


class TestLedgerIntegration:
    def test_absent_engine_is_logged_as_a_degradation(self):
        """A plan shaped by a missing package must say so, not look considered."""
        from aar.failures import DegradationLedger, FailureKind

        reg = CapabilityRegistry()
        reg.probe()
        absent = [e for e, c in reg.probe().items() if not c.available]
        assert absent, "expected at least one absent engine in any environment"

        ledger = DegradationLedger()
        node = _node(NodeType.GROUPBY,
                     supported_engines=frozenset(absent[:1]))
        reg.record_absent(node, ledger)
        assert len(ledger) == 1
        d = ledger.entries[0]
        assert d.kind is FailureKind.ENGINE_ABSENT
        assert d.exception is None
        assert "excluded" in d.detail

    def test_present_engines_are_not_logged(self):
        from aar.failures import DegradationLedger

        reg = CapabilityRegistry()
        caps = reg.probe()
        eligible = set(reg.declared_engines_for(NodeType.GROUPBY))
        present = {e for e, c in caps.items()
                   if c.available and e in eligible}
        assert present, "expected an available engine for GROUPBY"

        ledger = DegradationLedger()
        reg.record_absent(
            _node(NodeType.GROUPBY, supported_engines=frozenset(present)),
            ledger)
        assert len(ledger) == 0


    def test_undeclared_node_logs_nothing(self):
        from aar.failures import DegradationLedger

        ledger = DegradationLedger()
        CapabilityRegistry().record_absent(_node(NodeType.GROUPBY), ledger)
        assert len(ledger) == 0


class TestRendering:
    def test_render_lists_every_engine_and_its_tier(self):
        text = CapabilityRegistry().render()
        assert "ENGINE CAPABILITY" in text
        for spec in ENGINES:
            assert spec.id in text
        assert "tier 1" in text and "tier 6" in text

    def test_render_marks_availability_clearly(self):
        text = CapabilityRegistry().render()
        assert "[yes]" in text or "[no ]" in text

    def test_probe_engine_on_a_fake_spec(self):
        spec = EngineSpec(
            id="fake", label="Fake", device=Device.CPU,
            tier=Tier.COMPATIBILITY, probe="definitely_not_installed_xyz",
        )
        cap = probe_engine(spec)
        assert not cap.available
        assert "not installed" in cap.reason
        assert cap.blocked_by == "missing_package"


