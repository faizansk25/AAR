"""The data profiler: does it measure better than a declared guess?

The acceptance criterion is not "it returns numbers" - it is that the
numbers are *right*. Each test compares a profile against ground truth it can
compute independently, because a profiler that reports a plausible row count
is indistinguishable from one that reports the declared one.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from aar.interchange import Table  # noqa: E402
from aar.stats import DataProfiler  # noqa: E402


def _table(**columns) -> Table:
    return Table(pa.table(columns))


class TestInMemoryProfiling:
    """An in-memory table knows its own size, so nothing is extrapolated."""

    def test_row_count_and_size_are_exact(self):
        t = _table(k=[1, 2, 3, 4, 5], v=["a", "b", "c", "d", "e"])
        p = DataProfiler().profile_table(t, name="t")
        assert p.rows == 5
        assert p.nbytes == t.nbytes
        assert p.exact_rows is True
        assert p.source == "measured"

    def test_bytes_per_row_is_derived_not_assumed(self):
        """A table of long strings is not the same size as one of shorts."""
        short = DataProfiler().profile_table(_table(v=["a", "b", "c"]))
        long = DataProfiler().profile_table(
            _table(v=["x" * 200, "y" * 200, "z" * 200]))
        assert long.nbytes > short.nbytes * 10
        assert long.bytes_per_row > short.bytes_per_row

    def test_an_empty_table_does_not_divide_by_zero(self):
        p = DataProfiler().profile_table(_table(k=pa.array([], type=pa.int64())))
        assert p.rows == 0
        assert p.bytes_per_row == 0.0


class TestColumnStatistics:
    def test_distinct_count_is_exact_for_a_small_column(self):
        t = _table(g=["a", "a", "b", "b", "c"])
        column = DataProfiler().profile_table(t).column("g")
        assert column.distinct == 3
        assert column.distinct_estimate == 3

    def test_null_fraction_is_measured(self):
        t = _table(v=[1, None, 3, None])
        column = DataProfiler().profile_table(t).column("v")
        assert column.null_fraction == pytest.approx(0.5)

    def test_an_all_null_column_is_reported_as_unsupported(self):
        """Distinct 0 and null 100% is a real answer, not a missing one."""
        t = _table(v=[None, None, None])
        column = DataProfiler().profile_table(t).column("v")
        assert column.null_fraction == 1.0
        assert column.unsupported is True

    def test_average_width_reflects_the_data(self):
        t = _table(s=["ab", "abcd"])
        assert DataProfiler().profile_table(t).column("s").avg_chars == \
            pytest.approx(3.0)

    def test_min_and_max_come_from_the_data(self):
        t = _table(v=[5, 1, 9])
        column = DataProfiler().profile_table(t).column("v")
        assert column.min_value == 1
        assert column.max_value == 9


class TestSelectivity:
    """The number the cost model actually needs: how many rows survive."""

    def test_equality_on_a_ten_value_column_keeps_a_tenth(self):
        t = _table(v=[i % 10 for i in range(1000)])
        p = DataProfiler().profile_table(t)
        assert p.column("v").selectivity_of_equality() == pytest.approx(0.1)

    def test_selectivity_scales_the_row_estimate(self):
        t = _table(v=[i % 10 for i in range(1000)])
        p = DataProfiler().profile_table(t)
        selectivity = p.column("v").selectivity_of_equality()
        assert p.estimate_after_filter(selectivity) == pytest.approx(100,
                                                                    abs=5)

    def test_a_unique_column_has_full_selectivity(self):
        t = _table(v=list(range(500)))
        p = DataProfiler().profile_table(t)
        assert p.column("v").selectivity_of_equality() == pytest.approx(
            1.0 / 500, rel=1e-6)


class TestSamplingIsBounded:
    """Profiling must not scan the whole file in order to plan the query."""

    def test_it_reads_at_most_the_sample_size(self, tmp_path):
        import csv

        path = tmp_path / "big.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["v"])
            for i in range(20_000):
                writer.writerow([i])

        p = DataProfiler(sample_rows=1_000).profile_csv(str(path))
        assert p.exact_rows is False
        assert p.source == "sampled"
        # Within 10% of the true 20,000 despite reading 5% of it.
        assert p.rows == pytest.approx(20_000, rel=0.10)

    def test_a_file_smaller_than_the_sample_is_read_whole(self, tmp_path):
        import csv

        path = tmp_path / "small.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["v"])
            for i in range(10):
                writer.writerow([i])

        p = DataProfiler(sample_rows=50_000).profile_csv(str(path))
        assert p.rows == pytest.approx(10, rel=0.20)


class TestParquetMetadataIsExact:
    """Parquet carries statistics in the footer: read them, don't scan."""

    def test_row_count_is_exact_and_cheap(self, tmp_path):
        import pyarrow.parquet as pq

        path = tmp_path / "t.parquet"
        pq.write_table(pa.table({"v": list(range(5_000))}), path)
        p = DataProfiler().profile_parquet(str(path))
        assert p.rows == 5_000
        assert p.exact_rows is True
        assert p.source == "parquet-metadata"

    def test_null_counts_come_from_the_footer(self, tmp_path):
        import pyarrow.parquet as pq

        path = tmp_path / "n.parquet"
        pq.write_table(pa.table({"v": pa.array([1, None, 3, None])}), path)
        column = DataProfiler().profile_parquet(str(path)).column("v")
        assert column is not None
        assert column.null_fraction == pytest.approx(0.5)


class TestProvenanceIsNeverHidden:
    """A guess must never be able to pose as a measurement."""

    def test_every_profile_names_its_source(self, tmp_path):
        import pyarrow.parquet as pq

        csv_path = tmp_path / "a.csv"
        csv_path.write_text("v\n1\n2\n3\n", encoding="utf-8")
        pq_path = tmp_path / "a.parquet"
        pq.write_table(pa.table({"v": [1, 2, 3]}), pq_path)

        for profile in (DataProfiler().profile_csv(str(csv_path)),
                        DataProfiler().profile_parquet(str(pq_path))):
            assert profile.source
            assert profile.source != "default"

    def test_an_unreadable_source_returns_none_rather_than_zero_rows(
            self, tmp_path):
        """Zero rows would plan a trivially cheap pipeline. Refuse instead."""
        from aar.ir import Node, NodeType, ScanSpec

        node = Node(NodeType.SCAN_CSV, scan=ScanSpec(
            kind="csv", path=str(tmp_path / "missing.csv")))
        assert DataProfiler().profile_node(node) is None

    def test_a_node_without_a_scan_is_not_profiled(self):
        from aar.ir import Node, NodeType

        assert DataProfiler().profile_node(Node(NodeType.GROUPBY)) is None


class TestCardinalityIsBounded:
    """Distinct counting must not become the memory problem it avoids."""

    def test_a_high_cardinality_column_is_estimated_not_stored(self):
        from aar.stats import _DistinctEstimator

        estimator = _DistinctEstimator(100_000)
        for i in range(50_000):
            estimator.add(i)
        # The exact set stops growing at the limit even though the sketch
        # keeps answering.
        assert len(estimator._exact) <= _DistinctEstimator.EXACT_LIMIT
        assert 40_000 < estimator.estimate() < 60_000

    def test_a_low_cardinality_column_stays_exact(self):
        from aar.stats import _DistinctEstimator

        estimator = _DistinctEstimator(10)
        for i in range(1_000):
            estimator.add(i % 10)
        assert estimator.exact == 10
        assert estimator.estimate() == pytest.approx(10)


class TestRenderIsReadable:
    def test_a_profile_renders_its_own_provenance(self):
        p = DataProfiler().profile_table(_table(v=[1, 2, 3]), name="sales")
        text = p.render()
        assert "sales" in text
        assert "measured" in text
        assert "rows" in text

    def test_a_column_renders_its_statistics(self):
        p = DataProfiler().profile_table(_table(g=["a", "b"]))
        assert "distinct" in p.column("g").render()


class TestThePlannerPrefersAMeasuredSize:
    """The profiler only matters if the planner actually consumes it."""

    def test_a_measured_size_beats_a_declared_one(self, tmp_path):
        from aar.ir import Node, NodeType
        from aar.planner import estimate_bytes
        from aar.stats import TableProfile

        node = Node(NodeType.SCAN_PARQUET)
        node.estimated_bytes = 999_999_999
        assert estimate_bytes(node) == 999_999_999

        # Attaching a measured profile overrides the declaration.
        node.aar_profile = TableProfile(rows=100, nbytes=4096,
                                        source="measured")
        assert estimate_bytes(node) == 4096

    def test_a_zero_byte_profile_does_not_override_a_real_estimate(self):
        """A failed measurement must not make a plan look free."""
        from aar.ir import Node, NodeType
        from aar.planner import estimate_bytes
        from aar.stats import TableProfile

        node = Node(NodeType.SCAN_PARQUET)
        node.estimated_bytes = 5_000
        node.aar_profile = TableProfile(rows=0, nbytes=0, source="sampled")
        assert estimate_bytes(node) == 5_000


