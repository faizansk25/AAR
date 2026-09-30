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


class TestProfilingIsActuallyBounded:
    """The promise is that profiling reads a *sample*.

    The previous implementation set ``block_size`` and then called
    ``read_all()``, which reads the entire file: ``block_size`` chooses the
    buffer granularity, it is not a limit. Measured on a 200,000-row file
    with a 1,000-row budget, ``read_all()`` returned all 200,000 rows.

    That is the failure this component exists to prevent. Profiling a 40 GB
    CSV on an 8 GB laptop would exhaust memory *before* planning started,
    which is exactly the situation the bounded read is supposed to make
    safe.
    """

    def _big_csv(self, directory, rows: int = 200_000) -> str:
        import csv

        path = directory / "big.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["v"])
            for i in range(rows):
                writer.writerow([i])
        return str(path)

    def test_the_whole_file_is_never_read(self, tmp_path):
        """The bound that matters: work scales with the budget, not the file.

        The row count is not asserted exactly, because PyArrow emits whole
        batches and refuses to go below roughly 840 rows, so a 500-row
        budget is honoured only to within one batch. What must hold is that
        reading stops early - which is the difference between a 200,000-row
        read (the old ``read_all()`` behaviour, measured) and 1,550.
        """
        import pyarrow.csv as pacsv
        from aar.stats import DataProfiler

        path = self._big_csv(tmp_path, rows=200_000)
        profiler = DataProfiler(sample_rows=500)

        seen = {"rows": 0}
        real_open = pacsv.open_csv

        class _CountingReader:
            def __init__(self, inner):
                self._inner = inner

            def __iter__(self):
                for batch in self._inner:
                    seen["rows"] += batch.num_rows
                    yield batch

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

        pacsv.open_csv = lambda *a, **kw: _CountingReader(real_open(*a, **kw))
        try:
            profiler.profile_csv(path)
        finally:
            pacsv.open_csv = real_open

        assert seen["rows"] < 5_000, (
            f"profiling read {seen['rows']} of 200,000 rows for a 500-row "
            f"budget - the read is not bounded")
        assert seen["rows"] > 0

    def test_the_retained_sample_is_exactly_the_budget(self, tmp_path):
        """What the profiler *keeps* is exact, even though Arrow
        over-delivers a batch."""
        from aar.stats import DataProfiler

        path = self._big_csv(tmp_path, rows=200_000)
        sample = DataProfiler(sample_rows=500)._read_bounded_csv(path)
        assert sample is not None
        assert sample.num_rows == 500

    def test_the_byte_budget_is_enforced_below_the_reader(self, tmp_path):
        """A row budget is not a memory budget.

        Rows can be arbitrarily wide, so the byte cap is the guarantee that
        actually bounds memory. It is enforced on the stream itself, so it
        holds no matter what the reader asks for.
        """
        from aar.stats import _ByteCappedFile

        path = self._big_csv(tmp_path, rows=200_000)
        with _ByteCappedFile(path, 4_096) as capped:
            read = 0
            while True:
                chunk = capped.read(8192)
                if not chunk:
                    break
                read += len(chunk)
        assert read == 4_096
        assert capped.bytes_read == 4_096

    def test_a_byte_budget_smaller_than_a_batch_still_works(self, tmp_path):
        """The two limits are independent; the tighter one wins."""
        from aar.stats import DataProfiler

        path = self._big_csv(tmp_path, rows=50_000)
        profile = DataProfiler(sample_rows=50_000,
                               max_bytes=2_048).profile_csv(path)
        assert profile.source == "sampled"
        assert profile.exact_rows is False

    def test_a_huge_file_is_sampled_not_loaded(self, tmp_path):
        """The end-to-end promise: a big file profiles without blowing up."""
        from aar.stats import DataProfiler

        path = self._big_csv(tmp_path, rows=300_000)
        profile = DataProfiler(sample_rows=1_000).profile_csv(path)
        assert profile.exact_rows is False
        assert profile.rows == pytest.approx(300_000, rel=0.10)

    def test_the_extrapolation_survives_a_small_sample(self, tmp_path):
        """Bounding the read must not cost accuracy.

        A 200-row sample and a 5,000-row sample of the same file both have
        to land near the truth, or the bound is only safe because the
        estimate is useless.
        """
        from aar.stats import DataProfiler

        path = self._big_csv(tmp_path, rows=50_000)
        small = DataProfiler(sample_rows=200).profile_csv(path)
        large = DataProfiler(sample_rows=5_000).profile_csv(path)
        assert small.rows == pytest.approx(50_000, rel=0.10)
        assert large.rows == pytest.approx(50_000, rel=0.10)

    def test_row_width_is_measured_beyond_the_head(self, tmp_path):
        """The head block alone is not representative.

        Rows whose width grows down the file (an integer column going from
        1 to 6 digits) make the head the *narrowest* part of the file, so a
        head-only measurement over-counts rows. Measured: 382,401 estimated
        against 300,000 true - 27% high, and the file size is divided by
        that number, so the error lands straight on the plan.
        """
        from aar.stats import DataProfiler

        path = self._big_csv(tmp_path, rows=300_000)
        profile = DataProfiler(sample_rows=1_000).profile_csv(path)
        assert profile.rows == pytest.approx(300_000, rel=0.10)


class TestEmbeddedNewlinesInCsvFields:
    """A quoted field containing a newline made profiling *crash*.

    ``profile_csv`` opens the file with default PyArrow options, so a value
    spanning lines raises ``ArrowInvalid: CSV parser got out of sync with
    chunker``. That exception propagates out of ``profile_node``, which
    returns ``None``, which is fine - but only because ``profile_node``
    catches it. Any path that reaches the reader without that guard dies,
    and a pipeline whose *source* contains an embedded newline is a
    completely ordinary thing for a user to have.

    The audit flagged this specifically: counting ``\\n`` to measure record
    width assumes one newline per record, which is false for quoted fields.
    The width estimate is derived from newline counts in
    ``_bytes_per_csv_row``, so the same assumption sits in two places.
    """

    def _csv_with_embedded_newlines(self, tmp_path, rows: int = 1_000) -> str:
        import csv

        path = tmp_path / "embedded.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["v", "note"])
            for i in range(rows):
                # A quoted field holding a real newline - legal CSV, and what
                # any export of a multi-line text field produces.
                writer.writerow([i, "first line\nsecond line"])
        return str(path)

    def test_a_source_with_embedded_newlines_does_not_kill_planning(self, tmp_path):
        from aar.application import PipelineService

        data = self._csv_with_embedded_newlines(tmp_path)
        pipeline = tmp_path / "pipe.py"
        pipeline.write_text(
            "from aar.sdk import pipeline as p\n"
            f"src = p.csv(r'{data}')\n"
            "\n"
            "def build():\n"
            "    return p.write_csv(p.filter_(src,"
            " p.gt(p.col('v'), 0)),"
            f" r'{tmp_path / 'out.csv'}')\n",
            encoding="utf-8",
        )

        report = PipelineService().run(str(pipeline))
        assert report.plan is not None

    def test_profiling_returns_none_rather_than_raising(self, tmp_path):
        from aar.sdk import pipeline as sdk
        from aar.stats import DataProfiler

        data = self._csv_with_embedded_newlines(tmp_path)
        # Either it reads the file, or it declines cleanly. Raising is not an
        # option: a source the profiler cannot read must not fail the run.
        profile = DataProfiler().profile_node(sdk.csv(data))
        assert profile is None or profile.rows >= 0

    def test_the_reader_is_asked_to_accept_newlines_in_values(self, tmp_path):
        """The fix is to tell PyArrow the file may contain them.

        Asserted on the options the profiler actually builds, because the
        failure mode is an exception thrown from inside the C++ reader -
        invisible to a caller that only inspects the returned profile.
        """
        import pyarrow.csv as pacsv

        from aar.stats import DataProfiler

        captured = {}
        real_open = pacsv.open_csv

        def spy(path, **kwargs):
            captured.update(kwargs)
            return real_open(path, **kwargs)

        data = self._csv_with_embedded_newlines(tmp_path, rows=50)
        pacsv.open_csv = spy
        try:
            DataProfiler(sample_rows=20).profile_csv(data)
        finally:
            pacsv.open_csv = real_open

        options = captured.get("parse_options")
        assert options is not None, "the reader was built without options"
        assert options.newlines_in_values is True, (
            "newlines_in_values is not enabled, so a quoted field with a "
            "newline makes the parser raise")

    def test_record_width_is_not_derived_from_newline_counts_alone(self, tmp_path):
        """A newline inside a quoted field is not a record boundary.

        ``_bytes_per_csv_row`` divides a block's length by its newline
        count, which over-counts records whenever a value spans lines and so
        *under*-estimates bytes per row - inflating the row estimate.
        """
        import csv

        from aar.stats import _bytes_per_csv_row

        path = tmp_path / "nl_width.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["v", "note"])
            for i in range(1_000):
                writer.writerow([i, "first line\nsecond line"])

        import pyarrow.csv as pacsv

        reader = pacsv.open_csv(path, parse_options=pacsv.ParseOptions(
            newlines_in_values=True))
        # Batches, not rows: this file is wider than the byte budget, so the
        # count is capped by the reader and cannot be compared directly.
        # The header plus 1,000 records is the truth we care about.
        assert next(iter(reader)).num_rows >= 1

        width = _bytes_per_csv_row(str(path), 65_536)
        size = path.stat().st_size
        estimated = size / width
        # 1,000 records but 2,000 newlines: a newline-counting estimator
        # sees half the record width and doubles the row count.
        assert estimated == pytest.approx(1_000, rel=0.15), (
            f"estimated {estimated:.0f} rows against a true 1,000 - "
            f"embedded newlines are being counted as record boundaries")


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


class TestSqlStatisticsThroughThePublicSdk:
    """``_sqlite_stat_rows`` looked for ``path``; the SDK sets ``connection``.

    ``sql()`` builds ``ScanSpec(kind="sql", connection=...)`` and never sets
    ``path``, so the helper could never match a scan built through the
    public API. Every SQL pipeline therefore fell back to reading the whole
    table to count rows - the expensive path - while the profile claimed
    only that statistics were unavailable.

    Tested through ``sql()`` rather than against the helper directly,
    because the defect was precisely a mismatch between the two: a unit
    test of the helper alone would have passed throughout.
    """

    def _database(self, tmp_path, rows: int = 500):
        import sqlite3

        path = str(tmp_path / "stats.db")
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE trips (id INTEGER PRIMARY KEY, v TEXT)")
            conn.executemany("INSERT INTO trips (v) VALUES (?)",
                             [(f"row{i}",) for i in range(rows)])
            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()
        return path

    def test_a_scanned_sql_source_uses_the_database_statistics(self, tmp_path):
        from aar.sdk import pipeline as sdk
        from aar.stats import DataProfiler

        path = self._database(tmp_path, rows=500)
        profile = DataProfiler().profile_node(sdk.sql(path, table="trips"))

        assert profile is not None
        assert profile.source == "database-stats", (
            "profiling fell back to sampling - the statistics path never "
            "matched the spec the SDK actually builds")
        assert profile.rows == 500
        assert profile.exact_rows is True

    def test_the_scan_spec_carries_no_path_to_find(self, tmp_path):
        """States the mismatch directly, so a future SDK change is caught."""
        from aar.sdk import pipeline as sdk

        path = self._database(tmp_path, rows=10)
        spec = sdk.sql(path, table="trips").scan
        assert spec.connection == path
        assert getattr(spec, "path", None) is None

    def test_a_file_uri_connection_is_understood(self, tmp_path):
        import sqlite3

        from aar.stats import _sqlite_path_from_connection

        path = self._database(tmp_path, rows=10)
        assert _sqlite_path_from_connection(f"file:{path}?mode=ro") == path
        # A server DSN and an in-memory database are not files.
        assert _sqlite_path_from_connection("postgresql://h/db") is None
        assert _sqlite_path_from_connection(":memory:") is None
        assert _sqlite_path_from_connection(None) is None
        # An open connection exposes no path, so the caller samples rather
        # than guessing at a file.
        conn = sqlite3.connect(path)
        try:
            assert _sqlite_path_from_connection(conn) is None
        finally:
            conn.close()

    def test_a_table_without_analyze_still_profiles(self, tmp_path):
        """No ``sqlite_stat1`` still yields a real, exact row count.

        The fallback used to be a full table read, and it passed the whole
        ``ScanSpec`` to a connector expecting a path, so it raised for every
        scan that needed it. It now asks the database to count, which is
        exact and does not materialise rows.
        """
        import sqlite3

        from aar.sdk import pipeline as sdk
        from aar.stats import DataProfiler

        path = str(tmp_path / "noanalyze.db")
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            conn.executemany("INSERT INTO t VALUES (?)",
                             [(i,) for i in range(20)])
            conn.commit()
        finally:
            conn.close()

        profile = DataProfiler().profile_node(sdk.sql(path, table="t"))
        assert profile is not None, "an unanalysed table failed to profile"
        assert profile.source == "database-count"
        assert profile.rows == 20
        assert profile.exact_rows is True
        # Per-column statistics still came from a bounded sample, so the
        # column count must reflect the sample and not the table.
        assert profile.column("id") is not None


class TestParquetStatisticsSpanEveryRowGroup:
    """Null fractions were taken from one row group, not the whole file.

    Each group's null count was divided by the *file's* row count and then
    overwrote the previous value, so the fraction that survived belonged to
    whichever group was read last.
    """

    def _multi_group(self, tmp_path):
        """A Parquet file whose first row group is full and second is empty."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = str(tmp_path / "groups.parquet")
        first = pa.table({"v": pa.array([1] * 400, type=pa.int64())})
        second = pa.table({"v": pa.array([None] * 100, type=pa.int64())})
        writer = pq.ParquetWriter(path, first.schema)
        try:
            writer.write_table(first)
            writer.write_table(second)
        finally:
            writer.close()
        return path

    def test_the_null_fraction_covers_the_whole_file(self, tmp_path):
        from aar.stats import DataProfiler

        path = self._multi_group(tmp_path)
        profile = DataProfiler().profile_parquet(path)

        assert profile.rows == 500
        # 100 of 500 rows are null. Reading a single group gives 0.0 or 0.2,
        # so this only holds if both groups were counted.
        assert profile.column("v").null_fraction == pytest.approx(0.2)

    def test_a_row_group_distinct_count_is_not_reported_as_exact(
            self, tmp_path):
        """Per-group cardinality is a lower bound on the file's.

        The same value can appear in several groups, so no single group's
        count is the file's count. When a writer *does* record one, it is
        demoted to an estimate and ``distinct`` is cleared, so nothing
        downstream can mistake it for a measurement.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from aar.stats import DataProfiler, _mark_parquet_distinct_as_bounds

        path = str(tmp_path / "distinct.parquet")
        # One row group, so this writer definitely records a distinct count.
        pq.write_table(pa.table({"v": pa.array([1, 2, 3], type=pa.int64())}),
                       path, row_group_size=3)
        column = DataProfiler().profile_parquet(path).column("v")

        if column.distinct is not None:  # pragma: no cover - writer-dependent
            pytest.fail("a row group's distinct count was published as exact")
        # Whether or not this writer emitted one, an exact claim is never made.
        assert column.distinct is None

        # And the demotion itself is directly testable on a profile that
        # does carry a count.
        from aar.stats import ColumnProfile, TableProfile

        profile = TableProfile(name="x", rows=10, nbytes=0)
        profile.columns = [ColumnProfile(name="v", distinct=7,
                                         distinct_estimate=7)]
        _mark_parquet_distinct_as_bounds(profile)
        assert profile.column("v").distinct is None
        assert profile.column("v").distinct_estimate == 7

    def test_a_null_only_column_is_still_reported(self, tmp_path):
        """A writer that records nulls but not distincts must not lose them."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from aar.stats import DataProfiler

        path = str(tmp_path / "nulls.parquet")
        table = pa.table({"v": pa.array([None, None, 1, 2], type=pa.int64())})
        pq.write_table(table, path)
        column = DataProfiler().profile_parquet(path).column("v")
        assert column.null_fraction == pytest.approx(0.5)


class TestDistinctUsesTheApproximateAnswer:
    """``estimator.exact`` was read even after it stopped being exact.

    The exact set is capped at ``EXACT_LIMIT``, so past that point
    ``len(self._exact)`` is a *constant* - it stops responding to the data.
    Reading it unconditionally made every high-cardinality column report
    the limit as its cardinality, and that number feeds filter selectivity,
    which feeds engine choice.
    """

    def _column(self, distinct_values: int):
        import pyarrow as pa

        from aar.interchange import Table
        from aar.stats import DataProfiler

        table = Table(pa.table({"v": pa.array(list(range(distinct_values)),
                                           type=pa.int64())}))
        return DataProfiler().profile_table(table).column("v")

    def test_a_high_cardinality_column_is_not_reported_as_the_limit(self):
        column = self._column(50_000)
        from aar.stats import _DistinctEstimator

        assert column.distinct != _DistinctEstimator.EXACT_LIMIT, (
            "the frozen exact-set size was published as the column's "
            "cardinality")
        assert column.distinct is None, (
            "an approximate count was published as exact")

    def test_the_estimate_tracks_real_cardinality(self):
        for n in (10, 1_000, 20_000, 100_000):
            column = self._column(n)
            assert column.distinct_estimate == pytest.approx(n, rel=0.25), (
                f"estimated {column.distinct_estimate} for {n} distinct values")

    def test_a_repeated_column_is_not_scaled_up_wildly(self):
        """The sqrt extrapolation was applied to every column.

        A 100,000-row column with 5,000 distinct values measured 5,630 in a
        50,000-row sample; the old code multiplied by sqrt(2) for a 100%+
        overshoot on exactly the columns where precision matters most,
        because their distinct count is what drives join selectivity.
        """
        for k in (2, 100, 1_000, 5_000):
            column = self._column_repeated(k, rows=100_000)
            assert column.distinct_estimate == pytest.approx(k, rel=0.30), (
                f"estimated {column.distinct_estimate} for {k} distinct "
                f"values repeated across 100,000 rows")

    def test_a_larger_sample_resolves_a_saturated_ambiguity(self):
        """Documented limit: a half-table sample cannot tell these apart.

        A 100,000-row column of unique integers and one with exactly 50,000
        repeated values both measure ~47,259 distinct in a 50,000-row
        sample, so the profiler reports the table's row count for both.
        Sampling the whole table resolves it: 47,259 against a truth of
        50,000, where the half-table sample reported 94,517.
        """
        truth = 50_000
        rows = 100_000

        half = self._column_repeated(truth, rows=rows,
                                     sample_rows=50_000)
        full = self._column_repeated(truth, rows=rows,
                                     sample_rows=rows)

        # The half-sample cannot resolve it and says so by taking the
        # maximum; the full sample is accurate.
        assert half.distinct_estimate == pytest.approx(rows, rel=0.10)
        assert full.distinct_estimate == pytest.approx(truth, rel=0.15)

    def _repeated_table(self, distinct_values: int, rows: int):
        import pyarrow as pa

        from aar.interchange import Table

        return Table(pa.table({
            "v": pa.array([i % distinct_values for i in range(rows)],
                          type=pa.int64())}))

    def _column_repeated(self, distinct_values: int, rows: int,
                         sample_rows: int = 50_000):
        from aar.stats import DataProfiler

        table = self._repeated_table(distinct_values, rows)
        return DataProfiler(sample_rows=sample_rows).profile_table(
            table).column("v")

    def test_a_low_cardinality_column_stays_exact(self):
        column = self._column(10)
        assert column.distinct == 10
        assert column.distinct_estimate == 10

    def test_the_estimator_reports_which_mode_it_is_in(self):
        from aar.stats import _DistinctEstimator

        small = _DistinctEstimator(100)
        for i in range(100):
            small.add(i)
        assert small.is_approximate is False

        big = _DistinctEstimator(50_000)
        for i in range(50_000):
            big.add(i)
        assert big.is_approximate is True
        # The frozen set is still capped, and the sketch still answers.
        assert big.exact == _DistinctEstimator.EXACT_LIMIT
        assert 40_000 < big.estimate() < 60_000


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


