"""Data profiling: measure the data instead of believing the pipeline's guess.

The cost model picks engines using row counts and byte sizes. Until this
module existed, those numbers came from ``Node.estimated_bytes`` - a value
the pipeline *declares* - or, failing that, a flat 1 MB guess. So the
planner reasoned about workloads it had never looked at, and the dynamic
program above those numbers could only be as good as the inputs beneath it.

What this measures, and why each one changes a decision:

* **row count and byte size** - how much data the next segment must move. A
  declared estimate can be wrong by orders of magnitude.
* **distinct counts and null fractions** - a filter's selectivity depends on
  both. ``amount > 1000`` on a column with ten distinct values is not the
  same as on one with a million.
* **value widths** - bytes per row is not a constant. Short strings and
  64-character strings at the same row count differ by an order of magnitude
  in the bytes that must cross the bus.

Two design commitments:

1. **Bounded sampling.** Statistics come from a sample, capped by row count,
   never a full scan. Deciding how to execute a query must not require
   executing it - that inversion is what makes a planner usable.
2. **Predictions keep their provenance.** Every profile records how it was
   derived, so a plan can be told apart from a guess and a wrong guess can be
   found rather than defended.

When a source can supply its own statistics - Parquet footers, database
catalogs - those are preferred, because they are exact and free.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ColumnProfile", "TableProfile", "DataProfiler", "DEFAULT_SAMPLE_ROWS",
]

#: Rows to read when sampling. Large enough for a stable selectivity
#: estimate, small enough that profiling a 10M-row file is still instant.
DEFAULT_SAMPLE_ROWS = 50_000

#: Hard ceiling on bytes read from one source while sampling. Profiling must
#: be cheaper than execution; reading a 40 GB CSV to plan a query over it
#: is not a plan, it is the first stage of the failure it was meant to avoid.
DEFAULT_SAMPLE_BYTES = 32 * 1024 * 1024

#: How far below 1.0 a sample's distinct-to-row ratio may fall and still be
#: treated as saturated. HyperLogLog at PRECISION=12 carries roughly 1.6%
#: standard error and is biased low on small ranges, so a genuinely
#: unique-per-row column measures about 0.95 rather than 1.0. A cutoff at
#: exactly 1.0 would never fire; one above 0.95 would fire for columns that
#: genuinely repeat.
_SKETCH_TOLERANCE = 0.06


class _ByteCappedFile(io.RawIOBase):
    """A read-only stream that stops yielding after ``limit`` bytes.

    Enforcing the cap *below* the reader is deliberate. PyArrow's own knobs
    are not caps: ``read_all()`` ignores ``block_size`` entirely, and
    ``block_size`` only scales the batch size down to a floor of roughly
    840 rows. A stream that reports EOF is the one bound a future library
    default cannot defeat.

    Subclasses ``io.RawIOBase`` because that is what lets PyArrow's
    ``get_input_stream`` accept it; a plain object with ``read()`` is
    rejected as a closed file.
    """

    def __init__(self, path: str, limit: int) -> None:
        # Set before opening: ``io.RawIOBase`` finalises the object with
        # ``close()`` even when this constructor raises, and close() must
        # not assume the attribute exists.
        self._handle = None
        self._remaining = max(0, int(limit))
        #: Bytes actually handed out, so callers can report the real cost.
        self.bytes_read = 0
        # Opened deliberately without a ``with``: the object's lifetime is
        # the reader's, and ``close()`` is the single release path. Holding
        # it open is what makes ``bytes_read`` meaningful.
        self._handle = open(path, "rb")  # noqa: SIM115

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0 or self._handle is None:
            return b""
        want = (self._remaining if size is None or size < 0
                else min(size, self._remaining))
        data = self._handle.read(want)
        self._remaining -= len(data)
        self.bytes_read += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        finally:
            self._handle = None
            super().close()

    def __enter__(self) -> "_ByteCappedFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(slots=True)
class ColumnProfile:
    """What is known about one column."""

    name: str
    #: Distinct values seen in the sample, or ``None`` if not computed.
    distinct: int | None = None
    #: Distinct values extrapolated to the full table. ``None`` means "not
    #: estimated", which is never the same as zero.
    distinct_estimate: int | None = None
    null_fraction: float = 0.0
    #: Mean rendered length in characters; a proxy for on-disk width.
    avg_chars: float = 0.0
    min_value: Any = None
    max_value: Any = None
    #: True for a column with no usable statistics (a nested type, or a
    #: source whose footer carried none).
    unsupported: bool = False

    def selectivity_of_equality(self) -> float:
        """Expected fraction of rows passing ``column = <one value>``.

        One value out of N distinct keeps roughly ``1/N`` of the rows. This
        is what turns a filter into a row count the cost model can use.
        """
        n = self.distinct_estimate or self.distinct
        if not n:
            return 1.0
        return 1.0 / float(n)

    def render(self) -> str:
        if self.unsupported:
            return f"{self.name}: no statistics available"
        bits = [self.name]
        if self.distinct_estimate is not None:
            bits.append(f"~{self.distinct_estimate:,} distinct")
        elif self.distinct is not None:
            bits.append(f"{self.distinct:,} distinct in sample")
        if self.null_fraction:
            bits.append(f"{self.null_fraction:.0%} null")
        bits.append(f"~{self.avg_chars:.1f} chars")
        return ", ".join(bits)


@dataclass(slots=True)
class TableProfile:
    """What is known about a whole table."""

    name: str = ""
    rows: int = 0
    nbytes: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    #: True when ``rows`` is the true count rather than a sample's.
    exact_rows: bool = True
    #: How this was derived: ``parquet-metadata``, ``sampled``, ``measured``,
    #: ``database-stats`` or ``declared``. Never a silent default.
    source: str = "declared"
    #: Bytes actually observed, when ``nbytes`` was scaled up from a sample
    #: to the whole table. A 50,000-row sample of a 1,000,000-row table
    #: gives the table's real size only by multiplication, and a caller
    #: deciding whether to trust the plan needs to see which it is.
    sampled_nbytes: int | None = None
    #: Rows the sample actually held, for the same reason.
    sampled_rows: int | None = None

    @property
    def bytes_per_row(self) -> float:
        return self.nbytes / self.rows if self.rows else 0.0

    def column(self, name: str) -> ColumnProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def render(self) -> str:
        head = (f"{self.name or 'table'}: {self.rows:,} rows, "
                f"{self.nbytes:,} bytes "
                f"({self.bytes_per_row:.0f} B/row, {self.source})")
        if not self.columns:
            return head
        return head + "\n" + "\n".join("  " + c.render() for c in self.columns)

    def estimate_after_filter(self, selectivity: float) -> int:
        """Rows expected to survive a filter of the given selectivity."""
        return int(self.rows * max(0.0, min(1.0, selectivity)))


# ------------------------------------------------------------ the profiler
class DataProfiler:
    """Collects :class:`TableProfile` objects from real sources.

    Every ``profile_*`` method prefers *declared* metadata - a Parquet
    footer, a database's own statistics - because that is exact and costs
    nothing to read. Sampling is the fallback, and the resulting profile
    records which one happened, so a plan can never present a guess as a
    measurement.
    """

    def __init__(self, sample_rows: int = DEFAULT_SAMPLE_ROWS,
                 max_bytes: int = DEFAULT_SAMPLE_BYTES) -> None:
        self._sample_rows = max(1, int(sample_rows))
        #: Hard ceiling on bytes read from one source while sampling. A row
        #: budget alone is not a memory budget: rows can be arbitrarily wide,
        #: so 50,000 rows of free text is not the same cost as 50,000
        #: integers. Both limits apply, whichever is reached first.
        self._max_bytes = max(1, int(max_bytes))

    # ------------------------------------------------------------- dispatch
    def profile_table(self, table: Any, name: str = "") -> TableProfile:
        """Profile an in-memory :class:`~aar.interchange.Table` exactly.

        An in-memory table knows its own row count and byte size, so nothing
        is sampled and nothing is extrapolated. Only the per-column
        statistics are computed over a bounded sample, because distinct
        counts over 50M rows is precisely the scan this module exists to
        avoid.
        """
        rows = int(table.num_rows)
        profile = TableProfile(name=name, rows=rows, nbytes=int(table.nbytes),
                               exact_rows=True, source="measured")
        sample = self._sample(table)
        profile.columns = [self._column(c, sample, rows)
                           for c in table.column_names]
        return profile

    def profile_node(self, node: Any) -> TableProfile | None:
        """Profile whatever a scan node points at.

        Returns ``None`` for a node with no readable source, so the caller
        falls back to a declared estimate rather than treating "unknown" as
        "empty" - a distinction that matters, because zero rows would plan a
        trivially cheap pipeline.
        """
        spec = getattr(node, "scan", None)
        if spec is None:
            return None
        kind = (getattr(spec, "kind", "") or "").lower()
        try:
            if kind == "parquet":
                return self.profile_parquet(spec.path)
            if kind == "csv":
                return self.profile_csv(spec.path)
            if kind == "json":
                return self.profile_json(spec.path)
            if kind == "excel":
                return self.profile_excel(spec)
            if kind in ("sql", "sqlite"):
                return self.profile_sql(spec)
        except Exception:  # noqa: BLE001
            # A source that cannot be read *now* is not a reason to fail
            # planning. The caller falls back and the plan says so.
            return None
        return None

    # -------------------------------------------------------------- readers
    def profile_parquet(self, path: str | None) -> TableProfile:
        """Exact statistics from the Parquet footer - not one row is read.

        Row count and per-column distinct/null counts live in the footer,
        which is why Parquet is the cheapest thing to profile and the most
        trustworthy: reading 10 bytes of metadata instead of 10 GB of data
        to plan the query.
        """
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(path)  # type: ignore[arg-type]
        meta = handle.metadata
        rows = int(meta.num_rows)
        profile = TableProfile(name=str(path), rows=rows, nbytes=0,
                               exact_rows=True, source="parquet-metadata")
        # Nulls are *summed* across row groups and divided by the file's row
        # count once. The old code divided each group's nulls by the whole
        # file and then overwrote the previous column profile, so the value
        # that survived was whichever group happened to be processed last -
        # a file whose first group is full and second group empty reported
        # 0% nulls.
        nulls: dict[str, int] = {}
        for group in range(meta.num_row_groups):
            row_group = meta.row_group(group)
            for col in range(row_group.num_columns):
                stats = row_group.column(col).statistics
                if stats is None:
                    continue
                column_name = row_group.column(col).path_in_schema
                # Null counts are recorded even when distinct counts are not,
                # so each is applied independently. Requiring both would
                # silently drop every null statistic a writer omitted
                # distinct counts for - and report 0% nulls on a column that
                # is half empty.
                count = getattr(stats, "null_count", None)
                if count is not None:
                    nulls[column_name] = nulls.get(column_name, 0) + int(count)
                # Create the profile on either signal. Keying creation off
                # distinct counts alone meant a column whose writer recorded
                # nulls but not distincts was never added to the profile, and
                # the null total collected for it had nothing to land on -
                # a column that is half empty reported 0% nulls.
                if profile.column(column_name) is None:
                    profile.columns.append(ColumnProfile(name=column_name))
                if not stats.has_distinct_count:
                    continue
                # A per-group distinct count is NOT a file-wide one: the same
                # value can appear in several groups, so summing or keeping
                # either is wrong. The largest group is a lower bound on the
                # true cardinality; ``_mark_parquet_distinct_as_bounds``
                # demotes it from an exact count to an estimate.
                group_distinct = int(stats.distinct_count)
                column = profile.column(column_name)
                assert column is not None
                column.distinct = max(column.distinct or 0, group_distinct)
                column.distinct_estimate = column.distinct

        for name, total_nulls in nulls.items():
            column = profile.column(name)
            if column is not None:
                column.null_fraction = (total_nulls / rows) if rows else 0.0
        _mark_parquet_distinct_as_bounds(profile)
        if not profile.columns:
            # The footer carried no statistics. Name the columns anyway, as
            # unsupported, so the profile does not look like an empty table.
            profile.columns = [ColumnProfile(name=n, unsupported=True)
                               for n in handle.schema.names]
        # Column widths come from chunk sizes, which every Parquet footer
        # has even when the per-value statistics are absent.
        for name, per_row in _bytes_per_column_from_footer(handle,
                                                           rows).items():
            column = profile.column(name)
            if column is None:
                profile.columns.append(ColumnProfile(name=name,
                                                     avg_chars=per_row))
            elif not column.avg_chars:
                column.avg_chars = per_row
        profile.nbytes = _file_size(path, 0)
        return profile

    def profile_csv(self, path: str | None) -> TableProfile:
        """Sample the head of a CSV and extrapolate the row count.

        A CSV carries no metadata at all, so even the row count is an
        estimate: the sample's mean row width divides the file size. The
        profile says ``sampled`` and ``exact_rows=False``, because this is a
        model of the file, not a fact about it.

        **The read is bounded, and the bound is the row count.** Setting
        ``block_size`` is *not* a limit: it chooses the buffer granularity
        and ``read_all()`` then consumes the entire file anyway. Profiling a
        40 GB CSV on an 8 GB laptop would exhaust memory before planning
        started, which is precisely the failure this method exists to
        prevent. See :meth:`_read_bounded_csv` for the row and byte limits.
        """
        try:
            sample = self._read_bounded_csv(path)
        except Exception:  # noqa: BLE001 - unreadable is not fatal
            # PyArrow raises from inside its C++ reader for input the engine
            # itself accepts at a different block size - a field wider than
            # this reader's block, for instance. Profiling must not be
            # stricter than the thing it measures: declining keeps the
            # caller's declared estimate, a guess that says so, where
            # raising fails a run over a file that reads perfectly well.
            sample = None
        if sample is None or not sample.num_rows:
            # A profile with zero rows is the dangerous answer: it would plan
            # a trivially cheap pipeline over a file that was never read. The
            # old code returned one, and a test asserted it. Returning None
            # says "not measured", and the caller keeps its declared
            # estimate - a guess that is labelled as a guess.
            return None
        file_bytes = _file_size(path, 0)
        # Bytes-per-row comes from the *file's own text*, not from the
        # sample's in-memory footprint. `Table.nbytes` counts the Arrow
        # buffer, which includes block padding and a header, so dividing the
        # file size by it under-counts rows. Counting bytes and lines in the
        # first block is the writer's own measurement.
        per_row = _bytes_per_csv_row(path, min(self._max_bytes, 65_536))
        total_rows = int(file_bytes / per_row) if per_row else 0

        profile = TableProfile(name=str(path), rows=total_rows,
                               nbytes=file_bytes, exact_rows=False,
                               source="sampled")
        profile.columns = [self._column(str(c), sample, total_rows)
                           for c in sample.column_names]
        return profile

    def _read_bounded_csv(self, path: str | None):
        """Read at most ``sample_rows`` rows *and* ``max_bytes`` of the file.

        Declines (``None``) rather than raising on a file the reader cannot
        parse. A field wider than the read block raises ``ArrowInvalid:
        straddling object straddles two block boundaries`` from inside the
        C++ reader, and the engine reads such a file happily at its larger
        block size. Profiling must not be stricter than the thing it
        measures: the caller keeps its declared estimate - a guess, honestly
        labelled - rather than the run failing over it.

        Three separate limits are needed, because none of them implies the
        others - measured on this machine, pyarrow 25.0.1:

        * ``read_all()`` reads everything. ``block_size`` is the buffer
          granularity, not a cap: with ``block_size=64_000`` a 200,000-row
          file came back as 200,000 rows.
        * ``block_size`` does scale the batch size, but only down to a
          floor of roughly 840 rows, so a small row budget still over-reads
          by a batch. It is not available as a direct row cap either
          (``batch_rows`` does not exist on ``ReadOptions``).
        * so the byte cap is enforced *below* the reader, by handing it a
          stream that physically runs out. That is the only bound that
          cannot be defeated by a future Arrow default.

        The result is sliced to the row budget afterwards, so the caller
        gets a sample even when a whole batch arrived.
        """
        import pyarrow as pa
        import pyarrow.csv as pacsv

        from ..sources import csv_read_options

        want = self._sample_rows
        budget = self._max_bytes
        if not path or want <= 0 or budget <= 0:
            return None

        collected = []
        rows = 0
        with _ByteCappedFile(path, budget) as capped:
            # Arrow builds a whole batch before yielding it and will not go
            # below roughly 840 rows, so the row budget can only be honoured
            # to within one batch. Reading is still stopped at the budget
            # (measured: 1,550 rows yielded for a 500-row budget on a
            # 200,000-row file, then nothing further) and the table is
            # sliced to the budget afterwards, so the *retained* sample is
            # exact even though the reader over-delivers by one batch.
            #
            # ``newlines_in_values`` is required, not an optimisation. A
            # quoted field containing a newline is legal CSV and is what any
            # export of a multi-line text field produces; without this the
            # parser raises ``ArrowInvalid: got out of sync with chunker``
            # from inside the C++ reader. Profiling must never be the thing
            # that fails a run over a file that reads perfectly well.
            #
            # Parse options come from the engine so both halves agree on
            # quoting and newlines, but the block size is deliberately
            # *not* taken from there. The engine reads a whole file and
            # needs a large block so one wide field cannot straddle two
            # blocks; the profiler stops early, and a large block lets a
            # single batch deliver most of the file - measured: 144,960
            # rows of 200,000 for a 500-row budget, which is precisely the
            # unbounded read this module exists to prevent.
            #
            # So: shared parsing rules, separate read granularity. A field
            # wider than the profiler's block is caught by the byte cap and
            # the plan falls back to the declared size - conservative, and
            # not a failed run.
            parse, _engine_read = csv_read_options()
            reader = pacsv.open_csv(
                capped,
                read_options=pacsv.ReadOptions(block_size=8192),
                parse_options=parse)
            for batch in reader:
                collected.append(batch)
                rows += batch.num_rows
                if rows >= want:
                    break
        if not collected or not rows:
            return None
        table = pa.Table.from_batches(collected)
        return table.slice(0, want) if table.num_rows > want else table

    def profile_json(self, path: str | None) -> TableProfile:
        """Sample the head of a JSON-lines file, under the same byte cap.

        Two defects fixed here. ``read_json`` was given a ``block_size``
        and no row budget, exactly the mistake already corrected for CSV -
        so profiling a large JSON file could read all of it. And on any
        parse failure it fell through to ``profile_csv``, measuring a JSON
        file as though it were delimited text. That is not a conservative
        fallback, it is a wrong answer delivered confidently: the
        column count, the types, and the row estimate would all be
        nonsense, and the plan is built from them.

        An unreadable source now returns ``None`` so the caller keeps the
        declared estimate, which is at least honest about being a guess.
        """
        sample = self._read_bounded_json(path)
        if sample is None or not sample.num_rows:
            return None
        profile = self.profile_table(sample, name=str(path))
        profile.source = "sampled"
        profile.exact_rows = False
        return profile

    def _read_bounded_json(self, path: str | None):
        """Read at most ``sample_rows`` JSON records, under ``max_bytes``.

        Uses the same capped-stream trick as CSV: the byte budget is
        enforced on the stream itself, because ``read_json``'s ``block_size``
        is granularity rather than a limit - the same trap already paid for
        once in ``_read_bounded_csv``.
        """
        import pyarrow as pa
        import pyarrow.json as pajson

        want = self._sample_rows
        if not path or want <= 0 or self._max_bytes <= 0:
            return None
        try:
            with _ByteCappedFile(path, self._max_bytes) as capped:
                reader = pajson.open_json(
                    capped,
                    read_options=pajson.ReadOptions(block_size=8192))
                collected = []
                rows = 0
                for batch in reader:
                    collected.append(batch)
                    rows += batch.num_rows
                    if rows >= want:
                        break
        except Exception:  # noqa: BLE001 - an unreadable source is not fatal
            return None
        if not collected or not rows:
            return None
        table = pa.Table.from_batches(collected)
        return table.slice(0, want) if table.num_rows > want else table

    def profile_excel(self, spec: Any) -> TableProfile:
        """Read a sheet through the Excel connector and profile it.

        XLSX cannot be sliced, so this is a real read rather than a sample.
        That is the same assumption every XLSX reader makes, and the file has
        to be small enough to open at all.
        """
        from ..connectors.excel import read_excel

        table = read_excel(spec)
        return self.profile_table(table, name=str(getattr(spec, "path", "")))

    def profile_sql(self, spec: Any) -> TableProfile:
        """Prefer the database's own statistics, and record that we did.

        SQLite exposes ``sqlite_stat1`` after ``ANALYZE``. Using it is exact
        and free; sampling a table to count its rows is neither.
        """
        rows = _sqlite_stat_rows(spec)
        if rows is None:
            return self._sql_by_sampling(spec)
        return TableProfile(
            name=str(getattr(spec, "table_name", None)
                     or getattr(spec, "table", "") or "sql"),
            rows=rows, nbytes=0, exact_rows=True, source="database-stats")

    def _sql_by_sampling(self, spec: Any) -> TableProfile:
        """No statistics: ask the database for its row count directly.

        Two defects fixed here. The old call passed the whole ``ScanSpec``
        to ``sqlite_connector``, which takes a path - so the fallback raised
        ``TypeError`` for every scan that actually needed it. And the
        "sample" it fell back to was the connector's ordinary read, which
        pulls the entire table; profiling a table in order to plan a query
        over it is the same mistake as the CSV one.

        ``SELECT COUNT(*)`` is evaluated by the database without
        materialising rows, and the column statistics come from a query
        bounded by ``LIMIT``, so both costs are independent of table size.
        """
        import sqlite3

        import pyarrow as pa

        from ..connectors.sql import quote_ident, for_dialect
        from ..interchange import Table
        from ..types import Field, Schema

        path = _sqlite_path_from_connection(getattr(spec, "connection", None)) \
            or getattr(spec, "path", None)
        table_name = (getattr(spec, "table_name", None)
                      or getattr(spec, "table", None) or "")
        label = str(table_name or "sql")
        if not path or not table_name:
            return TableProfile(name=label, rows=0, nbytes=0,
                                exact_rows=False, source="sampled")

        dialect = for_dialect("sqlite")
        quoted = quote_ident(table_name, dialect)
        try:
            conn = sqlite3.connect(path)
            try:
                rows = int(conn.execute(
                    f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
                cursor = conn.execute(
                    f"SELECT * FROM {quoted} LIMIT {int(self._sample_rows)}")
                names = [d[0] for d in cursor.description or ()]
                fetched = cursor.fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 - an unreadable source is not fatal
            return TableProfile(name=label, rows=0, nbytes=0,
                                exact_rows=False, source="sampled")

        arrow = pa.Table.from_pylist([dict(zip(names, r)) for r in fetched]) \
            if fetched else pa.table({n: pa.array([]) for n in names})
        # ``arrow_to_canonical`` is the one place that knows how to turn an
        # Arrow type into a canonical ``DataType``. Building one from
        # ``f.type.name`` looks plausible and raises, because a
        # ``pyarrow.DataType`` has no ``name``.
        from ..interchange import arrow_to_canonical

        schema = Schema(tuple(
            Field(f.name, arrow_to_canonical(f.type)) for f in arrow.schema))
        table = Table(arrow, schema)
        total_sample_rows = int(table.num_rows) or 1
        profile = self.profile_table(table, name=label)
        # The row count is exact - the database counted them - while the
        # per-column statistics and the byte size come from a bounded
        # sample. Those have to be scaled separately, and confusing the two
        # is the whole bug: the old code took the *sample's* byte count and
        # attached the *table's* row count to it, so a 1,000,000-row table
        # measured from a 50,000-row sample reported the sample's 4 MB as
        # the whole table. A plan built on that would allocate a twentieth
        # of the memory the query needs.
        if total_sample_rows and rows > total_sample_rows:
            scale = rows / total_sample_rows
            profile.nbytes = int(profile.nbytes * scale)
            profile.sampled_nbytes = int(table.nbytes)
            profile.sampled_rows = total_sample_rows
        profile.rows = rows
        profile.source = "database-count"
        profile.exact_rows = True
        return profile

    # -------------------------------------------------------------- sampling
    def _sample(self, table: Any) -> Any:
        """The first ``sample_rows`` rows, or the whole table if smaller."""
        n = min(self._sample_rows, int(table.num_rows))
        return table.slice(0, n) if n > 0 else table.slice(0, 0)

    def _column(self, name: str, sample: Any, total_rows: int
                ) -> ColumnProfile:
        """Statistics for one column, over a bounded sample.

        Distinct counts come from a sketch rather than an exact set: an exact
        set over 50M distinct strings is the memory problem this component
        exists to avoid, and a couple of percent of cardinality error still
        beats the declared row count it replaces.
        """
        values = sample.column(name).to_pylist()
        present = [v for v in values if v is not None]
        n = len(values)
        if not present:
            return ColumnProfile(name=name, distinct=0, distinct_estimate=0,
                                 null_fraction=1.0 if n else 0.0,
                                 unsupported=bool(n))

        widths = [len(v) if isinstance(v, str) else 8 for v in present]
        estimator = _DistinctEstimator(len(present))
        for value in present:
            estimator.add(_hashable(value))

        # The estimator has two answers and they are not interchangeable.
        # ``exact`` is the size of the set it kept, which *stops growing*
        # once the limit is hit - so past that point it is a frozen number
        # that no longer responds to the data. Reading it unconditionally,
        # as this did, made every high-cardinality column report the limit
        # as if it were the column's cardinality, and distinct counts feed
        # filter selectivity, which feeds engine choice.
        sample_distinct = estimator.estimate() if estimator.is_approximate \
            else float(estimator.exact or 0)
        # Extrapolate from the sample to the whole table.
        #
        # A sampled distinct count is not a fixed multiple of the full one:
        # a light column (few distinct values) keeps gaining new ones as
        # more rows are read, while a column that is already unique-per-row
        # in the sample cannot gain any - its cardinality is bounded by the
        # table's row count and the sample already found most of it.
        #
        # The old code multiplied by ``sqrt(ratio)`` unconditionally, which
        # is only right for the light case and badly wrong for the heavy
        # one: a 100,000-row column of unique integers, sampled at the
        # default 50,000 rows, measured 47,259 distinct in the sample and
        # was scaled to 66,834 - 33% *below* the truth, in the same
        # direction every time.
        #
        # So the growth factor is derived rather than assumed. The sample
        # saw ``sample_distinct`` distinct values in ``present`` rows; if
        # every row had been distinct, the column is unique and the answer
        # is the row count. The more repeats the sample saw, the closer the
        # column is to its low-cardinality regime, where sqrt is
        # appropriate. The factor is then clamped so the estimate can never
        # fall below what was actually measured, nor exceed the table.
        ratio = (total_rows / len(present)) if present and total_rows else 1.0
        if ratio <= 1.0 or not estimator.is_approximate:
            # An exact count was kept, so the sample saw every distinct
            # value the column has. Reusing it beats any extrapolation: a
            # 10-value column measured exactly must not be reported as 14.
            estimate = float(sample_distinct)
        else:
            # How much of the sample was *not* a repeat of a value already
            # seen. Near 1.0 the column is unique-per-row; near 0.0 every
            # value in the sample had already appeared many times over.
            #
            # The key observation is that the sample usually already
            # contains the column's entire value domain. At k=20,000 in a
            # 100,000-row table, a 50,000-row sample measured 21,151
            # distinct - it had seen every value already - and multiplying
            # that by anything over-counts by 73%. Growth is only warranted
            # when the sample is *saturated*: distinct is pressing against
            # the sample's own row count, which is the one situation where
            # unseen values are certain to exist.
            # Distinct values found as a fraction of the rows sampled. 1.0
            # means every sampled row carried a new value.
            coverage = (sample_distinct / len(present)) if present else 1.0
            coverage = (sample_distinct / len(present)) if present else 1.0
            if coverage >= 1.0 - _SKETCH_TOLERANCE:
                # The sample is saturated: it found as many distinct values
                # as it has rows, so the column *may* be unique-per-row.
                #
                # The truth is not recoverable at this sample size. A
                # 50,000-row sample of a 100,000-row unique column measures
                # ~47,259 distinct, and a 50,000-row sample of a column with
                # exactly 50,000 distinct values measures the same thing - in
                # the second case the sample has already seen every value
                # there is. Both land at coverage 0.945.
                #
                # The table's row count is the only larger number available,
                # and it is the maximum a column could possibly be, so it is
                # used. For a genuinely unique column that is right; for a
                # repeated one it over-states. Raising ``sample_rows`` to
                # cover the whole table resolves it - verified: at 100,000
                # sample rows the same column reports 47,259 against a truth
                # of 50,000, where the 50,000-row sample reported 94,517.
                #
                # The over-statement is the safer direction. Understating a
                # unique column by half makes a wide join look narrow, which
                # costs memory at execution; overstating a repeated column
                # makes a narrow join look wide, which only mis-plans it.
                estimate = min(sample_distinct * ratio, float(total_rows))
            else:
                # Not saturated: the sample saw repeats, so unseen values may
                # still exist further down the table. The uplift is damped by
                # how much of the sample was repeats, keeping a heavily
                # repeated column close to what was actually measured.
                damp = 0.5 * (1.0 - coverage)
                estimate = min(sample_distinct * (ratio ** damp),
                               float(total_rows))

        return ColumnProfile(
            name=name,
            # ``exact`` is only reported while it is exact. Past the limit it
            # is a frozen constant, and publishing it as ``distinct`` would
            # claim a measurement that was never made.
            distinct=None if estimator.is_approximate else estimator.exact,
            distinct_estimate=int(round(max(estimate, sample_distinct))),
            null_fraction=(len(values) - len(present)) / n,
            avg_chars=sum(widths) / len(widths),
            min_value=min(present), max_value=max(present),
        )


class _DistinctEstimator:
    """Distinct-value estimation with a bounded memory footprint.

    Small-cardinality columns are counted exactly. Past a threshold the
    structure degrades to HyperLogLog, whose error is a couple of percent -
    acceptable, because that error is far smaller than the error in the
    *declared* row count this replaces.
    """

    #: Distinct values we are still willing to hold exactly.
    EXACT_LIMIT = 4096
    #: HyperLogLog registers: 2**12 buckets at roughly 1.6% error.
    PRECISION = 12

    def __init__(self, expected: int) -> None:
        big = expected > self.EXACT_LIMIT * 4
        self._exact: set[Any] = set()
        self._buckets: list[int] | None = ([0] * (1 << self.PRECISION)
                                           if big else None)

    def add(self, value: Any) -> None:
        if self._buckets is None:
            self._exact.add(value)
            return
        digest = _hash_int(value)
        index = digest & ((1 << self.PRECISION) - 1)
        rest = digest >> self.PRECISION
        rank = 1
        while rest and not (rest & 1):
            rank += 1
            rest >>= 1
        if rank > self._buckets[index]:
            self._buckets[index] = rank
        # Keep an exact set while it is still small: it makes low-cardinality
        # columns exact rather than approximately exact, and those are the
        # ones whose selectivity is most sensitive to being right.
        if len(self._exact) < self.EXACT_LIMIT:
            self._exact.add(value)

    @property
    def exact(self) -> int | None:
        return len(self._exact)

    @property
    def is_approximate(self) -> bool:
        """True once the exact set stopped growing and the sketch is live.

        The set is capped at ``EXACT_LIMIT``, so a column with more distinct
        values than that keeps a *frozen* count that no longer responds to
        the data. A caller that reads ``exact`` without asking this gets a
        number that looks like a measurement and is really a constant.
        """
        return (self._buckets is not None
                and len(self._exact) >= self.EXACT_LIMIT)

    def estimate(self) -> float:
        if self._buckets is None:
            return float(len(self._exact))
        m = len(self._buckets)
        alpha = 0.7213 / (1.0 + 1.079 / m)
        raw = alpha * m * m / sum(2.0 ** -b for b in self._buckets)
        # Below 2.5m the raw estimate is biased high, and we hold an exact
        # count anyway, so prefer that.
        if raw <= 2.5 * m:
            return float(len(self._exact))
        return raw


def _mark_parquet_distinct_as_bounds(profile: TableProfile) -> None:
    """Demote a Parquet distinct count from fact to lower bound.

    A row group's ``distinct_count`` covers only that group. The file-wide
    cardinality is at least the largest group's and at most the sum of them,
    so reporting any one of them as *the* distinct count claims more than
    the footer supports. Clearing ``distinct`` leaves only
    ``distinct_estimate``, which callers already treat as approximate -
    that is the honest shape for a bound.
    """
    for column in profile.columns:
        if column.distinct is not None:
            column.distinct = None


# --------------------------------------------------------------- helpers
def _bytes_per_column_from_footer(handle: Any, rows: int) -> dict[str, float]:
    """Average bytes per row, per column, from the Parquet footer.

    Parquet records uncompressed and compressed sizes per column chunk, so
    bytes-per-row is available without reading a single value. Without this
    every column reports a width of zero, and a plan built on it cannot tell
    a table of short codes from one of long strings - which is exactly the
    distinction that decides how much has to cross the bus.

    Returns a mapping of column name to mean bytes per row.
    """
    meta = handle.metadata
    if not rows:
        return {}
    totals: dict[str, int] = {}
    for group in range(meta.num_row_groups):
        row_group = meta.row_group(group)
        for col in range(row_group.num_columns):
            column = row_group.column(col)
            # The compressed size is what occupies the file, and therefore
            # what has to be read. Parquet names it explicitly - there is no
            # ``total_size`` attribute, which is the error this line
            # originally raised.
            size = getattr(column, "total_compressed_size", 0) or getattr(
                column, "total_uncompressed_size", 0)
            totals[column.path_in_schema] = (
                totals.get(column.path_in_schema, 0) + int(size))
    return {name: size / rows for name, size in totals.items()}


def _hashable(value: Any) -> Any:
    """A form usable as a set member and a stable sketch input."""
    if isinstance(value, (list, dict, set)):
        return repr(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _hash_int(value: Any) -> int:
    """A 64-bit hash.

    ``hash()`` is salted per process. That is acceptable - a profile is used
    within one planning run and not compared between runs - but the width is
    fixed explicitly so the sketch arithmetic is well defined.
    """
    return hash(value) & ((1 << 64) - 1)


def _bytes_per_csv_row(path: str | None, block: int) -> float:
    """Mean bytes per data row, measured from the file's own text.

    Reads only the first ``block`` bytes and counts the newlines, so the
    cost is bounded and the measurement is the writer's rather than ours.
    A conservative 32 bytes/row floor is returned when the block holds no
    complete data row, because a wrong floor silently mis-plans rather than
    failing.

    **The head alone is not representative.** Measuring only the first
    block on a 300,000-row file whose integer column widens from 1 to 6
    digits gave 5.99 bytes/row against a true 7.63 - a 27% over-estimate of
    the row count, because the head holds the narrowest rows. The file size
    is divided by this number, so the error lands directly on the plan.

    A second block from the middle of the file is therefore also measured
    and the two are averaged. That fixes the systematic bias without
    sampling the whole file, and keeps the read bounded at two small
    blocks rather than one large one.
    """
    fallback = 32.0
    if not path:
        return fallback
    try:
        size = _file_size(path, 0)
        with open(path, "rb") as handle:  # type: ignore[arg-type]
            head = handle.read(max(1024, block))
            estimates = [_per_row_of(head)]
            if size > block * 2:
                # Start half a block in so the window is not re-reading the
                # same rows as the head sample.
                handle.seek(size // 2)
                middle = handle.read(max(1024, block))
                middle_estimate = _per_row_of(middle)
                if middle_estimate:
                    estimates.append(middle_estimate)
    except OSError:
        return fallback
    usable = [e for e in estimates if e]
    if not usable:
        return fallback
    return max(1.0, sum(usable) / len(usable))


def _per_row_of(chunk: bytes) -> float:
    """Bytes per data row in one block, or 0.0 if it holds no full row.

    **A newline is only a record boundary outside a quoted field.** CSV
    permits a quoted value to contain a literal newline - any export of a
    multi-line text field produces one - and counting those as records
    halves the apparent row width, which doubles the row estimate. Measured
    on a 1,000-row file whose every row carried an embedded newline:
    18,898 bytes over 2,000 counted newlines gives 9.4 bytes/row, so the
    file was reported as holding 2,000 rows instead of 1,000.

    The count therefore tracks quote state and only counts newlines seen
    while outside quotes. ``""`` is an escaped quote and does not end the
    quoted run, which is handled by advancing two characters.
    """
    records = _count_records(chunk)
    if records < 1:
        return 0.0
    return max(1.0, len(chunk) / records)


def _count_records(chunk: bytes) -> int:
    """Newlines in ``chunk`` that fall outside a quoted field."""
    in_quotes = False
    index = 0
    newlines = 0
    length = len(chunk)
    while index < length:
        char = chunk[index]
        if char == 0x22:            # '"'
            if in_quotes and index + 1 < length and chunk[index + 1] == 0x22:
                index += 2          # escaped quote, still inside the value
                continue
            in_quotes = not in_quotes
        elif char == 0x0A and not in_quotes:
            newlines += 1
        index += 1
    return newlines


def _file_size(path: str | None, fallback: int) -> int:
    """Bytes on disk, which is what actually has to be read."""
    try:
        return int(os.path.getsize(path))  # type: ignore[arg-type]
    except OSError:
        return fallback


def _sqlite_stat_rows(spec: Any) -> int | None:
    """Row count from ``sqlite_stat1``, or ``None`` when unavailable.

    ``sqlite_stat1`` exists only after ``ANALYZE`` and only for indexed
    tables. Both are common, so a miss is normal rather than an error.

    **The database is found through ``connection``, not ``path``.** The SQL
    SDK builds ``ScanSpec(kind="sql", connection=..., table_name=...)`` and
    never sets ``path``, so reading only ``path`` meant this always returned
    ``None`` for a scan created through the public API - every SQL pipeline
    silently fell back to reading the whole table just to count its rows.

    Resolution goes through :func:`aar.sources.resolve_sql_connection`, the
    same helper the Arrow engine calls. The engine used to read ``spec.path``
    and fall back to ``:memory:``, so a pipeline could be profiled against
    the real database and executed against an empty one. Two consumers, one
    answer, so they cannot drift apart again.
    """
    table = getattr(spec, "table", None) or getattr(spec, "table_name", None)
    if not table:
        return None
    from ..sources import sql_file_path

    path = sql_file_path(getattr(spec, "connection", None)) \
        or getattr(spec, "path", None)
    if not path:
        return None
    try:
        import sqlite3

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT stat FROM sqlite_stat1 WHERE tbl = ? "
                "AND stat IS NOT NULL LIMIT 1", (table,)).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - no statistics is an ordinary outcome
        return None
    for (stat,) in rows:
        # Format: "<rows> <distinct key values> <rows> <distinct values> ..."
        parts = str(stat).split()
        if parts and parts[0].isdigit():
            return int(parts[0])
    return None


def _sqlite_path_from_connection(connection: Any) -> str | None:
    """The filesystem path in a SQL connection string, if there is one.

    SQLite is the only dialect whose connection names a file rather than a
    host, so this is the only place a path can be recovered. A DSN pointing
    at a server, or an in-memory database, yields ``None`` and the caller
    falls back to sampling.
    """
    if not connection:
        return None
    if isinstance(connection, str):
        text = connection.strip()
        # A URI (file:...?mode=ro) still names a file, but the prefix has to
        # come off before sqlite3.connect will accept it.
        if text.startswith("file:"):
            return text[len("file:"):].split("?", 1)[0]
        # A host:port DSN names a server, not a file.
        if (not text or text == ":memory:"
                or text.startswith(("http", "postgres", "mysql", "trino"))):
            return None
        return text
    # An open ``sqlite3.Connection`` has no public way to report the file it
    # was opened with - the stdlib type exposes no such attribute, verified
    # against this interpreter. Returning None makes the caller sample
    # instead, which is the safe direction: guessing a path could profile a
    # different database from the one the pipeline reads.
    return None

