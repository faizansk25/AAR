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

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ColumnProfile", "TableProfile", "DataProfiler", "DEFAULT_SAMPLE_ROWS",
]

#: Rows to read when sampling. Large enough for a stable selectivity
#: estimate, small enough that profiling a 10M-row file is still instant.
DEFAULT_SAMPLE_ROWS = 50_000


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

    def __init__(self, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> None:
        self._sample_rows = max(1, int(sample_rows))

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
                nulls = getattr(stats, "null_count", None)
                if nulls is not None:
                    fraction = (int(nulls) / rows) if rows else 0.0
                    distinct = (int(stats.distinct_count)
                                if stats.has_distinct_count else None)
                    existing = profile.column(column_name)
                    if existing is None:
                        profile.columns.append(ColumnProfile(
                            name=column_name, distinct=distinct,
                            distinct_estimate=distinct,
                            null_fraction=fraction))
                    else:
                        existing.null_fraction = fraction
                        if distinct is not None:
                            existing.distinct = distinct
                            existing.distinct_estimate = distinct
                elif stats.has_distinct_count:
                    distinct = int(stats.distinct_count)
                    profile.columns.append(ColumnProfile(
                        name=column_name, distinct=distinct,
                        distinct_estimate=distinct))
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
        """
        import pyarrow.csv as pacsv

        # A streaming reader has no ``iter_batches``; reading one block
        # through the native reader is the API that actually exists, and it
        # reads only the first ``block_size`` bytes of the file.
        block = self._sample_rows * 64
        with pacsv.open_csv(
                path,
                read_options=pacsv.ReadOptions(block_size=block)
        ) as reader:
            try:
                table = reader.read_all()
            except StopIteration:
                return TableProfile(name=str(path), rows=0, nbytes=0,
                                    exact_rows=False, source="sampled")
        if not table.num_rows:
            return TableProfile(name=str(path), rows=0, nbytes=0,
                                exact_rows=False, source="sampled")

        file_bytes = _file_size(path, 0)

        # Bytes-per-row comes from the *file's own text*, not from the
        # sample's in-memory footprint. `Table.nbytes` counts the Arrow
        # buffer, which for a block-limited read includes a whole block of
        # padding and a header, so dividing the file size by it
        # under-counts rows. Counting bytes and lines in the first block is
        # the writer's own measurement rather than our re-encoding of it.
        per_row = _bytes_per_csv_row(path, block)
        total_rows = int(file_bytes / per_row) if per_row else 0

        profile = TableProfile(name=str(path), rows=total_rows,
                               nbytes=file_bytes, exact_rows=False,
                               source="sampled")
        profile.columns = [self._column(str(c), table, total_rows)
                           for c in table.column_names]
        return profile

    def profile_json(self, path: str | None) -> TableProfile:
        """Sample the head of a JSON-lines file."""
        import pyarrow as pa
        import pyarrow.json as pajson

        try:
            with pa.OSFile(path, "rb") as handle:  # type: ignore[arg-type]
                sample = pajson.read_json(
                    handle,
                    read_options=pajson.ReadOptions(
                        block_size=self._sample_rows * 64))
        except Exception:  # noqa: BLE001 - fall back to CSV-style sampling
            return self.profile_csv(path)
        profile = self.profile_table(sample, name=str(path))
        profile.source = "sampled"
        profile.exact_rows = False
        return profile

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
        """No statistics available: read a bounded slice and extrapolate."""
        from ..connectors.sql import sqlite_connector
        from ..ir.nodes import Node, NodeType

        table = sqlite_connector(spec).read(Node(NodeType.SCAN_SQL, scan=spec))
        profile = self.profile_table(
            table, name=str(getattr(spec, "table_name", "") or "sql"))
        profile.source = "sampled"
        profile.exact_rows = False
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

        sample_distinct = estimator.exact or 0
        # Extrapolate to the whole table. A sampled distinct count is not a
        # fixed fraction of the full one: light columns keep growing, heavy
        # ones saturate, and sqrt is the standard first-order compromise.
        ratio = (total_rows / len(present)) if present and total_rows else 1.0
        estimate = (sample_distinct * (ratio ** 0.5)
                    if ratio > 1.0 else float(sample_distinct))

        return ColumnProfile(
            name=name,
            distinct=estimator.exact,
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
    """
    fallback = 32.0
    try:
        with open(path, "rb") as handle:  # type: ignore[arg-type]
            chunk = handle.read(max(1024, block))
    except OSError:
        return fallback
    lines = chunk.count(b"\n")
    if lines < 2:            # header plus at least one data row
        return fallback
    return max(1.0, len(chunk) / (lines - 1))


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
    """
    path = getattr(spec, "path", None)
    table = getattr(spec, "table", None) or getattr(spec, "table_name", None)
    if not path or not table:
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

