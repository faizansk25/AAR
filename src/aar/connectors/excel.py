"""Excel: a first-class source and target, not an afterthought.

The specification treats Excel as the place analysts actually live, so this
connector is deliberately forgiving about the things real workbooks do -
merged header rows, blank cells, mixed types, hidden sheets - and deliberately
strict about the one thing that must not be guessed: **which row is the
header**. A silently mis-detected header produces confidently wrong column
names, which is worse than a clear failure.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

from ..failures import (FailureKind, SchemaDriftError, SourceUnavailable)

from ..interchange import Table, require_arrow
from ..types import (BOOLEAN, DATE32, FLOAT64, INT64, NULL, TIMESTAMP, UTF8,
                     Field, Schema)

__all__ = ["read_excel", "write_excel", "detect_header_row", "excel_to_canonical"]


def _openpyxl():
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading and writing Excel needs openpyxl. Install it with:\n"
            "  pip install openpyxl\n"
            f"(import failed: {exc})") from exc
    return openpyxl



# ------------------------------------------------------------------ types
def excel_to_canonical(value: Any) -> Any:
    """Canonical type for a Python value read out of a cell.

    Excel has no real type system, so this infers. The inference is
    deliberately conservative: anything non-numeric becomes text rather than
    being coerced, because a column that is mostly numbers with a stray
    label should read as text and be visible, not silently become NaN.
    """
    if value is None:
        return NULL
    if isinstance(value, bool):
        return BOOLEAN
    if isinstance(value, int):
        return INT64
    if isinstance(value, float):
        return FLOAT64   # NaN and inf are representable and must survive
    if isinstance(value, _dt.datetime):
        return TIMESTAMP("us", None)
    if isinstance(value, _dt.date):
        return DATE32
    return UTF8


def _arrow_type_for(canonical: Any) -> Any:
    from ..interchange import canonical_to_arrow
    return canonical_to_arrow(canonical)


def _infer_column_type(values: list[Any]) -> Any:
    """One type for a whole column: the narrowest that holds every value."""
    seen = set()
    for v in values:
        seen.add(excel_to_canonical(v))
    if not seen:
        return UTF8
    if seen == {NULL}:
        return UTF8
    # A single non-null kind wins; anything mixed falls back to text, which
    # is lossless and visibly wrong rather than lossily wrong.
    non_null = {t for t in seen if t is not NULL}
    if len(non_null) == 1:
        return next(iter(non_null))
    if non_null <= {INT64, FLOAT64}:
        return FLOAT64
    return UTF8


# ------------------------------------------------------------------ header
def detect_header_row(rows: list[list[Any]], declared: int | None = None) -> int:
    """Which 1-based row holds the column names.

    If the caller declared one, it is believed - they have seen the file.
    Otherwise the first row that is mostly non-empty text wins. Guessing
    here is unavoidable in principle, so it is at least explicit and
    reported, and ``header_row`` lets the analyst override it.
    """
    if declared and declared > 0:
        return declared
    for index, row in enumerate(rows[:20], start=1):
        cells = [c for c in row if c is not None and str(c).strip()]
        if len(cells) >= 2 and all(isinstance(c, str) for c in cells):
            return index
    return 1


def _clean_name(raw: Any, index: int) -> str:
    """A usable column name from a header cell.

    Blank headers become ``column_<n>`` rather than empty strings, because an
    empty name breaks every downstream ``select`` and produces a schema that
    cannot be written back out.
    """
    if raw is None:
        return f"column_{index + 1}"
    text = str(raw).strip()
    return text or f"column_{index + 1}"


def _parse_range(spec: str) -> tuple[int, int, int, int]:
    """Parse ``A1:P200000`` into (min_col, min_row, max_col, max_row), 1-based."""
    import re

    m = re.match(r"^\$?([A-Za-z]+)\$?(\d*)\s*:\s*\$?([A-Za-z]+)\$?(\d*)$",
                 str(spec).strip())
    if not m:
        raise ValueError(f"not a cell range: {spec!r} (expected A1:P200000)")
    return (_column_index(m.group(1)), int(m.group(2) or 1),
            _column_index(m.group(3)), int(m.group(4)) or 0)


def _column_index(letters: str) -> int:
    value = 0
    for ch in letters.upper():
        if not ch.isalpha():
            break
        value = value * 26 + (ord(ch) - 64)
    return value



# -------------------------------------------------------------------- read
def read_excel(spec: Any) -> Table:
    """Read a workbook into a :class:`Table`.

    Honours the spec's range addressing: a sheet, a named range, an Excel
    table, or an explicit ``A1:P200000`` range. Formulas are read as their
    cached values by default, which is what an analyst sees when they open
    the file - a formula that has never been calculated has no value at all,
    and that case is reported rather than read as zero.
    """
    import os

    openpyxl = _openpyxl()
    require_arrow()

    path = spec.path
    if not path or not os.path.exists(path):
        raise SourceUnavailable(f"no such workbook: {path}", path=str(path))

    wb = openpyxl.load_workbook(
        path, data_only=(spec.formula_handling != "formulas"),
        read_only=True,
    )
    try:
        ws = _select_sheet(wb, spec)
        if ws is None:
            raise SourceUnavailable(
                f"workbook {path} has no sheet named {spec.sheet!r}; "
                f"it has {', '.join(wb.sheetnames)}")
        rows = _read_rows(ws, spec)
    finally:
        wb.close()

    if not rows:
        return Table.empty(Schema())

    header_at = detect_header_row(rows, spec.header_row)
    header = rows[header_at - 1]
    width = max((len(r) for r in rows), default=0)
    names = [_clean_name(header[i] if i < len(header) else None, i)
             for i in range(width)]
    # Duplicate headers would make a schema that cannot be selected from.
    names = _deduplicate(names)

    body = rows[header_at:]
    records: list[dict[str, Any]] = []
    for row in body:
        if all(c is None or str(c).strip() == "" for c in row):
            continue   # a wholly blank row is formatting, not data
        record = {}
        for i, name in enumerate(names):
            record[name] = row[i] if i < len(row) else None
        records.append(_clean_record(record))

    if not records:
        return Table.empty(Schema(tuple(Field(n, UTF8) for n in names)))

    types = {n: _infer_column_type([r.get(n) for r in records]) for n in names}
    schema = Schema(tuple(Field(n, t) for n, t in types.items()))
    arrow_schema = _arrow_schema(names, types)
    arrow = require_arrow().Table.from_pylist(records, schema=arrow_schema)
    return Table(arrow, schema)


def _select_sheet(wb: Any, spec: Any) -> Any:
    if spec.named_range:
        target = str(spec.named_range)
        if target not in wb.defined_names:
            raise SchemaDriftError(
                f"workbook has no named range {target!r}; "
                f"it defines {', '.join(sorted(wb.defined_names)) or 'none'}",
                named_range=target)
        wb.active = spec.sheet or wb.sheetnames[0]
        return wb[target]
    if spec.table:
        ws = wb[spec.sheet] if spec.sheet else wb.active
        if spec.table not in getattr(ws, "tables", {}):
            raise SchemaDriftError(
                f"sheet {ws.title!r} has no table {spec.table!r}; "
                f"it has {', '.join(sorted(getattr(ws, 'tables', {}))) or 'none'}",
                table=spec.table)
        return ws
    if spec.sheet:
        # openpyxl raises a bare KeyError for a missing sheet, which tells
        # an analyst nothing. Name the sheets that do exist instead.
        if spec.sheet not in wb.sheetnames:
            raise SchemaDriftError(
                f"workbook has no sheet named {spec.sheet!r}; it has "
                f"{', '.join(wb.sheetnames)}", sheet=spec.sheet)
        return wb[spec.sheet]
    return wb.active


def _read_rows(ws: Any, spec: Any) -> list[list[Any]]:
    if spec.cell_range:
        min_col, min_row, max_col, max_row = _parse_range(spec.cell_range)
        rows = []
        for r in ws.iter_rows(min_row=min_row, max_row=max_row or None,
                              min_col=min_col, max_col=max_col,
                              values_only=True):
            rows.append(list(r))
        return rows
    return [list(r) for r in ws.iter_rows(values_only=True)]



def _deduplicate(names: list[str]) -> list[str]:
    """Make header names unique, deterministically.

    Two columns called ``amount`` in one sheet is a real thing that happens;
    an Arrow table with duplicate names cannot be projected, so the second
    gets a suffix rather than the schema being rejected.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise cell values that Arrow will not accept.

    Excel is full of values that are not really data: error strings such as
    ``#DIV/0!``, and floats that are NaN because a formula has no cached
    result. Both become ``None`` here, and the caller reports the affected
    columns rather than letting a text error value masquerade as data.
    """
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and (math.isnan(value)
                                         or math.isinf(value)):
            out[key] = None
        elif isinstance(value, str) and value.startswith("#") and value.endswith(
                ("!", "?", "A", "0", "E")) and " " not in value:
            out[key] = None      # #DIV/0!, #N/A, #VALUE! and friends
        else:
            out[key] = value
    return out


def _arrow_schema(names: list[str], types: dict[str, Any]) -> Any:
    import pyarrow as pa

    return pa.schema([pa.field(n, _arrow_type_for(types[n])) for n in names])


# ------------------------------------------------------------------- write
def _sheet_has_content(ws: Any) -> bool:
    """Whether a sheet already holds a written block.

    ``max_row`` is 1 for a brand-new sheet, so ``max_row > 1`` is the obvious
    test and the wrong one: it reports "has data" for an empty sheet and
    skips the header row, producing a file whose first line is a number. This
    asks whether the used range actually contains a value.
    """
    if ws.max_row <= 1 and ws.max_column <= 1:
        return ws.cell(row=1, column=1).value is not None
    return ws.max_row > 1


def write_excel(table: Table, path: str, sheet: str | None = None,
                mode: str = "overwrite") -> int:
    """Write a table to a workbook. Returns rows written."""
    import os

    openpyxl = _openpyxl()

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    if mode == "append" and os.path.exists(path):
        wb = openpyxl.load_workbook(path)
        ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet or "Sheet1"

    if _sheet_has_content(ws):
        # The sheet already holds a block: put the new one below it.
        start = ws.max_row + 2
    else:
        for c, name in enumerate(table.column_names, start=1):
            ws.cell(row=1, column=c, value=name)
        # Data starts *below* the header. Starting at 1 here writes the
        # header and then immediately overwrites it with the first record,
        # producing a file whose first row is data with no column names.
        start = 2

    row_cursor = start

    for record in table.arrow.to_pylist():
        for c, name in enumerate(table.column_names, start=1):
            value = record.get(name)
            if isinstance(value, (dict, list)):
                value = str(value)
            ws.cell(row=row_cursor, column=c, value=value)
        row_cursor += 1

    wb.save(path)
    return table.num_rows

    return openpyxl
