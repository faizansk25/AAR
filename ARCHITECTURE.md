# Structural analysis: pydeps + pyan3

Regenerate with `python tools/cycle_roots.py`. Both tools are
dev-only; neither is a runtime dependency, and `verify_release.py`
still reports zero third-party dependencies for the installed wheel.

```powershell
pip install pydeps pyan3
pydeps --no-output --show-cycles src\aar\__init__.py
pyan3 --module-level --root . src\aar --text -C
python tools\cycle_roots.py
```

## Import cycles

`pyan3 -C` reports **155 cycles**, which is a rotation
count, not a problem count. Grouped by member set that is
**36 distinct module sets**:

- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.cudf_engine`, `engines.factory`
- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.duckdb_engine`, `engines.factory`
- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.excel_engine`, `engines.factory`
- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.pandas_engine`
- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.polars_engine`
- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.polars_gpu_engine`
- **6 rotations, 6 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.python_engine`
- **5 rotations, 5 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`
- **5 rotations, 5 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.cudf_engine`, `engines.factory`
- **5 rotations, 5 modules:** `connectors.__init__`, `connectors.mongo`, `engines.__init__`, `engines.excel_engine`, `engines.factory`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.cudf_engine`, `engines.factory`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.duckdb_engine`, `engines.factory`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.excel_engine`, `engines.factory`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.pandas_engine`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.polars_engine`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.polars_gpu_engine`
- **5 rotations, 5 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.python_engine`
- **4 rotations, 4 modules:** `connectors.mongo`, `engines.__init__`, `engines.arrow_engine`, `engines.factory`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.cudf_engine`, `engines.factory`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.duckdb_engine`, `engines.factory`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.excel_engine`, `engines.factory`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.pandas_engine`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.polars_engine`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.polars_gpu_engine`
- **4 rotations, 4 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.factory`, `engines.python_engine`
- **3 rotations, 3 modules:** `cli`, `workbench.__init__`, `workbench.server`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.arrow_engine`, `engines.factory`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.cudf_engine`, `engines.factory`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.duckdb_engine`, `engines.factory`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.excel_engine`, `engines.factory`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.factory`, `engines.pandas_engine`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.factory`, `engines.polars_engine`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.factory`, `engines.polars_gpu_engine`
- **3 rotations, 3 modules:** `engines.__init__`, `engines.factory`, `engines.python_engine`
- **2 rotations, 2 modules:** `engines.__init__`, `engines.factory`
- **2 rotations, 2 modules:** `workbench.__init__`, `workbench.server`

### Why they are all here anyway

Every back-edge is a **function-local import**, not a module-level
one. That is deliberate: it is what keeps `import aar` free of
pyarrow, duckdb, polars, pandas, cudf and openpyxl. An engine's
third-party import happens when the engine is *constructed*, which is
exactly when the dependency is known to be wanted.

So the cycles are the price of lazy loading. They are benign at
runtime, because Python only enters them at call time. The count is
the same with and without pyan3's `--init` flag, so it is not an
artifact of that flag's implicit package-`__init__` edges.

The real risk is not import failure. It is that a cycle makes a
*conceptual* boundary negotiable: a reader who sees
`engines.arrow_engine` importing `connectors.excel` cannot tell
whether that is a mistake or a rule. Hence this file - so the count
is visible, and a growth in it is a deliberate decision rather than
an accident. If a cycle ever needs breaking, invert the dependency
(a write-format registry that engines read and connectors register
into), do not shuffle the import into another function.

## What pydeps found

`--show-cycles` on the package entry point reports **no cycles
reachable from `aar.__init__` alone** - the path a plain `import aar`
takes. Everything above is reachable only once something calls into
an engine or the workbench.

External dependencies in the import graph are exactly the optional
extras (pyarrow, duckdb, polars, pandas, cudf, openpyxl, pymongo,
psycopg, pynvml). Nothing unexpected, and nothing unguarded at
module scope.

## Changes made because of this review

- **`src/aar/examples.py` + `aar examples`.** The same review found
  the CLI's worst ergonomic gap: `run` and `explain` both demand a
  pipeline file, and nothing told a user what one looks like or gave
  them one. Now `aar examples` lists them, `--show` prints one, and
  `--write NAME PATH` writes an editable copy. The templates are
  embedded in the package rather than shipped as data files, so
  there is no `package-data` entry to forget at build time and no
  difference between a checkout and an installed wheel.
- The examples are **tested as code**: every template is parsed,
  every `aar.sdk` import is checked against the live `__all__` (so
  renaming an SDK function fails a test rather than a user), no
  third-party import is allowed in an example, and the shortest one
  is written to disk and run through `aar explain` for real.
- `--write` refuses to overwrite. A pipeline someone has edited is
  theirs.
