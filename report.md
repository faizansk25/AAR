# AAR — Build Progress Report

**Project:** Adaptive Analytics Runtime
**Workspace:** `d:\AAR`
**Repository:** https://github.com/faizansk25/AAR.git (branch `main`)
**Specification:** `system.md`
**Status:** AAR runs pipelines, propagates privacy, and enforces it · 512 tests
**Last updated:** 2026-09-28

---

## 1. Vision

Give an analyst one execution environment for Excel, SQL, NoSQL, Python, local
files and large-scale data, while automatically selecting the most efficient
**legal** execution path for the available hardware — with full explainability
and zero external AI dependency. The goal is **minimum total analytical cost**,
not GPU utilisation.

---

## 2. System creation status

| # | Layer | Spec ref | Status | Verified by |
|---|---|---|---|---|
| 1 | Canonical type system | §11 | ✅ **Complete** | 42 tests |
| 2 | Internal Analytics IR | §4 | ✅ **Complete** | 20 tests |
| 3 | Hardware profiler + calibration | §5 | ✅ **Complete** | 32 tests + live |
| 4 | Failure registry / never-silently-fail | §15, §23 | ✅ **Complete** | 20 tests |
| 5 | Capability registry (engine × op × dtype) | §4 | ✅ **Complete** | 41 tests |
| 6 | Cost model + transfers + history | §6 | ✅ **Complete** | 42 tests |
| 7 | CLI — `aar doctor / engines / calibrate / explain / run` | §16 | ✅ **Complete** | 30 tests + live |
| 8 | Adaptive planner (segment DP) | §7 | ✅ **Complete** | planner tests + live |
| 9 | Execution engines (Arrow, DuckDB, Polars, pandas, UDF, Excel) | §8 | ✅ **Complete** | cross-engine agreement |
| 10 | Arrow interchange layer | §9 | ✅ **Complete** | 16 tests |
| 11 | Connectors — Excel + Parquet/CSV/JSON | §10 | ✅ **Complete** | 9 tests + live |
| 11b | Connectors — SQL, MongoDB | §10 | 🟡 **Partial — pushdown + typing done** | SQLite: real database · Mongo: mongomock |
| 11c | Topological executor + `aar run` | §7, §16 | ✅ **Complete** | end-to-end + smoke |
| 12 | Metadata & lineage | §12 | ✅ **Complete** | 28 tests + live |
| 13 | Privacy, security & governance | §13 | ✅ **Complete** | 37 tests + live |
| 14 | Scheduler & resource manager | §14 | 🟡 Partial (budgets + profile) | — |
| 15 | Explainability & observability | §16 | 🟡 Partial (`explain` + run trace + policy audit) | — |
| 16 | Execution history feedback | §6 | ✅ **Complete** | history tests |
| 16 | Runtime executor + cache | — | ⬜ Not started | — |
| 17 | SDK (`ctx.excel()` …) | §1 | ⬜ Not started | — |
| 18 | CLI — `aar explain plan` | §16 | ⬜ Blocked on the planner | — |
| 19 | Analyst Workbench UI | §17 | ⬜ Not started | — |

**Progress: 8 of 19 layers complete (42% of the specified system), plus
3 partial.** The completed layers form the chain from "what does this data
mean" through "what can this machine do" to "what will it cost" to "let me
see and change it". Each was built to be independently useful and testable
before its consumer existed.

**What still does not work, stated plainly:** there is no executor, no
connectors and no SDK, so an analyst cannot yet point AAR at a file and get an
answer. Everything built so far is substrate. `aar explain plan` is
deliberately absent rather than stubbed, because a command that prints "not
implemented" is worse than no command.


### 3.6 Capability registry — `src/aar/capability/registry.py`

The answer to "which engine can do this operation, and is it even
installed?" — split deliberately into two questions:

- **Declared** — a static catalogue of 16 engines across six priority tiers,
  each with its device class, the operations it implements, and its
  provenance. This is what lets `aar explain plan` produce a plan on a machine
  that could not run it.
- **Available** — a memoised import probe using `find_spec`, never an actual
  import, because importing cuDF or Spark costs seconds and allocates device
  memory just to answer "is it there?".

The **feasible set** is the intersection of three filters, each reportable:
what the node declares, what the catalogue says, and what is installed. Every
rejection carries a sentence — "duckdb is not installed", "no GPU (no /dev/
nvidia* device nodes)", "does not support PythonUDF" — because a rejection
the analyst cannot see is one they will not trust.

Hard capability facts are enforced, not preferences: **no GPU engine claims
`PythonUDF`**, asserted by test. Memory fitness distinguishes VRAM from RAM,
so a 50 GB dataset is rejected by an 8 GB card no matter how much system
memory is free.

### 3.7 Cost model — `src/aar/cost/model.py`

    Cost(O, E) = T_startup + T_read + T_transfer + T_compute + T_spill + T_materialize
    T_GPU      = T_H2D + T_kernel + T_D2H + T_startup

- **Three sources, in descending trust**: execution history, a measured
  calibration curve, or a documented pessimistic prior. Whichever was used is
  returned and recorded, so a plan built on priors is visibly distinct from
  one built on measurements.
- **The specification's GPU example is reproduced by the model, not asserted
  by hand.** Given an 80 ms GPU kernel against 400 ms of CPU work and
  240+160 ms of transfers, `node_cost` reports a faster kernel and a slower
  total — the correct answer, reached by arithmetic.
- **Transitions are first class.** Same engine: free. CPU↔CPU in-process:
  genuinely free (C Data Interface, no copy, no serialisation). CPU↔GPU: the
  bus both ways. GPU↔GPU: a peer copy, not a host round trip. Cross-network:
  network bandwidth.
- **Residency is tracked**, so two consecutive GPU nodes pay the bus once, not
  twice — the mechanism that makes segment optimisation worth doing.
- **Spill is charged only when the working set actually exceeds VRAM**;
  charging it unconditionally would bias the optimiser toward CPU for reasons
  unrelated to reality.
- **`ExecutionHistory`** learns locally and deterministically: a lookup for
  one observation, EWMA for several, least-squares across sizes for more.
  Failed runs are never averaged in — a crash's duration is not a cost — and
  a non-monotone history falls back to a constant rather than predicting that
  bigger is faster.



---

## 3. What has been built

### 3.1 Canonical type system — `src/aar/types.py`

The defence against the verified failure mode where pandas and Polars produce
different payloads for equivalent temporal types and a timestamp column is
silently corrupted during an engine handoff.

- **28 canonical types** across integers, floats, decimals, temporals, binary,
  nested (list/struct/map) and categorical.
- **Total source mapping** for PostgreSQL, MySQL, SQLite, Spark, MongoDB,
  Excel, Arrow/Parquet, Polars, pandas, DuckDB, cuDF. Unknown types raise
  `UnmappableType` — nothing is guessed, and there is no silent fallback to
  text.
- **`lossy(src, dst)`** — a single, authoritative function answering "is this
  conversion safe?" with a human-readable reason. It encodes signed/unsigned
  boundary rules, integer narrowing, timestamp resolution and timezone
  changes, float truncation, and nested-type recursion.
- **`SchemaDiff`** — structured schema-drift detection (added / removed /
  retyped / nullability-changed), feeding failure mode #5.
- **`ConversionLog`** — append-only record of every boundary conversion, with
  lossy entries surfaced for review.
- **Classification travels with data**: `Field` carries `classification` and
  `lineage`, and both survive `cast`, `rename` and `select`. A `CONFIDENTIAL`
  tag applied in PostgreSQL is still present when the column lands in Excel.

### 3.2 Internal Analytics IR — `src/aar/ir/nodes.py`

Deliberately **not** Substrait: Substrait is pre-1.0 and has no vocabulary for
Excel ranges, Python UDFs, privacy classifications or materialisation
boundaries.

- **25 node types** across scan (8), transform (11), Python UDF, quality
  check, and sink/boundary nodes.
- **Typed expression algebra** — `Col`, `Lit`, `BinOp`, `UnaryOp`, `Func`,
  `Agg`, `CastExpr`, `WindowSpec`. Expressions are a closed algebra, not a
  string DSL, so the planner can analyse them and pushdown can render them to
  SQL. Literals are properly escaped (verified by an injection test).
- **One `Node` type for logical and physical decisions** — `assigned_engine`,
  `reason`, `estimated_ms`, `expected_saving_ms` and `fallback_engine` are
  fields you read, not a diff you compute. This is what makes "why did this
  run on the GPU" a one-line answer.
- **Deterministic topological ordering** with cycle detection, so two runs
  over the same graph produce byte-identical plans.

### 3.3 Hardware profiler — `src/aar/hardware/detect.py`

- Six probes, all pure standard library, none of which can raise: OS, CPU,
  memory, GPU, storage, network, software versions.
- **GPU detection with a documented reason.** Four evidence tiers (NVML →
  `nvidia-smi` → Apple unified memory → a CUDA runtime), and on failure a
  plain-language explanation ("no /dev/nvidia* device nodes and no vendor
  tooling"). A missing GPU is an *explained decision*, not an absence of
  information.
- **What could not be measured is recorded.** Every probe carries an
  `unknowns` list. "We could not measure this" is always distinguishable from
  "we assumed this".
- **Container-aware budgets.** `memory_budget_bytes` uses available memory,
  further capped by cgroup limits — planning against host memory inside a
  4-core/8 GiB cgroup is how a plan becomes an OOM.
- **Stable `fingerprint()`** that excludes volatile quantities (free memory)
  but includes engine versions and VRAM, so a calibration profile is reused
  on the same machine and discarded on a different one.
- **`HardwareProfile` facade** with lazy memoisation: one coherent snapshot,

### 3.4 Microbenchmark calibration — `src/aar/hardware/calibrate.py`

Where the eighth design principle ("empirical, not hardcoded") is cashed in.

- **10 CPU benchmarks against real generated data**: scan, filter, sort,
  groupby, window, hash join, string ops, Arrow IPC, Parquet decode, CSV
  decode — plus sequential disk read and GPU transfer/kernel benchmarks.
- **Fitted cost curves** `t = fixed + slope·n + power·n^exp` with R²
  retained, so the planner can widen its margin when a model is poor.
- **Physically-constrained fitting.** Cost is monotonically non-decreasing in
  size and never negative. Unconstrained least squares will happily return a
  negative slope when a noisy measurement inverts; such fits are rejected and
  replaced by a *conservative* interpretation (over-estimating cost biases
  toward CPU, which always exists, rather than toward an accelerator that may
  not be faster).
- **Measured on this machine, today:**

  ```
  scan/cpu              3.441e-11 s/byte   R2=0.991
  filter/cpu            1.261e-09 s/byte   R2=0.999
  groupby/cpu           3.267e-10 s/byte   R2=0.953
  hash_join/cpu         2.728e-09 s/byte   R2=0.997
  sort/cpu              4.943e-09 s/byte   R2=1.000
  window/cpu            2.435e-09 s/byte   R2=0.970
  parquet_decode/cpu    1.020e-08 s/byte   R2=0.991
  csv_decode/cpu        1.508e-08 s/byte   R2=1.000
  arrow_ipc/cpu         5.444e-10 s/byte   R2=0.999
  sequential_read/disk  6.254e-10 s/byte   R2=1.000
  ```

  All 11 curves are high confidence. Cross-device crossover is computed from
  these, not asserted.
- **A profile from a different machine is discarded**, not used. A stale
  profile is worse than none: it is confidently wrong.
- **Honest caveats are stored and displayed**, e.g. "page cache could not be
  dropped; `sequential_read` is a warm-cache figure and understates cold I/O".

### 3.5 Failure registry — `src/aar/failures/registry.py`

- **All 22 failure modes** from the specification's matrix, each with
  detect / handle / fallback / log template and a severity.
- **Every error AAR raises carries a `failure_mode` tag**, so a bare
  exception is already a bug and any handler can tell what went wrong without
  parsing a message string.
- **`DegradationLedger`** — the structural guarantee against silent failure.
  Every fallback taken is recorded with cause, from-engine, to-engine and
  severity. `assert_clean()` raises if any BLOCKING degradation went
  unresolved, so a run that needs a human cannot return a success object.
  A broken sink cannot mask the event it was meant to report.


---

## 4. Testing

**440 tests: 435 passing, 5 skipped, 0 failing.** 47 s. Every skip states the
missing dependency rather than passing vacuously.

| Suite | Tests | Coverage |
|---|---|---|
| `test_types.py` | 42 | Source mapping, loss detection, coercion, schema algebra, drift, conversion log |
| `test_capability.py` | 38 | Catalogue integrity, declared vs available, feasible sets, rejection reasons, memory fitness, ledger integration |
| `test_cost.py` | 42 | Cost breakdown, three cost sources, **the spec's GPU example**, transitions, memory, history, explain |
| `test_hardware.py` | 32 | Byte helpers, all 6 probes, profile facade, curve fitting, store round-trip, stale-profile rejection, **live calibration** |
| `test_failures.py` | 20 | Full matrix coverage, error hierarchy, ledger semantics |
| `test_ir.py` | 20 | Expressions incl. SQL-injection safety, node metadata, DAG ordering, cycle detection |
| `test_planner.py` | 24 | Segment DP, transitions charged once, feasibility, explain output |
| `test_cli.py` | 37 | `doctor` / `engines` / `calibrate` / `explain` / `run` / `policy`, exit codes |
| `test_runtime.py` | 125 | **Cross-engine agreement, interchange, connectors, executor, end to end** |
| `test_governance.py` | 37 | Egress, sensitivity, RLS, CLS, masks, enforcement in a real run |
| `test_lineage.py` | 28 | **Propagation into derived columns, source→aggregate→policy** |



Tests are behavioural, not coverage-chasing. Examples of what they pin down:

- `test_string_literal_is_not_injectable` — a value containing
  `'; DROP TABLE users; --` renders as a quoted literal.
- `test_monotonicity_is_never_violated` — a noisy measurement cannot produce
  negative cost.
- `test_profile_from_other_machine_is_discarded` — calibration from a
  different fingerprint is rejected.
- `test_blocking_degradation_fails_the_run` — the never-silently-fail
  contract is enforced, not aspirational.
- `test_model_reproduces_the_same_conclusion` — the specification's GPU
  worked example, run through the real cost model.
- `test_every_registered_engine_can_actually_be_built` — catches an engine
  whose methods were misplaced, which a syntax check cannot see and which
  would degrade silently on every single use.
- `test_group_by_returns_one_row_per_group` — names the defect instead of
  reporting a downstream `KeyError` on a region that is simply missing.
- `test_write_then_read_back` — every engine, every format. A file the engine
  wrote and cannot itself read is still a broken pipeline.
- `test_null_keys_do_not_join_to_the_string_none` — the classic NULL-join
  bug, pinned shut.
- `test_quick_calibration_measures_real_operations` — a real microbenchmark
  run against this machine in the test suite.

**Cross-engine agreement** is the property that matters most in the runtime
suite: every engine that is installed runs the same assertions and must produce
the same rows. A divergence between DuckDB and Polars is exactly the silent
corruption the Arrow interchange layer exists to prevent.

### Development tooling (`tools/`)

- `check_syntax.py` — normalises UTF-8/LF and reports parse errors with
  context. Written because PowerShell redirection on Windows introduces a
  BOM, which is a hard syntax error for Python.
- `check_assets.py` — validates `logo.svg` structure and theme support.
- `make_sample_data.py` — writes `FY26-orders.xlsx`, so the example pipeline
  can actually be executed rather than only explained.
- `smoke_run.py` — **the check that matters most**: generates Parquet, runs a
  full pipeline through `aar explain` and `aar run`, then reads the written
  Parquet and Excel back and checks the numbers against an independent Python
  calculation. No mocks anywhere in the path.
- `debug_calib.py` — runs each benchmark individually and reports the
  exceptions the calibration harness deliberately swallows.
- `smoke.py` — end-to-end manual verification of the planning stack.
- `truncate.py` — safely drop an orphaned trailing block after an interrupted
  edit.

---

## 5. Bugs found and fixed during verification

Testing found real defects, not just typos. The notable ones:

| Bug | Impact | Fix |
|---|---|---|
| Integer bit-width computed as `enum_index // 4 * 8 + 8` | Labelled UINT8 as 16-bit and INT16/32/64 as 8-bit, corrupting every range check | Explicit `_INT_BITS` table |
| `datetime.timezone.utc.key` | `AttributeError` on any non-`zoneinfo` tzinfo | `_tz_name()` handling all tzinfo types |
| `_gauss` identity matrix used undefined `i` | Any 3-point fit raised `NameError` | Correct index |
| Benchmark warm-up not discarded | First `group_by` measured **46 ms** vs 0.91 ms warm — a 40x error that made every small-input cost wrong | Warm-up call discarded before timing |
| Quick ladder only 2 sizes | Fixed term and per-byte term inseparable; all curves flat and the crossover table meaningless | 3-size minimum, documented why |
| `low_confidence = len(pts) < 4` | Flagged every curve low confidence despite R2 0.95-1.00 | Threshold corrected to 3 |
| `ChunkedArray.sum()` | `scan` never calibrated on this Arrow build | `pc.sum()` kernel instead |
| Several PowerShell launches per probe | Multi-second stall per plan; made the test suite appear to hang | One JSON round trip, memoised |
| `Schema(Field(...))` | Opaque `'Field' object is not iterable` | Accepts a single Field, with a helpful error |
| Serialisation charged for in-process CPU→CPU | Charged 20 ms for a handoff the C Data Interface makes free — would make the segment optimiser avoid good local handoffs | Zero-copy handoff costs nothing |
| GPU spill charged unconditionally | Every GPU plan paid a spill cost for work that fits in VRAM, biasing toward CPU for no real reason | Charged only when the working set actually exceeds VRAM |
| `ScanConst` had no capable engine | An IR node type nothing could execute — a hole in the catalogue | Added to the columnar op set |
| `add_points` called per point in a test helper | Each call re-fits, leaving a one-point curve and a spurious low-confidence flag | Add all points in one call |
| `DuckDBEngine` methods nested inside a module-level function | The class silently became abstract; every use degraded to Polars while still returning correct numbers | Rewrote the file; added a test that constructs every registered engine and asserts the contract is complete |
| DuckDB `group_by` fetched its result *after* unregistering the relation | Returned an **empty table instead of raising** — a silently wrong aggregation | Materialise before `release()`; added a row-count assertion so this class of failure names itself |
| `duckdb.connect(config=None)` | DuckDB rejects the kwarg; every connect failed and degraded | Omit the kwarg when there is nothing to configure |
| `HEADER` passed to `COPY ... FORMAT PARQUET` | Syntax error; the Parquet write degraded every run | Options built per format — `HEADER` is CSV-only |
| `Table()` handed a Polars DataFrame instead of Arrow | Walked a `{name: dtype}` dict as if it were an Arrow schema, failing far from the boundary | Every read path converts through `_table()` |
| `canonical_to_arrow` returned `pa.int64`, not `pa.int64()` | Every non-null type conversion returned a *function object* | Constructors are called, not referenced |
| Excel header written at row 1, data also starting at row 1 | The first record overwrote the header; output files had no column names | Data starts at row 2 when a header is written |
| `ws.max_row > 1` used to test "sheet has content" | A brand-new sheet reports `max_row == 1`, so the header was skipped entirely | Check the used range for an actual value |
| `require_arrow`, `_import_polars`, `_import_openpyxl`, `_read_excel` lost their `return` | Every Arrow operation raised `AttributeError: 'NoneType'` | Restored; covered by the end-to-end tests |
| UDF arity guessed to mean "column" | `def risk_band(amount)` was handed a list, failing inside the user's function | `mode` is now explicit; `"row"` is the documented default and column mode receives *all* columns |
| Column UDF received only the first column | A UDF could quietly operate on the wrong data and return a plausible wrong answer | Column mode receives a dict of every column |
| `BinOp` accepted a raw Python value | `cannot evaluate int` deep inside the predicate compiler | `__post_init__` lifts non-`Expr` operands to `Lit` |
| NULL join keys matched each other | Fabricated join rows no engine agrees on | SQL semantics: `NULL` never equals `NULL` |
| `pc.invert(mask)` passed to `take()` | `take` wants positional indices; a boolean array is not one | `Table.filter` takes the mask directly |
| `agg_functions` holds a tuple of `Agg`, engines expected one | Every group-by raised "expected an aggregate, got tuple" | Normalised in the executor, with a clear error for the unsupported multi-aggregate case |
| `sensitivity_of` lost its `return` | Every classification evaluated to `None`, so no policy could ever match a column | Restored |
| RLS literal compared as a string | A rule like `amount = 3.0` against a float column filtered out *every* row and looked like it worked | Literal is typed from the column's declared type |
| `Lit` imported from `aar.interchange` in the policy engine | That module exports only the `Table`; every RLS evaluation raised `ImportError` | Imported from `aar.ir`, where it lives |
| Unknown key in a policy file silently ignored | A misspelled `mask_threshold` leaves a policy that looks configured and protects nothing | `policy_from_dict` rejects unknown keys by name |
| Column UDF received only the first column | A UDF could quietly operate on the wrong data and return a plausible wrong answer | Column mode receives a dict of every column |
| BOM from PowerShell redirection | Hard syntax error in every written file | `check_syntax.py` normaliser |


---

## 6. Environment

- **Dedicated venv at `d:\AAR\.venv`** (Python 3.14.7), created for this
  workspace and not shared with any other project.
- Installed: `pyarrow 25.0.1`, `duckdb 1.5.5`, `polars 1.44.2`,
  `pandas 3.0.6`, `openpyxl 3.1.5`, `pytest 9.1.1`, `pyyaml`, `pytest-timeout`.
- **Zero required third-party runtime dependencies.** The four completed
  layers import and run on the standard library alone; every engine is
  optional, probed at runtime, and its absence logged rather than swallowed.
- Machine detected: Windows 10, Intel i5-7200U (4C/4T), 8.6 GB RAM, no GPU,
  SSD, egress `deny`.

---

## 7. What is required next

Ordered by dependency. Each item is scoped to be buildable and testable
against the layers that already exist.

### 7.1 ~~Capability registry~~ — **DONE** (`aar/capability/`)

Delivered: 16 engines across 6 tiers, declared-vs-available split, memoised
`find_spec` probe, feasible sets with per-engine rejection reasons, VRAM/RAM
memory fitness, `ENGINE_ABSENT` ledger integration. 38 tests.

### 7.2 ~~Cost model and segment transitions~~ — **DONE** (`aar/cost/`)

Delivered: the full six-term breakdown, three-source cost estimation
(history / calibration / pessimistic prior), GPU transfer accounting that
reproduces the specification's worked example, transition costs by device
pair, residency tracking, VRAM-aware spill, and a local deterministic
`ExecutionHistory`. 42 tests.

### 7.3 ~~Adaptive planner~~ — **DONE** (`aar/planner/`)

Delivered: segment decomposition, dynamic programming over segments minimising
`SUM ExecutionCost + SUM TransitionCost` subject to capability, privacy and
hardware constraints, and the specification's explain format with the binding
constraint named when a node has no feasible engine. The GPU counter-examples
(per-operation bouncing vs a single boundary; a 5x-faster kernel that still
loses to transfers) are reproduced by the model and asserted in tests.

### 7.4 ~~Engines, interchange, connectors, executor, SDK, CLI~~ — **DONE**

Delivered in this session:

- **`aar/interchange/`** — `Table` over `pyarrow.Table` carrying the canonical
  schema and classification, plus a total Arrow↔canonical type bridge.
- **`aar/engines/`** — Arrow, DuckDB, Polars, pandas, Python UDF worker and
  Excel. One contract, Arrow in and Arrow out, with a recorded fallback.
- **`aar/connectors/excel.py`** — sheet/table/named-range/`A1:P200000`
  addressing, header detection, duplicate-header disambiguation, and error
  cells normalised to nulls.
- **`aar/runtime/executor.py`** — topological execution, per-node engine
  attribution, degradation ledger, and a node-by-node trace on failure.
- **`aar run`** — plans, executes, and prints what actually ran.

Verified end to end by `tools/smoke_run.py`, which generates data, runs a
pipeline, and checks the written files' arithmetic against an independent
Python calculation.

### 7.5 ~~Governance, policy engine, execution history~~ — **DONE**

Delivered in this session:

- **`aar/governance/policy.py`** — the four obligations from §13: egress,
  classification, row-level security and column-level security. Defaults are
  deny; absent rules refuse rather than permit.
- **Masks** — `full`, `hash`, `partial`, `redact` and `email`, chosen so a
  masked column stays *usable*. A numeric column masked to text breaks the
  next aggregate, and a broken pipeline is how masks get removed.
- **Enforcement in the executor** — applied before the write, never after.
  The table returned is the one written, so a caller cannot print a
  "successful" result containing the values the policy just removed.
- **`aar policy`** — `show`, `check` and `--write-example`. An unknown key is
  a hard error, because a silently dropped rule is the one failure this file
  must not have.
- **Execution history** — the executor records every node's observed rows,
  bytes and duration into the cost model's `ExecutionHistory`, failures
  flagged. The next run plans from measurement on this machine rather than
  from a prior.

37 governance tests, adversarial by construction: each one tries to get
confidential data out by a route the implementation might have left open.

### 7.6 Observability — **PARTIAL**

`aar explain`, the per-node run trace and the policy audit cover plan
explanation, engine attribution and rule enforcement. Structured decision
logging and streaming progress are not yet built.

### 7.7 ~~Column-level lineage propagation~~ — **DONE**

- **`aar/lineage/taint.py`** — the rule is one sentence: *a derived column is
  at least as sensitive as everything it was derived from*. The edges are
  where the care went:

| Operation | Inherits | Why |
|---|---|---|
| `SUM(x)`, `AVG(x)`, `MIN(x)` | `x`'s tags | the result is a function of `x` |
| `COUNT(x)` | `x`'s tags | conservative; a non-null count is a property of `x` |
| `COUNT(*)` | nothing | a row count is a property of the table |
| Group key | its own tags | **the key is the value** |
| UDF output | every column's tags | a function is opaque |
| Join | both sides | either input can contribute |

- **`classify()` in the SDK** and a `Tag` node type. Formats cannot carry
  AAR's tags — Parquet has nowhere to put them and an Excel header is a
  string — so this is where an analyst says what a column *is*, and
  everything downstream inherits it.
- **`declassify()`** removes a tag with a written justification stored on the
  column. Without that escape hatch, analysts delete the source tags, which
  is strictly worse.

Verified end to end: a pipeline that tags `salary` as CONFIDENTIAL, sums it
by region, and writes under a masking policy now writes zeros. Before this
change the same pipeline wrote 500, 200 and 300.

The sensitivity ladder moved from `aar/governance` to `aar/types`, because it
describes the tag rather than the rule, and a metadata module importing from
a policy module to ask how sensitive a column is has its layering backwards.

### 7.8 SQL/MongoDB connectors, UI — **NEXT**

Both SQL connectors need a live server to validate against. Then the Analyst
Workbench and structured decision logging.



---

## 8. Assumptions and limitations

1. **Substrait is not the core IR.** Substrait is approaching 1.0 but is not
   frozen, and it has no vocabulary for Excel ranges, Python UDFs, privacy
   classifications or cache boundaries. Adapters will translate in both
   directions; the core IR will not depend on Substrait's shape.
2. **No GPU was available on the build machine.** The GPU code paths
   (detection, transfer-cost model, `ENGINE_ABSENT` degradation) are written
   and exercised, but the GPU *calibration curves* are untested against real
   hardware. This is stated rather than assumed away.
3. **Disk I/O numbers on Windows are warm-cache.** The page cache cannot be
   dropped without elevation, so `sequential_read` understates cold I/O. The
   caveat is stored in the profile and displayed rather than hidden.
4. **SQL/MongoDB connectors need live servers to validate.** Their pushdown
   capability matrices are specified and the `ScanSpec` carries DSN, table and
   pipeline fields, but the connector implementations are not yet written, and
   no integration test can pass until a server exists.
5. **Calibration is machine-specific by design.** A profile is keyed to a
   hardware fingerprint and discarded on mismatch. This is intentional.
6. **Cardinality estimates are declared, not yet measured.** The IR carries
   them; a data profiler that measures real statistics is not yet built. The
   executor does return observed row counts and elapsed times, so the history
   store can be wired to real measurements next.
7. **One aggregate per output column.** The IR stores `tuple[Agg, ...]`, but
   no engine has a representation for a column that is simultaneously a sum
   and a count. The executor refuses that case with a clear message rather
   than silently keeping the first aggregate.

---

## 9. Design principles already enforced in code

Not aspirations — each is a test:

| Principle | How it is enforced |
|---|---|
| Zero external AI dependency | No network calls anywhere; egress is `deny` by construction |
| Empirical, not hardcoded | All cost curves measured live on this machine |
| Never silently fails | `DegradationLedger` + `assert_clean()`; every error tagged with a failure mode |
| Explainable by default | Engine choice, reason, estimate, saving and fallback are `Node` fields |
| Orchestration, not replacement | Zero required dependencies; every engine optional and probed |
| Types travel with data | Classification and lineage survive cast/rename/select |
| Data movement is minimised | Pushdown capability modelled per source; cost model makes transfer explicit |

---

## 10. How to run

```powershell
cd d:\AAR

# The workspace venv already exists at .\.venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Verify everything
.\.venv\Scripts\python.exe -m pytest -q                 # 373 tests
.\.venv\Scripts\python.exe tools\check_syntax.py        # parse every module
.\.venv\Scripts\python.exe tools\check_assets.py        # validate logo.svg
.\.venv\Scripts\python.exe tools\smoke_run.py           # plan + run + verify the numbers
.\.venv\Scripts\python.exe tools\debug_calib.py         # per-benchmark timings

# Run the example pipeline for real
.\.venv\Scripts\python.exe tools\make_sample_data.py    # writes FY26-orders.xlsx
.\.venv\Scripts\python.exe -m aar explain pipelines\example_orders.py
.\.venv\Scripts\python.exe -m aar run    pipelines\example_orders.py
.\.venv\Scripts\python.exe -m aar run    pipelines\example_orders.py --json
```

---

## 11. Summary

**AAR now runs pipelines and enforces policy on them.** Twelve of nineteen
specified layers are complete, tested and verified against real data on real
files. The chain is end to end: the canonical type system, the internal IR,
the hardware profiler with live microbenchmark calibration, the
never-silently-fail failure registry, the capability registry, the cost model,
the adaptive planner, the Arrow interchange layer, the execution engines, the
Excel and file connectors, the topological executor behind `aar run`, and the
policy engine behind `aar policy`.

```
  ScanExcel    excel                  0 ->       240 rows    1467.0 ms
  Filter       arrow                240 ->       233 rows       0.6 ms
  GroupBy      arrow                233 ->         4 rows       1.6 ms
  PythonUDF    python_worker          4 ->         4 rows       3.0 ms
  Sort         arrow                  4 ->         4 rows       0.4 ms
  Limit        arrow                  4 ->         4 rows       0.1 ms
  Write        duckdb                 4 ->         4 rows     943.1 ms

  No degradations. Full-fidelity execution.
```

The system is **measuring rather than assuming** at every level. Cost curves
are fitted from stopwatch runs on this machine; engine availability comes from
a real import probe; and every rejection, fallback and downgrade is recorded
with a sentence explaining it. Note that the UDF ran on `python_worker` and
Excel I/O on `excel` — the plan says what happened, not what was hoped for.

Testing found and fixed around thirty real defects, several of which would
have produced **confidently wrong results rather than visible failures**:

- A DuckDB group-by that fetched its result after unregistering the relation
  and returned an **empty table instead of raising**.
- A DuckDB class that had silently become abstract, so every use degraded to
  another engine while still producing correct numbers.
- A 40x benchmarking error from an un-discarded warm-up, which made the
  system measure Arrow's initialisation cost as the cost of a kernel.
- An Excel writer whose first record overwrote the header row.
- A join that matched `NULL` keys to each other, fabricating rows.
- A column UDF that received only the first column, letting it quietly
  operate on the wrong data.

The specification's central GPU example is **reproduced by the model rather
than asserted in prose**: given an 80 ms GPU kernel against 400 ms of CPU work
and 240+160 ms of transfers, `node_cost` reports a faster kernel and a slower
total, and picks the CPU.

### 3.11 SQL and MongoDB connectors — `src/aar/connectors/sql.py`, `mongo.py`

The interesting property of a data connector is **pushdown**: turning a
scan, a filter and a projection into one statement the database executes
itself. A filter over two of forty columns should make the *database* read
two, not make AAR move forty and filter on arrival.

**The design refuses to approximate.** Both `render_where` (SQL) and
`render_match` (MongoDB) return `None` when a predicate cannot be
translated *exactly*, and the caller then filters after the fetch instead.
This is the central correctness decision in the layer: a SQL clause that
means something slightly different from the predicate returns the wrong rows
and nothing downstream can tell, whereas a slower query is merely slower. An
unsupported function returns `None`, it is not approximated.

SQL's three-valued logic is preserved rather than flattened. `x IS NULL` is
never rewritten as a comparison against a literal, because a comparison
involving NULL is *unknown*, and unknown is not true — rewriting one as the
other silently changes which rows a filter keeps.

**Verification is recorded as data, not prose.** Every dialect and connector
carries a `verified_live` field and a `verification` property that returns
the distinction in words. `SQLITE.verified_live` is `True` because SQLite is
in the standard library and the tests execute real SQL against real database
files. `POSTGRESQL.verified_live` is `False` and `MongoConnector` built on
`mongomock` reports `False`, because a connector that has only had its SQL
generation tested is a different thing from one that has moved a customer's
rows. A test asserts `postgresql_connector(...).verified_live is False` —
**constructing a connector is not evidence that it works.**

The MongoDB connector widens each output column to one type and reports it,
because a document store has no schema and a field can be an Int64 in one
document and a string in the next. Nested objects and arrays are rendered as
JSON text: Arrow has nested types, but a document store's nesting is
free-form, and mapping it to a fixed Arrow struct would either fail or invent
a shape the data does not have.

### 3.13 Five real bugs found by cross-engine parity probing

A probe ran filter, project, sort, limit and group-by on every installed
engine against the same CONFIDENTIAL-tagged input. It found **ten tag-losses
and three further defects**, all of the "correct numbers, wrong privacy
label" variety. Every one is now fixed and pinned by a test.

**1. Ten tag-losses across DuckDB, Polars and pandas.** An engine that
round-trips a table through its own native representation — a DuckDB
relation, a Polars DataFrame, a pandas DataFrame — comes back as a *bare*
Arrow table. `Table(arrow)` derives a schema whose every field is
unclassified. The data is right, the plan is right, and a CONFIDENTIAL
column has silently become public, so a policy trusting classification has
nothing to act on. The fix is a single `interchange.reconcile(result,
source, derived)` applied at every engine boundary, rather than a
correction in each engine: the classification rule now lives in one place,
and a new engine gets it by calling the boundary helper.

**2. `ArrowEngine.sort` could not sort.** It called `table.sort_indices(...)`
on AAR's `Table` wrapper, which has no such method — sorting is a
`pyarrow.compute` kernel — and the IR's booleans also had to be translated
into Arrow's sort-order enum. Every other engine could sort; the
*reference* engine raised `AttributeError` on any non-empty sort. No test
covered it.

**3. `Sink.of()` failed open.** Inference tested membership of
`NETWORK_SINKS` and concluded "not in it, therefore local", so `ftp`, `smb`
and every name AAR had never heard of were classified local and therefore
**permitted** — under a default-deny policy whose own docstring said the
opposite. There was already a test named for this failure, and it passed,
because it only checked sinks on both sides of the divide. The fix inverts
the inference: only a name in a short, explicit `LOCAL_SINKS` is believed
local, everything else is egress. A test now asserts the two sets are
disjoint and that the local one is the short one.

**4. DuckDB returned `Decimal` where Arrow returned `int`.** `SUM(INTEGER)`
is typed `DECIMAL(38,0)` by DuckDB, so a group-by handed back
`Decimal('400')` while Arrow handed back `400`. Both are the right *number*,
which is why it is easy to miss — but a pipeline that returns `int` on one
run and `Decimal` on the next, purely because the cost model chose a
different engine, is exactly the surprise the canonical type system exists to
remove. Fixed at the DuckDB boundary.

**5. Polars joins dropped both sides' tags** — its own join path went
through the unreconciled boundary.

Also recorded, and deliberately **not** "fixed": a group-by emits rows in
first-appearance order on Arrow and hash order on DuckDB and Polars. The set
of groups is the contract; the order is not. Making it deterministic is a
real change with a real cost, so it is pinned by a test that says so rather
than quietly assumed either way.

### 3.14 What the large-data audit found (3,066,766 real rows)

`tools/fetch_data.py` downloads the NYC TLC yellow-taxi Parquet files
(3.07M and 3.63M rows, 47.7 MB and 55.7 MB) and records their SHA-256 and
fetch time in `data/MANIFEST.json`. `tools/audit.py` then runs four
independent sweeps: every engine on every real row, cross-engine agreement,
every CLI command as a subprocess, and the specification's principles. It
writes `data/audit/audit.{txt,json}` and exits non-zero on any failure.

Running against data that is real, large and full of nulls found four more
defects that small fixtures never would have.

**1. `ArrowEngine.group_by` did not finish on 3.07M rows.** It called
`table.arrow.to_pylist()`, materialising all 19 columns as Python objects —
roughly 58 million of them — for a group-by that needs three. Measured: a
group-by over 3.07M rows was still running after **ten minutes**, and had to
be killed. Filter (0.4 s) and sort (2.5 s) on the same data were fine, which
is what makes it look like a hang rather than a slow path.

This matters more than an ordinary performance bug. The Arrow engine is the
*fallback* — it is what runs when DuckDB, Polars and pandas are all absent —
and a fallback that never finishes is not graceful degradation, it is a hang.

Fixed by projecting to the key and aggregate columns before converting to
Python, and by trying Arrow's own `TableGroupBy` kernel first. The test suite
went from about 13 minutes to **55.8 seconds** as a direct consequence.

**2. pyarrow 25.0.1's `TableGroupBy` is broken.** Every documented spelling
fails: the legacy list-of-tuples form (`AttributeError: 'tuple' object has
no attribute 'startswith'`), the dict form (`ValueError: too many values to
unpack`), a bare string key, and every `count`/`min`/`max` spelling. AAR
probes the kernel once on a two-row synthetic table and caches the verdict,
rather than discovering it per call — the probe on a multi-million-row table
does a great deal of work *before* it fails, so probing per call turned a
fast failure into a hang.

**3. DuckDB turned `x IS NOT NULL` into `x <> NULL`, which matches zero
rows.** SQL's three-valued logic makes any comparison with NULL *unknown*,
and unknown is not true. The audit measured **0 rows** against Arrow's
correct non-null count — on the fastest engine, with no error raised
anywhere. A filter that silently returns nothing is the worst failure AAR
can have.

**4. The other engines disagreed with each other on the same predicate.**
pandas kept *every* row for `IS NOT NULL`, because `IS NOT` was evaluated as
`a != b` and an integer column with nulls arrives from pandas as float, so
a null is `NaN` and `NaN != None` is `True`. Arrow's `IS NULL` matched
*nothing*, because `IS` was not handled at all. Three engines, three
different answers, one predicate.

The root cause is the same in both cases: a null test was implemented in
three places — the SQL connector, the DuckDB engine, and the shared Python
predicate evaluator — and only one of them was right. Fixed by making
`PredicateCompiler` the single owner of null semantics (with `_is_null`
treating `NaN` and `None` as the same thing, because that is what the three
frame types mean by "missing"), and by having DuckDB emit real `IS NULL` /
`IS NOT NULL`. `tests/test_metadata.py::TestNullSemanticsAgreeAcrossEngines`
pins all three cases across all four engines.

The two null bugs are the clearest argument for this kind of audit: no unit
test caught them because **no small fixture had a null in it**, and a filter
that returns zero rows raises no error anywhere.

### Measured performance, 500,000 rows, 19 columns

| Operation | arrow | duckdb | polars_cpu | pandas |
|---|---|---|---|---|
| filter | 15.7 s | **0.1 s** | 16.2 s | 20.8 s |
| sort | 0.5 s | 0.3 s | **0.1 s** | 0.3 s |
| group-by | 2.3 s¹ | **0.0 s** | **0.0 s** | 4.9 s |

¹ 300,000 rows, the Arrow engine's Python fallback.

`polars_cpu.filter` being as slow as `arrow` is worth noting: the null
predicate is not expressible in Polars' expression builder, so it takes the
documented Arrow fallback. That is correct behaviour and it is recorded, but
it means a common filter loses 160x. Worth a real fix, not yet done.

### Honest limitations of this audit

- The float difference that this section previously reported as an
  unresolved discrepancy **was a bug in the audit harness, not in AAR.**
  `Agg` is `Agg(func, arg, distinct, custom)` — the third positional
  parameter is `distinct`, not an output alias. The harness passed the alias
  string `"total"` there, which is truthy, so every engine computed
  `SUM(DISTINCT …)` and `COUNT(DISTINCT …)`. The 38x-wrong total and the
  `n=609` were the *correct* answers to that different question. It took
  computing plain-Python ground truth (`tools/diag_groupby.py`) to see it:
  truth is `2038047.96` with `n=34393`, and all four engines now match it
  exactly. The harness is fixed, and a comment at the call site records why
  the third positional is not an alias.

  This is worth recording rather than quietly correcting. I reported a
  "likely float drift, maybe a null-handling bug" hypothesis when the
  actual cause was a mistake in my own test, and the way to tell them apart
  was to compute the answer independently rather than compare engines to
  each other.

- The harness error did expose one **genuine** product bug underneath it:
  `PolarsEngine` rendered `SUM`/`AVG`/`MIN`/`MAX` as plain `col.sum()` etc.
  and consulted `agg.distinct` only for `COUNT`. So `SUM(DISTINCT x)`
  returned the sum *including duplicates* — the right shape of answer to a
  different question, with nothing to signal it. Fixed by taking distinct
  values first, and pinned by
  `TestDistinctIsHonouredEverywhere`, which asserts `SUM(DISTINCT)` is 3
  while `SUM` is 6 on the same rows — a fixture chosen so the two answers
  differ, because a fixture where they coincide tests nothing.

- A residual difference of about **3e-16 relative** remains between DuckDB
  and the other engines on float sums (`2038047.959999998` vs
  `2038047.9600000046`). This is floating-point summation order — a Python
  loop and a vectorised kernel accumulate in a different order — and it is
  benign, but it is real and is not asserted to be bit-identical.
- `data/` is not committed. The 100 MB of Parquet is fetched on demand, and
  the manifest records exactly which bytes were measured.



Two structural checks now live *inside* the test suite rather than in a
separate script, because pytest must not be able to report a green run for
code that does not compile:

- **Every module parses.** A docstring split by an interrupted edit produces a
  module that imports nothing, so every other test in the file fails with a
  confusing error instead of pointing at the one wrong line.
- **No function is truncated to a docstring and imports.** A function that
  lost its body returns `None` where a `Table` was expected. The check is
  deliberately narrow — only *imports* count as a harmless body — because
  `_RUN_CACHE.clear()` is a legitimate no-return function and a check that
  merely looked for "no return statement" would fire on every well-written
  mutator in the codebase.

Both checks found real damage during this build: an orphaned `sqlite_type_name`
body, a duplicated `projection_sql`/`explain_pushdown` pair, a duplicated
`read()` tail, and a duplicated docstring terminator that had been swallowing
roughly 140 lines of the SQL connector as string content.


- **No GPU on this machine.** Detection, the transfer-cost model and
  `ENGINE_ABSENT` degradation are written and exercised; the GPU *calibration
  curves* are unverified against real hardware.
- **SQL and MongoDB are verified at different levels, and the code says which.**
  SQLite is in the standard library, so the SQL connector is tested against a
  **real database on real files** — a wrong pushdown returns wrong rows and
  fails the suite. MongoDB is tested against `mongomock`, which implements the
  query, projection and aggregation semantics in Python: that genuinely
  verifies AAR's translation, pushdown decisions and type mapping, and
  verifies nothing about BSON encoding, the wire protocol, indexes or the
  server's optimiser. PostgreSQL and MySQL have tested dialect and SQL
  generation and **no** live verification. `SqlConnector.verification` and
  `MongoConnector.verification` return this distinction as a string rather
  than leaving it to a reader of the comments, and tests pin it.
- **Windows disk figures are warm-cache** and labelled as such in the profile.
- **One aggregate per output column.** The IR allows several; no engine has a
  representation for a column that is simultaneously a sum and a count, so the
  executor refuses it with a clear message rather than silently keeping the
  first.
- **Column-level lineage now propagates.** An aggregate, UDF or join output is
  at least as sensitive as its inputs, so `SUM(salary)` over a CONFIDENTIAL
  column is itself CONFIDENTIAL and a policy can act on it. Two residual gaps
  remain, both stated rather than assumed away: a `declassify()` call is the
  only way to shed a tag, and it requires a written justification that is
  *stored* but not yet surfaced in the run trace.

Next is **structured decision logging** and the **Analyst Workbench**, then
live-server integration tests for PostgreSQL and MongoDB once servers are
available.


