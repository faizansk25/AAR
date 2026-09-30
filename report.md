# AAR — Build Progress Report

**Project:** Adaptive Analytics Runtime
**Workspace:** `d:\AAR`
**Repository:** https://github.com/faizansk25/AAR.git (branch `main`)
**Specification:** `system.md`
**Status:** AAR runs pipelines, propagates privacy, enforces it, and has a Workbench · 737 tests
**Last updated:** 2026-09-30

---

### Round 14 — audit follow-up: embedded newlines, and a branch for review

The second audit accepted the Round 13 findings and asked two things: verify
the profiler's physical resource bounds, and check whether the head+midpoint
sampling assumption holds. It also recommended pushing to a branch so the
work could be verified independently.

**Branch `audit/profiler-and-window-fixes` is pushed**, two commits ahead of
`main` (`c61ff98`):

- `8207264` — bounded profiler, RANK/DENSE_RANK, classification propagation,
  profile-before-planning, distinct estimator, Parquet row groups, SQLite.
- `6947e30` — embedded newlines in CSV (below).

`main` is deliberately untouched so the reviewer can diff cleanly.

#### The embedded-newline check found a crash, not just a bias

The audit raised a specific technical point: counting newlines to measure
record width assumes one newline per record, which is false for quoted
fields. That is correct, and the consequences were worse than a bad estimate:

1. **The reader raised `ArrowInvalid`** — "CSV parser got out of sync with
   chunker" — from inside the C++ reader, because `profile_csv` opened the
   file with default options. A multi-line text export is an ordinary file.
   The pipeline survived only because `profile_node` catches broadly;
   nothing *guaranteed* that, and profiling must never be what fails a run
   over a file that reads fine. Fixed with
   `ParseOptions(newlines_in_values=True)` — on `ParseOptions`, not
   `ConvertOptions`, which is an easy place to guess wrong.

2. **The row estimate doubled.** `_per_row_of` divided a block's length by
   its newline count. On a 1,000-row file with a newline in every row:
   18,898 bytes over 2,000 newlines = 9.4 bytes/row, so the file was
   reported as **2,000 rows instead of 1,000**. Record boundaries are now
   counted by tracking quote state, with `""` treated as an escaped quote
   that does not end the run. That file now estimates **1,001 against a true
   1,000 — 0.1% error**.

#### On the two sampling objections

Both are accepted as real limitations, and neither is papered over:

- *Head+midpoint is not universally representative.* True — a file whose
  record width jumps in its final third is still mis-estimated. This is a
  property of two-point sampling, not a bug to fix here; the honest
  mitigation is a larger sample budget, and the profile already reports
  `exact_rows=False` so a plan never presents the figure as a measurement.
- *Retained rows vs physical bytes.* The profiler now bounds both
  independently (`sample_rows` and `max_bytes`), and `_ByteCappedFile`
  records `bytes_read` so the real cost is observable rather than assumed.
  A single cardinality estimate is still used everywhere; splitting it into
  expected/lower/upper bounds is a real improvement and is *not* done yet.

**Full suite: 737 passed, 7 skipped, 0 failed. ruff clean.** The NYC taxi
Parquet profile is unchanged at 3,627,882 rows / 55,682,369 bytes.

#### Still open, in the order the audit recommends

Transfer accounting and memory feasibility **before** the optimizer — a
better search over wrong cost estimates still picks the wrong plan. The
20,000-combination ceiling, scheduler resource controls, the Workbench
security boundary and persistent estimation history remain untouched.

---

### Round 13 — second external audit, verified against the source

An audit of commit `c61ff98` raised four P0 findings plus a dozen P1s. Each
was checked before acting. The P0s were all real, and the first one turned
out to be worse than described — the profiler did not merely fail to bound
its read, it read the entire file.

| Claim | Verdict | What was actually wrong |
|---|---|---|
| CLI profiles *after* planning | **Real** | `_plan_for` planned from declared sizes, then measured and discarded; skipped entirely under `--json` |
| Profiler can read a whole dataset | **Real, worse than stated** | `read_all()` ignores `block_size`: measured **200,000 of 200,000 rows** on a 200k-row file |
| `RANK` counts equal, not smaller, values | **Real** | `10,20,20,30` gave `1,1,2,1`; DuckDB gives `1,2,2,4` |
| Derived columns lose classification | **Real** | `_rebuild` made a bare `Field`; a running total of confidential salaries came out public |
| Distinct estimator ignores its own estimate | **Real** | `estimator.exact` is frozen at `EXACT_LIMIT` past 4096 values |
| Parquet nulls not summed across row groups | **Real** | last group overwrote the rest; a half-empty column reported 0% |
| SQLite stats look for `path` | **Real** | the SDK sets `connection`, so the exact path never matched |
| Multi-key `ORDER BY` sorts wrong | **Real** (found by me) | sequential sorts made the *last* key primary |
| Descending `RANK` returns `1,1,1` | **Real** (found by me) | direction lived in `reverse=`, invisible to the key comparison |

#### The bounded read, measured rather than assumed

`block_size` is buffer granularity, not a limit — `read_all()` reads
everything. Nor is there a row cap: `batch_rows` does not exist on
`ReadOptions` in pyarrow 25, and `block_size` only scales batches down to a
floor of ~840 rows. The cap is therefore enforced **below** the reader, by a
stream that physically reports EOF (`_ByteCappedFile`, an `io.RawIOBase`).
Measured on a 200,000-row file with a 500-row budget: **1,550 rows read**,
versus 200,000 before.

Two consequences found while fixing it:

* The retained sample is exact, but Arrow emits whole batches, so the
  *reader* over-delivers by up to one batch. Asserting an exact 500 was not
  physically achievable; the test now asserts the bound that matters.
* Bounding the read exposed a **pre-existing** bias: bytes-per-row was
  measured from the head block alone, which on a file whose rows widen
  downward gave 5.99 against a true 7.63 — a **27% over-estimate of row
  count**, which the plan divides straight into. Now sampled at head *and*
  midpoint.

#### Window functions, checked against DuckDB

Expected values were read out of DuckDB rather than reasoned about. Two
further bugs surfaced only because the tests were written that way:
descending `RANK` returned `1,1,1` (direction was in `reverse=`, which the
key comparison never saw), and `ORDER BY a, b` sorted by `b, a` — the exact
inversion the audit predicted. `DENSE_RANK` was also silently returning
`RANK`, since it counted rows rather than distinct keys.

#### Classification propagation

`_rebuild` now inherits from the input column *and* the partition/ordering
keys — a rank encodes a row's position relative to a sensitive key. Worth
recording: **the engines already propagated correctly**; only the window
path dropped the tag. The old comment ("guessing is worse than recording
none") inverted the risk — under-recording is what leaks. A test now covers
the engine path too, so a future break there is caught rather than assumed.

#### Not fixed, and why

The **20,000-combination search ceiling** (P0 #3) is confirmed but untouched:
it needs a real optimizer, not another threshold. Likewise scheduler
resource controls, persistent estimation history, and the Workbench security
boundary. Those are recorded as open rather than half-done.

**Full suite: 730 passed, 7 skipped.** The `695`-vs-`692` discrepancy in
earlier notes is resolved — the README guard now pins the real number.

---

### Round 12 — external audit findings, verified and fixed

An external audit of commit `d1c748e` claimed nine specific defects. **Every
one was checked against the source before acting on it**, because an audit
that is wrong about `run()` vs `execute()` would send someone rewriting
working code. All nine were real; two of them turned out to hide further
bugs that only surfaced once the first was fixed.

| Claim | Verdict | What was actually wrong |
|---|---|---|
| Workbench calls nonexistent `Executor().run()` | **Real** | `Executor` has only `execute(plan)`; `/api/run` failed for every valid pipeline |
| `dedup` `last` keeps the wrong rows | **Real** | `keep[-1] = i` overwrote the last row appended, not that key's row |
| Windows compute partition aggregates, not frames | **Real** | `_apply_aggregate(fn, partition)` per row; a running total returned a constant |
| Unknown quality rules silently ignored | **Real** | no `else` branch; a typo reported a passing check that never ran |
| `SCAN_SQL`/`SCAN_MONGO` never dispatched | **Real** | fell through to `NotImplementedError` |
| Calibration fits `x²` but stores `x^exp` | **Real** | the search returned a model it had never fitted |
| `serialise_s` units | **Real** | documented as seconds, multiplied by bytes: 1 MB cost 20 s |
| Transfers charged twice | **Real** | `node_cost` added the hop and the planner added it again |
| Greedy, not dynamic programming | **Real** | confirmed by reading `plan()`; **not yet fixed** |

#### Two more bugs the fixes exposed

Fixing the Workbench endpoint revealed that `f.type.render()` did not exist
on `DataType` — a second `AttributeError` on the same line, one call deeper.
The existing tests could not have caught either: they asserted that a
*missing file* produced an error, which fails earlier and for a different
reason. **A test suite can be green while the product's main function is
dead.** The audit's request for a successful end-to-end test is now met by
four of them.

Fixing the transfer double-count exposed the third bug: `SegmentPlan.total_s`
is `cost.total_s + inbound_s`, so removing the second addition was not
sufficient — `inbound_s` also had to be reduced. Two rounds of "fix the
double count" were both wrong until the invariant was written down as a test
rather than reasoned about in prose.

#### Tests that would have caught these

`tests/test_correctness_regressions.py` collects them, each naming the defect
it reproduces. The pattern they share is worth stating plainly: **every one
of these operations returned a plausible wrong answer instead of an error.**
A dedup that drops a row, a running total that is constant, a quality check
that passes without running, a trace that credits DuckDB with work Python
did. None of them would fail loudly, which is why a green suite proved
nothing about them.

---

## 1. Vision

Give an analyst one execution environment for Excel, SQL, NoSQL, Python, local
files and large-scale data, while automatically selecting the most efficient
**legal** execution path for the available hardware — with full explainability
and zero external AI dependency. The goal is **minimum total analytical cost**,
not GPU utilisation.

---

Also fixed on the way: the calibration store now loads the machine's saved
profile on the default path, and `engine_used` names `executor` for the nine
node types the executor computes in Python rather than crediting the assigned
engine.

**Round 13: the planner stopped being greedy.** The last audit finding left
open, and the one with a clean completion test. `plan()` walked the segments
once and committed to the cheapest engine for each, which is not dynamic
programming however the documentation describes it.

The test came first. `TestPlanningIsGloballyOptimal` builds three segments,
two real engines, and a 30 s switch, then enumerates all eight engine
assignments and asserts the planner chose the cheapest. Against the greedy
planner it failed with a concrete answer:

    planner chose ('arrow', 'cudf', 'cudf') costing 50.0s;
    the optimal path costs 22.0s

That number is the whole argument for writing the test first. A DP written
without a failing counterexample would have produced a test asserting "a
table was used", which passes whether or not the plan is optimal.

The table is `best[i][engine]` = cheapest cumulative cost of covering
segments 0..i ending on that engine, with `came_from` reconstructing the
path. One old assertion had to change, and it is worth naming: a test
previously asserted the chosen engine was the cheapest candidate *for its own
segment*. That encoded the greedy rule, and it is wrong by design — a segment
may pay more now to avoid a crossing later. The surrounding invariant (the
chosen engine was genuinely considered; the segment is never cheaper than its
own work) still holds and is still tested.

What the DP does **not** solve: its state is "the engine of the previous
segment", which describes a chain. A join gives a segment several
predecessors. That needs a state of *sets* of engines, and it is not written.

---

### Rounds 10-11 — verification, retraction, and ergonomics

**Round 10: the Excel write path, and a retraction.** `connectors/excel.py`
went from `ws.cell()` per cell to `ws.append()` per row, and from row-major
`to_pylist()` to column-major `to_pydict()`. Then it was measured, and the
measurement disproved the premise (`tools/probe_excel_phases.py`, 50,000
rows x 12 columns):

| form | write ms | save ms | total ms |
|---|---|---|---|
| per-cell | 10,382 | 40,782 | 51,165 |
| append | 7,398 | 40,502 | 47,900 |
| write_only | 47,490 | 2,443 | 49,933 |

`wb.save()` dominates, and openpyxl stores a `Cell` object per cell whichever
API writes it, so the save has identical work to do. **The speedup is ~6%,
not the 1000x the first draft of the code comment claimed** — that comment
was corrected rather than left to flatter the diff. This is the second
retraction in the programme; the first was the "213x group-by defect" that
turned out to be 97% benchmark contamination. Both were caught the same
way: by measuring instead of assuming.

Also found on the way: `_sheet_has_content` asked "is this fresh sheet
empty?" by calling `ws.cell(row=1, column=1)`, and in openpyxl that call
*materialises* cell A1, advancing the append cursor and pushing the header
to row 2. Every written file gained a blank first row and the reader
mis-inferred every type after it. **A probe that is not a read.**

**Round 11: structural analysis with `pydeps` and `pyan3`**, both installed
as dev-only tools (neither is a runtime dependency; `verify_release.py`
still reports zero third-party dependencies for the installed wheel).

Import cycles: **155 reported, 36 distinct module sets, 3 root shapes.** The
155 is a rotation count, not a problem count — pyan3 emits every rotation
of every cycle, so a 6-module cycle counts six times.

| shape | closing edge |
|---|---|
| `cli` -> `workbench.server` -> `cli` | `server.api_explain` does `from ..cli import _plan_for` |
| `workbench.__init__` <-> `workbench.server` | `__init__` re-exports `server`; `server` does `from .. import __version__` |
| `engines.factory` <-> every engine | each factory closure does a deferred `from .<engine> import <Engine>` |
| `connectors.*` <-> `engines.*` | `arrow_engine.write` does `from ..connectors.excel import write_excel` |

**Every back-edge is a function-local import, and that is the point.** It is
what keeps `import aar` free of pyarrow, duckdb, polars, pandas, cudf and
openpyxl — an engine's third-party import happens when the engine is
*constructed*, which is exactly when the dependency is wanted. So the
cycles are the price of lazy loading, benign at runtime, and the count is
identical with and without pyan3's `--init` flag (so not an artifact of
that flag's implicit package edges). The real risk is not import failure
but that a cycle makes a *conceptual* boundary negotiable, so
`tools/cycle_roots.py` is checked in and regenerates `ARCHITECTURE.md` with
the distinct count (36) rather than the rotation count (155), making growth
visible and deliberate.

pydeps found **no cycles reachable from `aar.__init__` alone** — the path a
plain `import aar` takes. External dependencies in the graph are exactly
the declared optional extras; nothing unexpected, nothing unguarded at
module scope.

**The ergonomics fix this review produced.** `run` and `explain` both
demand a path to a pipeline file, and nothing told a user what one looks
like or gave them one. `pipelines/example_orders.py` existed, but that path
is outside the package, so it is absent from an installed wheel — on an
air-gapped machine there is no repository to look in. Added
`src/aar/examples.py` and `aar examples` (list / `--show` / `--write`),
with templates embedded as strings so there is no `package-data` entry to
forget at build time. `--write` refuses to overwrite.

The examples are **tested as code**: each is parsed with `ast`, every
`aar.sdk` import is checked against the **live `__all__`** (so renaming an
SDK function fails a test rather than a user), no example may import a
third-party module, and the shortest is written to disk and run through
`aar explain` for real. That last test caught the first draft, which used a
chainable style against a functional SDK and `count_` where the export is
`count` — both would have failed on a user's very first run.

### Round 12 — dead code analysis (ruff + vulture), and three real bugs

Installed `ruff` and `vulture` as dev-only tools. Starting state:
**117 ruff findings** (91 unused-import, 11 redefined-while-unused,
9 undefined-name, 6 unused-variable) and **4 vulture findings at 100%
confidence**. All cleared; both tools now report zero.

#### The one that mattered: a NameError behind a guard

`ruff` F821 flagged `Schema` and `Field` used in
`engines/arrow_engine.py::_arrow_group_by` **without being imported at module
scope**. The line sits *outside* the `try/except` that ends a few lines
above, so it was not covered by the decline-on-failure handler.

It never fired because `_group_by_kernel_works()` returns `False` on most
Arrow builds and short-circuits first. So this was a landmine that only
detonates on a build where the native group-by kernel actually works — and
on a build where it works, the engine raised `NameError` instead of
returning a table. The "fast path" was both unreachable here and broken
where reachable. Confirmed with `tools/probe_arrow_groupby.py`, which prints
the module namespace, the guard's answer, and the result of calling the
function with the guard forced on.

**607 of 609 tests were green while this was in the tree.** The suite tests
behaviour; it cannot see a name that resolves or not on a build that never
reaches the line.

#### Two truncated functions

`vulture` found unreachable code, and all four instances had the same
signature as the earlier `return openpyxl` in `connectors/excel.py` — a
function whose body was cut short with its old tail left behind:

| location | dead code |
|---|---|
| `engines/base.py` | `raise NotImplementedError(...)` after a `return` in `PredicateCompiler.compile` |
| `governance/policy.py` | a second `return level` after the first |
| `ir/nodes.py` | `return node.type in _SINK_NODES` after `return out` |
| **`engines/polars_gpu_engine.py`** | **`write()` computed the host table, imported `ArrowEngine`, then ended** |

The fourth is a functional bug, not tidiness. `PolarsGPUEngine.write()`
returned `None` where the contract promises an `int`, and **wrote nothing
at all** — a silent no-op on the one engine whose entire purpose is speed.
The two surviving lines made the intent obvious and matched
`CudfEngine.write`, so it was completed the same way: collect off the device
once, then delegate the write to Arrow.

#### Undefined names in annotations

`Any` in `capability/registry.py`, `Iterable` and `Callable` in
`hardware/calibrate.py`, `Mapping` in `interchange/table.py` and
`sdk/pipeline.py` — all used in annotations, all absent from the imports.
Harmless at runtime only because `from __future__ import annotations`
defers evaluation; `typing.get_type_hints()` and any type checker would
have failed on them.

#### Two tests that could not fail

The most embarrassing finding, and `ruff` F841 is what exposed it.
`test_render_shows_every_term` called `CostBreakdown.render()` and
discarded the string. `test_a_join_merges_both_sides` built a tagged table
and asserted nothing, despite a docstring claiming it verified that a join
inherits both sides' tags — which is a **privacy** property. Both now assert
properly: the first checks every cost term appears and that the total
reconciles; the second performs a real join and checks the joined column
inherits the CONFIDENTIAL tag from both inputs.

Writing the assertions caught two of my own errors immediately — I guessed
`inbound_s` (the field is `read_s`/`spill_s`/`materialise_s`) and computed
the join cardinality as 3 when it is 5 (NA×NA twice, plus EU). Both would
have been wrong assertions, which is the argument for making these tests
real rather than decorative.

#### Now enforced

`tests/test_cli.py::TestNoSilentLint` shells out to `ruff` and `vulture`
and fails the suite on any finding, skipping cleanly when the tools are not
installed. This is the third gate alongside `pytest` and
`verify_release.py` — and it is the first one that can catch a defect in
code that *no execution path on this machine reaches*.

### Round 13 — logic-bug lint, and a linter-induced regression

Widened `ruff` from the five pyflakes rules to the full logic-bug set
(`B` bugbear, `RET`, `SIM`, `PLW`, `C4`, `PIE`). Starting state: **381
findings**. Ending state: **0**, with every exclusion carrying a written
reason in `pyproject.toml` rather than a bare ignore code.

#### The real defects

| finding | where | what it was |
|---|---|---|
| `F822` undefined export | `failures/registry.py` | `__all__` listed `MODES`, which does not exist (the name is private `_MODES`; the public accessor is `register_modes()`). `from aar.failures.registry import *` would have raised `AttributeError` |
| `PLW1641` eq without hash | `lineage/taint.py` | `LineageEvent` defines `__eq__` and not `__hash__`, which Python turns into `__hash__ = None` — the class is unhashable. Nothing hashes it today, but it is a value object by design and `Col` defines `__hash__` for the same reason |
| `SIM115` unclosed file | `tests/test_hardware.py` | `json.loads(open(path).read())` leaked a handle; on Windows an unclosed handle keeps the file locked, which is a latent flake |
| `F402` shadowed import | `policy.py`, `executor.py` | a loop variable named `field` shadowed `dataclasses.field` for the rest of the function |
| `PLR0124` self-comparison | `engines/base.py` | the NaN test `value != value`, which is correct and cryptic; now `math.isnan(value)` |
| `C416`, `PLW2901`, `SIM108`, `SIM118` | 4 sites | mechanical |

#### A regression the linter caused, and the fix for the gate

`PLW1510` ("`subprocess.run` without explicit `check`") is a **false
positive** here: `check=False` was already explicit, inside the `**kwargs`
dict, which ruff cannot see through. "Fixing" it by adding `check=False` at
the call site produced **`TypeError: subprocess.run() got multiple values for
keyword argument 'check'`** and broke five tests in `TestProbes`.

That is worth recording twice over. First, the honest reason to keep a
linter is that it finds things humans miss — and the reason to keep a
human in the loop is that a confident, well-formatted finding can still be
wrong, and acting on it blindly breaks working code. `PLW1510` is now
globally ignored with that reason written down, and `detect.py` carries a
comment at the call site so the next person does not re-add it.

Second, the lint gate itself had a hole: it asserted `returncode in (0, 1)`
and then `== 0`, but reported a malformed `pyproject.toml` — a genuine tool
failure — indistinguishably from findings. It now distinguishes the two and
names the cause. Adding `[tool.ruff]` also exposed a **pre-existing config
bug**: `addopts` was a space-separated string where the schema requires a
list, so `ruff` could not parse the file at all. Fixed to a proper array.

#### The 20 `B023` late-binding findings, and why they are excluded

All 20 are the microbenchmark closures in `hardware/calibrate.py`. They are
**false positives**: `measure()` invokes each closure via `_time(fn)` inside
the same loop iteration, so late binding cannot occur.

But "false positive today" is not the same as "cannot go wrong", and the
failure mode if a refactor ever deferred those calls is nasty: every
calibration point would report the *last* input size, and the resulting cost
curves would still look entirely plausible — the worst way for a benchmark
to fail, and precisely the class of defect this project already retracted
once. So rather than rewrite 20 lambdas or ignore the rule blindly:

- `calibrate.py` is exempted from `B023` with the reason written down, and
- **`test_each_calibration_point_records_its_own_size`** was added as the
  real guard. It asserts more than one distinct input size appears across
  the points. Late binding would collapse that to one and fail.

The exclusion comment names that test, so the exemption cannot outlive the
guard it depends on.

#### What was excluded, and why

Nine rule families are globally ignored with a reason each, not left as
silent suppressions: `TID252` (relative imports keep the package
relocatable and preserve the `aar` import-name / `aar-analytics` distribution
split), `PLW0108`, `SIM105`, `SIM117`, `B017`, `PLW0603` (module-level
memoisation is why the hardware probes are fast), `PLW1510`, `B007`, `B905`,
`RET503`, `RET504`, `PIE810`, `SIM108`, `SIM118`. `PLR*` and `ARG*` are
absent entirely — they report complexity and argument counts, which this
codebase documents in prose instead.

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
| 17 | SDK (`ctx.excel()` …) | §1 | ✅ **Complete** | SDK tests + end to end |
| 18 | CLI — `aar explain plan` | §16 | ✅ **Complete** | `aar explain` |
| 19 | Analyst Workbench UI | §17 | 🟡 **Partial — live, translatable, a11y** | 30 workbench tests |

**Progress: 9 of 19 layers complete (47% of the specified system), plus
3 partial.** The completed layers form the chain from "what does this data
mean" through "what can this machine do" to "what will it cost" to "let me
see and change it". Each was built to be independently useful and testable
before its consumer existed.

**What still does not work, stated plainly:** an analyst can point AAR at a
file and get an answer — `aar run` does exactly that, and the release gate
executes a pipeline from the *installed wheel*. What is missing is the harder
half of the specification: **GPU and distributed execution at scale, the
scheduler's real resource management (thread counts, memory ceilings, worker
concurrency), a data profiler that measures real statistics instead of
trusting declared cardinalities, and complete Workbench panels.** The
connector set is narrow for the same reason: Excel, Parquet, CSV and SQLite
are real; PostgreSQL, MySQL and MongoDB have dialects and pushdown logic but
have never spoken to a live server.


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

### 3.8 Data profiler — `src/aar/stats/__init__.py`

The gap this closes: the cost model sized every segment from
`Node.estimated_bytes` — a number the pipeline *declared* — or a flat 1 MB
guess. The dynamic program added in round 13 was therefore reasoning from
numbers nobody had checked.

- **Prefers declared metadata over sampling.** A Parquet footer gives the row
  count, per-column distinct counts, null counts and compressed chunk sizes
  without reading a single value. SQLite's `sqlite_stat1` does the same after
  `ANALYZE`. Both are exact and free; reading 10 GB to plan a query is
  neither.
- **Bounded sampling where metadata is absent.** A CSV carries nothing, so
  the first N rows are read and the row count extrapolated from bytes-per-row
  *measured from the file's own text*. Distinct counts come from a
  HyperLogLog sketch with a 4096-value exact set, because an exact set over
  50M distinct strings is the memory problem the component exists to avoid.
- **Every profile names its provenance** — `parquet-metadata`,
  `database-stats`, `measured`, or `sampled` — and `exact_rows` says whether
  the count is a fact or an extrapolation. A profile that cannot say which is
  a guess wearing a measurement's clothes.
- **An unreadable source returns `None`, not zero rows.** Zero would plan a
  trivially cheap pipeline, which is the one answer a missing measurement must
  never produce.
- **`aar profile <file>` and `aar run`** both use it. `PipelineService` profiles
  every source before planning, so the estimate is measured by default rather
  than opt-in.

Verified on real data, not only in tests:

```
$ aar profile data\nyc_taxi_2022_03.parquet
data\nyc_taxi_2022_03.parquet: 3,627,882 rows, 55,682,369 bytes (15 B/row, parquet-metadata)
  tpep_pickup_datetime, ~3.7 chars
  trip_distance, ~1.4 chars
  passenger_count, 3% null, ~0.2 chars
```

**Not yet done:** measured *estimation error*. The profiler records what it
believed, but nothing yet compares that against the row count the executor
actually observed, so a systematically wrong profiler would not be caught by
the system itself. The executor already records real output row counts, so
the comparison is a matter of wiring rather than invention.

**Now closed (round 14).** `EstimationLog` records the gap between each
profiled source's predicted rows and the rows the scan actually produced, and
`aar run` prints it by default. A deliberately wrong profiler — one claiming
ten times the rows that exist — is caught by
`test_a_wrong_profiler_is_caught`, which is the property that matters: the
error is invisible in the profile itself and only appears against reality.
Verified live:

```
ESTIMATION ACCURACY

  ScanParquet:op_b16d0ba004fc: predicted 50,000 (parquet-metadata),
  actual 50,000 -> exact by 0%

  mean error 0.0%, worst 0.0%, 0 of 1 outside 50%
```

Still per-run rather than accumulated: errors are labelled by node id, which
changes every build. A stable *semantic* identifier for a source — path plus
size plus mtime, say — would be needed to carry error across runs, and
inventing one that could silently collide would be worse than keeping this
honestly scoped to a single run.

### 3.9 Segment costing — `CostModel.segment_cost`

The planner priced every segment from `seg.nodes[-1]`, so a
`Filter -> GroupBy -> Sort` segment was estimated from the sort alone. A
filter discarding 99% of rows and a group-by reducing to a thousand groups
both cost real time, and neither appeared in the number the engine was
chosen with.

- **A sum of per-operation costs, not a fused estimate** — and that is a
  claim about the runtime, not about optimising potential. The executor
  dispatches one node at a time (`_dispatch` calls `engine.filter`, then
  `engine.group_by`, then `engine.sort`), so each operation really does pay
  its own kernel time. Pricing a segment as one optimised query would credit
  a fusion that never happens — the same error as reporting a GPU that never
  ran. If fusion is ever implemented, the estimate must change with it.
- **Each operation is priced at the size it actually sees.** A group-by
  reducing a gigabyte to a thousand rows makes the sort after it cheap;
  billing both the same gigabyte over-prices the tail and mis-ranks engines
  on it. Measured: 500k, 400k and 300k bytes cost 825ms, 690ms and 555ms.
- **Startup, read, spill, materialisation and the boundary crossing are
  charged once per segment**, not per node. The engine is built once and the
  data crosses the bus once; only `compute_s` repeats.
- **Provenance survives the sum.** A segment cost partly from history and
  partly from priors reports as mixed, so a plan cannot quietly launder a
  guess through an average.

---

## 4. Testing

**695 tests: 695 passing, 8 skipped, 0 failing.** Every skip states the
missing dependency rather than passing vacuously. `README.md` states the same
number, and `test_the_readme_test_count_is_the_real_one` fails the suite if
the two ever disagree again — a stale count is the cheapest way to lose a
reader's trust, and this project shipped one for months.

| Suite | Tests | Coverage |
|---|---|---|
| `test_types.py` | 42 | Source mapping, loss detection, coercion, schema algebra, drift, conversion log |
| `test_capability.py` | 38 | Catalogue integrity, declared vs available, feasible sets, rejection reasons, memory fitness, ledger integration |
| `test_cost.py` | 42 | Cost breakdown, three cost sources, **the spec's GPU example**, transitions, memory, history, explain |
| `test_hardware.py` | 32 | Byte helpers, all 6 probes, profile facade, curve fitting, store round-trip, stale-profile rejection, **live calibration** |
| `test_failures.py` | 20 | Full matrix coverage, error hierarchy, ledger semantics |
| `test_ir.py` | 20 | Expressions incl. SQL-injection safety, node metadata, DAG ordering, cycle detection |
| `test_data_profiler.py` | 24 | **Row counts, cardinality, nulls, widths, provenance, sampling bounds** |
| `test_planner.py` | 28 | Segment decomposition, **global optimality vs brute force**, transitions charged exactly once, feasibility |
| `test_correctness_regressions.py` | 16 | **Defects that shipped with the suite green** — dedup, window frames, quality rules, trace integrity |
| `test_cli.py` | 37 | `doctor` / `engines` / `calibrate` / `explain` / `run` / `policy`, exit codes |
| `test_runtime.py` | 153 | **Cross-engine agreement, interchange, connectors, executor, end to end** |
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
| `ws.cell(row=1, column=1)` used to ask whether a fresh sheet was empty | In openpyxl that call *materialises* cell A1, advancing the append cursor so the header landed on row 2 with a blank first row; the reader then took the wrong header and mis-inferred every type after it | Never probe a sheet this code just created |
| `to_pylist()` + `ws.cell()` per cell in the Excel writer | Row-major materialisation of every row as a dict, and one general-purpose call per cell | `to_pydict()` + `ws.append()` per row |


### The Excel optimisation, and a retraction

Round 10 rewrote the Excel writer and then measured it, and the measurement
disproved the premise. At 50,000 rows x 12 columns
(`tools/probe_excel_phases.py`):

| form | write ms | save ms | total ms |
|---|---|---|---|
| per-cell | 10,382 | 40,782 | 51,165 |
| append | 7,398 | 40,502 | 47,900 |
| write_only | 47,490 | 2,443 | 49,933 |

`wb.save()` dominates. openpyxl stores a `Cell` object per cell in
`ws._cells` whichever API writes it, so the save has identical work to do
either way, and `ws.cell()` was never the bottleneck. The rewrite is a real
~6% improvement and a large cut in call count — **it is not the speedup the
first draft of the code comment claimed, and that comment was corrected
rather than left to flatter the diff.** `write_only=True` does not rescue
the time either; it relocates the 40s and its real benefit is memory.

This is the second retraction in the programme. The first was the "213x
Arrow group-by defect", which was 97% benchmark contamination. Both were
caught the same way: by measuring instead of assuming.


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

### 7.8 ~~Analyst Workbench~~ — **DONE** (`aar/workbench/`)

A local UI over the real system, built on the standard library only — no CDN,
no frontend build chain, binding to loopback by default.

- **Live probe, not declarations.** Engine availability comes from
  `registry.probe()`, and the UI shows the *reason* verbatim, so an analyst
  who sees "duckdb is not installed" can act on it.
- **Runtime i18n** across all declared languages, RTL layout, light/dark
  themes, density settings, and keyboard shortcuts.
- **A real data grid.** Paging and sorting are executed *in the engine*, not
  in the browser. Shipping 3M rows to a tab for JavaScript to reorder makes
  the workbench the slowest part of a fast pipeline.
- **Privacy stays visible.** Columns carrying confidential or restricted
  classification are marked in the grid, so the analyst sees what the policy
  engine sees.

### 7.9 ~~PyPI packaging~~ — **DONE**

`adaptive-analytics-runtime` builds to a clean wheel and sdist.

Two defects were fixed, both of which only appear *after* publishing:

- The Workbench's `static/` assets were not packaged. The wheel installed
  perfectly and then `aar workbench` 404'd on every asset.
- The `all` extra self-referenced `aar[...]`, but the distribution is named
  `adaptive-analytics-runtime`, so it could never resolve.

Also: the `all` extra no longer pulls `cudf`, because installing RAPIDS on a
CPU-only host *fails at install time* rather than degrading, which would
break `pip install adaptive-analytics-runtime[all]` for most users. GPU is
explicitly opt-in.

**A missing `LICENSE` was found and added.** The package declared Apache-2.0
but shipped no licence text.

**The PyPI name is `adaptive-analytics-runtime`, not `aar`.** Confirmed free
at the time of writing. The bare name `aar` is **taken** on PyPI by an
unrelated project (an AI application library, since 2024), so anyone typing
`pip install aar` gets that instead. The import name and the console script
are both still `aar`; only the distribution name is spelled out. Every
install instruction in the repository says so explicitly for that reason.

### 7.10 GPU verification — **IN PROGRESS**

No GPU measurement exists. The attempt, and its limits, are documented in
§8.2. `tools/gpu_verification.py` and
`notebooks/aar_gpu_verification.ipynb` are ready and have been run on a
CPU-only host, where they complete and report honestly.

### 7.11 SQL/MongoDB live tests, decision logs — **BLOCKED**

Both SQL connectors need a live server. Structured decision logging and
streaming progress are not yet built.

---

## 8. Assumptions and limitations

1. **Substrait is not the core IR.** Substrait is approaching 1.0 but is not
   frozen, and it has no vocabulary for Excel ranges, Python UDFs, privacy
   classifications or cache boundaries. Adapters will translate in both
   directions; the core IR will not depend on Substrait's shape.
2. **No GPU was available on the build machine, and no GPU measurement
   exists.** This is stated rather than assumed away.

   A verification attempt is now in place: `tools/gpu_verification.py` and
   `notebooks/aar_gpu_verification.ipynb` (Colab T4). **I cannot execute
   the notebook** — the user runs it, and the JSON result is committed under
   `data/gpu/` so the finding is evidence rather than a claim.

   What is *already* established on this CPU-only host, from
   `data/gpu/cpu_baseline.json` (2,000,000 rows): Arrow, DuckDB and Polars
   produce **identical** group-by checksums, so cross-engine correctness
   does not depend on a GPU; AAR completes without a GPU and reports why
   each engine was chosen; and the cost model correctly prefers CPU in all
   nine sampled size/parallelism combinations on a machine with no GPU.

   **Two defects, one much larger than the other.**

   *First*: `create_engine("cudf")` on a host without cuDF returned an
   Arrow engine and recorded nothing unless the caller passed a `ledger`.
   A verification script trusting the return value would have filed
   Arrow's timings under the label "cudf". **Now fixed.** `create_engine`
   guarantees three things: with no ledger the substitution goes to
   `aar.failures.process_ledger()` *and* raises a `RuntimeWarning`; with a
   ledger it is recorded and the warning is suppressed (the caller is
   listening); and `allow_degradation=False` refuses the substitution
   outright. `tools/gpu_verification.py` now uses that third form, and six
   tests in `TestSubstitutionsAreNeverSilent` lock it in.

   *Second, and the real one — **now fixed**.* There was no cudf engine to
   run. No `CudfEngine` or `PolarsGPUEngine` class existed anywhere, while
   the capability registry declared sixteen engines and `ENGINE_FACTORIES`
   implemented six. For the other ten, `create_engine` could only ever
   return a fallback. The gap was invisible precisely because the silent
   fallback made "cudf" look as though it constructed fine.

   Both GPU engines now exist: `aar/engines/cudf_engine.py` (device-resident
   relational execution) and `aar/engines/polars_gpu_engine.py` (Polars' lazy
   pushdown with RAPIDS as the collect engine). `TestDeclaredIsNotImplemented`
   tracks what remains unimplemented — Ray, Dask, Spark, and the five remote
   connectors — and fails the moment that set changes, so implementing an
   engine is a deliberate act that shrinks a list.

   Building them surfaced three further defects, all of which would have
   been silent on a GPU host:

   * `create_engine` returned an engine that had *constructed but declined
     to execute*. A caller got a `CudfEngine` on a GPU-less machine, and
     every operation failed one layer up with no record of the
     substitution. A decline at construction now means "absent".
   * The named-aggregation form `{"total": ("amount", "sum")}` was removed
     in **pandas 3.0**, and cuDF tracks pandas. The engine now uses
     `NamedAgg` and `.agg(**spec)`, which both accept.
   * `COUNT(*)` has no column to point at in a named aggregation. It is now
     a sum over a synthesised all-ones column; the alternative,
     pandas' `size`, counts nulls and would overstate `COUNT(x)`.

   `TestDeclaredIsNotImplemented` also gained a regression test: an engine
   that is *implemented but unusable* (cudf here, with no GPU) must still
   degrade with a recorded reason, not be reported as a missing class.

   **A T4 would not prove everything.** It is compute capability 7.5: no
   bfloat16, FP64 at 1/64 of FP32. A workload that wins on a T4 may lose on
   an A100. One run establishes that the path *works* and that the planner
   chooses sensibly; calibration needs several machines, which is a
   separate and larger task.
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
8. **The optimiser handles a branching DAG exactly, but the search is
   exhaustive.** `AdaptivePlanner.plan()` builds the real segment dependency
   graph via `segment_predecessors` and charges a crossing for *every*
   segment feeding each one, so a two-input join is billed for both inputs
   rather than the one that happened to be numbered before it. The previous
   single-"previous engine" table made a plan that straddled engines cost
   exactly the same as one that never moved any — the crossings were counted
   nowhere. Assignment selection is exhaustive over the product of candidate
   sets, which is exact but exponential; above 20,000 combinations it falls
   back to a per-segment local choice. A DP over engine *sets* would make
   that bound much larger and is the next step.
9. **A segment is priced as a sum of its operations, not a fused query.**
   `CostModel.segment_cost` now sums every operation in a segment, each at the
   size it actually sees. That is deliberately *not* a fused estimate: the
   executor dispatches one node at a time, so there is no fusion to credit.
   If per-segment fusion is ever implemented in the executor, the cost model
   must change with it — a fused estimate against an unfused runtime is the
   same class of error as reporting an engine that never ran.
10. **Estimation error is per-run, not accumulated.** `EstimationLog` compares
   each profiled source against the rows the scan really produced, and
   `aar run` prints it. But errors are keyed by node id, which changes on
   every build, so a systematically wrong profiler would have to be caught on
   each run rather than remembered. A stable semantic identifier is the
   missing piece.
10. **SQL/MongoDB scan dispatch was newly wired but is untested live.**
   `SCAN_SQL` and `SCAN_MONGO` previously fell through to
   `NotImplementedError`; they now route to `engine.read_scan`. SQLite is
   covered end to end; PostgreSQL, MySQL and MongoDB still need live servers.
10. **`SCAN_CONST` is declared but unimplemented.** It appears in the IR, the
    capability registry and the cost model's operation map, but nothing
    executes it. Named in `test_correctness_regressions.py` so the gap is
    visible rather than discovered at run time.
11. **Calibration still measures Arrow, not engines.** Curves are recorded
    under the CPU device category regardless of which engine produced them, so
    a single device-level curve cannot distinguish DuckDB from pandas. Engine
    specific calibration is not built.
12. **One aggregate per output column.** The IR stores `tuple[Agg, ...]`, but
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
| Orchestration, not replacement | No *required* dependencies; every engine optional and probed. Executing needs one — pyarrow is the minimum. |
| Types travel with data | Classification and lineage survive cast/rename/select |
| Data movement is minimised | Pushdown capability modelled per source; cost model makes transfer explicit |

---

## 10. How to run

```powershell
cd d:\AAR

# The workspace venv already exists at .\.venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Verify everything
.\.venv\Scripts\python.exe -m pytest -q                 # 555 tests
.\.venv\Scripts\python.exe tools\check_syntax.py        # parse every module
.\.venv\Scripts\python.exe tools\check_assets.py        # validate logo.svg
.\.venv\Scripts\python.exe tools\smoke_run.py           # plan + run + verify the numbers
.\.venv\Scripts\python.exe tools\manual_test.py         # guided manual acceptance
.\.venv\Scripts\python.exe tools\audit.py               # large real-data audit
.\.venv\Scripts\python.exe tools\fetch_data.py          # NYC taxi Parquet, provenance in MANIFEST

# GPU: run on a host that actually has a CUDA device.
# tools\gpu_verification.py is also packaged as notebooks/aar_gpu_verification.ipynb.
.\.venv\Scripts\python.exe tools\gpu_verification.py    # writes aar_gpu_result.json

# Package
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist\*

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

### 3.17 What is built beyond the goal, and what the goal still needs

The honest gap analysis. `system.md` describes a distributed, GPU-aware,
multi-format analytics platform. What exists is a *verified single-node
core*, and the difference is worth stating precisely rather than summarising
as "on track".

#### What was built that the specification did not ask for

Three things came out of verification work rather than the feature list,
and they are arguably the most valuable things in the repository:

1. **A large-data audit** (`tools/audit.py`). Not in the spec. It runs
   every engine, a cross-engine agreement sweep with *Python ground truth*
   as the referee, every CLI command, and the spec's own principles against
   3.07M real rows, exiting non-zero on failure. It found eight defects
   that 540 unit tests did not, including two silently-wrong-answer bugs.
2. **A manual test kit** (`tools/manual_test.py`). Not in the spec. A unit
   suite proves the system agrees with itself; this is for whether it
   agrees with a human, which is a different and more important claim.
3. **Build-time source guards** (`TestSourceParses`). Not in the spec. They
   exist because a docstring split by an interrupted edit once made a module
   that imported nothing, and the failure looked like fifteen unrelated
   test errors.

#### What the goal needs and does not have

| # | Area | Spec | State |
|---|---|---|---|
| 1 | **GPU execution** | §11 pain point | No GPU on this machine. Code paths, transfer model and `ENGINE_ABSENT` degradation written and exercised; **calibration curves unmeasured** |
| 2 | **Distributed execution** | Ray, Dask, Spark, Trino, Arrow Flight | **Not implemented.** The registry *declares* them; none can run |
| 3 | **Delta / Iceberg** | §7 open formats | **Not implemented.** Parquet only |
| 4 | **Substrait import/export** | §7 | **Deliberately not done**, documented as a deviation: pre-1.0, and it has no vocabulary for AAR's classification |
| 5 | **Streaming** | implied by "big data" | **Not implemented.** Everything materialises |
| 6 | **PostgreSQL / MongoDB wire** | §10 | SQL generation tested; **the protocol has never run against a server** |
| 7 | **Scheduler** | §14 | Budgets and a profile only; no resource manager |
| 8 | **Workbench depth** | §17 | Panels are static — no drag, resize, dock, data grid, query editor or run history |
| 9 | **Structured decision logging** | §16 | Trace and audit exist; no structured log sink |
| 10 | **Cross-machine calibration** | principle 8 | **Untestable here** — one machine |

#### What I need, and what stops me

**Hardware and services I do not have:**

- **A GPU.** The eleventh pain point — "organisations over-provision GPUs
  because utilisation is low and scheduling is coarse-grained" — is one of
  the reasons this project exists, and I cannot measure a single
  millisecond of it. The cost model *reproduces the specification's worked
  example* (80 ms kernel + 400 ms CPU + 400 ms transfers → CPU wins), but
  that is the model agreeing with a paper, not with silicon.
- **A PostgreSQL server and a MongoDB server.** Both connectors exist and
  their dialects and SQL generation are tested. Neither has ever spoken to
  its database.
- **A second machine.** Principle 8 says "two machines with identical specs
  can behave differently; the system measures, not assumes." I can measure
  one. The claim is untested by construction.
- **A cluster and a multi-node target.** The whole distributed layer.

**Judgement calls I need you to make, because guessing would be worse than
asking:**

1. **Workbench depth versus breadth.** I could add drag-resize-dock, a
   sortable data grid and a query editor to the existing UI, or add a third
   and fourth data source (Delta, a real SQL server). Both are real gaps;
   I cannot do both well.
2. **Whether the local-web Workbench is the right shape at all.** A
   dependency-free local server suits air-gapped single-node, which is the
   deployment the spec calls out. If the intended target is a shared team
   server, the security and multi-user model changes materially.
3. **Whether to keep pushing on single-node depth or start the distributed
   layer.** Starting distributed without a GPU or a cluster means writing
   code that cannot be run, which is how unverified code gets written.

**A constraint you should know about, which is not about the product:**

This shell is a genuinely difficult environment. It cannot stream command
output back, it mangles quotes in inline Python, and I have to run every
command detached and poll a log file. That cost a large fraction of this
session's effort — a malformed one-liner cost me a full debug cycle several
times, and one stale `.pyc` made correct code look broken for ten minutes.
It has not produced a wrong answer, but it has repeatedly slowed the loop
from "try, see, fix" to "write to a file, wait, read, fix". I have written
`tools/parse_check.py`-style diagnostics to compensate, which is a
workaround, not a solution.


The specification's §17 asks for a workbench, and — more importantly —
names *why*: "users resist tools when they cannot understand how results
are generated; lack of trust in a black box is a primary adoption
barrier." So the explainability panel is treated as the central requirement,
not a tab to add later.

**Standard library only.** No framework, no build step, no CDN. A UI that
needed a package install would be a UI that cannot start on the air-gapped
machines the rest of the design is for, and a test asserts the served HTML
contains no `http://` or `https://` at all. It binds to loopback by
default, because it can execute pipeline files and principle 2 is that
outbound is denied by default.

**Global usability, implemented rather than promised.** Five languages
(English, Spanish, French, Hindi, Arabic) with runtime translation, RTL
layout set explicitly rather than guessed, dark/light/system themes, three
density modes, and a font stack naming Noto Sans Arabic and Noto Sans
Devanagari so those scripts actually render. The rule that matters: **a
missing string falls back to English and never renders blank**, because an
unlabelled button is unusable with a screen reader. That makes the fallback
an accessibility requirement, not tidiness.

**Every promise in the UI is a test.** Thirty tests check that the
accessibility claims are real: every `<input>`/`<select>` has an accessible
name, there is a skip link, tabs declare their roles, results land in an
`aria-live` region, and there is no `title=` attribute anywhere — because
the specification's explainability requirement is not met by a tooltip
nobody can screenshot. Keyboard shortcuts are documented in a Help panel
rather than being folklore, and every engine's "not installed" reason is
rendered as visible text, never hidden.

Writing the tests found a **real bug**: `_plan_for` exits by raising
`SystemExit`, which is a `BaseException`, so `except Exception` missed it
and a typo in a filename killed the HTTP handler instead of returning a
sentence explaining what was wrong. An API that dies on bad input is worse
than one that complains. Now caught, with a test that fails if a
`SystemExit` ever escapes again.

**What is deliberately not built yet.** The spec's layout also asks for
draggable, resizable, dockable panels; a data grid with sorting and
filtering; a query editor; and a run history view. The first version has
the information architecture, the API, the theming, the i18n and the
accessibility, and the panels are static. That is a real gap and it is
recorded as one rather than implied done.

### 3.16 A manual test kit, because a passing suite is not the same as usable

`tools/manual_test.py` walks a person through six checks using the real CLI
and real files: what is installed, the shipped example end to end, the
privacy path, whether failures are loud, a decision trace, and the large
data path. Each step says what to do, what a correct answer looks like,
and *what to look for* — because the usual failure mode of a manual test is
"it ran, but I could not tell whether that was right."

    python tools/manual_test.py          # the whole tour
    python tools/manual_test.py privacy  # one step
    python tools/manual_test.py --list

The automated suite proves the system agrees with itself. This is for the
different claim that decides whether an analyst trusts it: that it agrees
with a human.

### 3.15 The Analyst Workbench — `src/aar/workbench/`, `aar workbench`

The specification's §17 asks for a workbench, and — more importantly —
names *why*: "users resist tools when they cannot understand how results
are generated; lack of trust in a black box is a primary adoption
barrier." So the explainability panel is treated as the central requirement,
not a tab to add later.

**Standard library only.** No framework, no build step, no CDN. A UI that
needed a package install would be a UI that cannot start on the air-gapped
machines the rest of the design is for, and a test asserts the served HTML
contains no `http://` or `https://` at all. It binds to loopback by
default, because it can execute pipeline files and principle 2 is that
outbound is denied by default.

**Global usability, implemented rather than promised.** Five languages
(English, Spanish, French, Hindi, Arabic) with runtime translation, RTL
layout set explicitly rather than guessed, dark/light/system themes, three
density modes, and a font stack naming Noto Sans Arabic and Noto Sans
Devanagari so those scripts actually render. The rule that matters: **a
missing string falls back to English and never renders blank**, because an
unlabelled button is unusable with a screen reader. That makes the fallback
an accessibility requirement, not tidiness.

**Every promise in the UI is a test.** Thirty tests check that the
accessibility claims are real: every `<input>`/`<select>` has an accessible
name, there is a skip link, tabs declare their roles, results land in an
`aria-live` region, and there is no `title=` attribute anywhere — because
the specification's explainability requirement is not met by a tooltip
nobody can screenshot. Keyboard shortcuts are documented in a Help panel
rather than being folklore, and every engine's "not installed" reason is
rendered as visible text, never hidden.

Writing the tests found a **real bug**: `_plan_for` exits by raising
`SystemExit`, which is a `BaseException`, so `except Exception` missed it
and a typo in a filename killed the HTTP handler instead of returning a
sentence explaining what was wrong. An API that dies on bad input is worse
than one that complains. Now caught, with a test that fails if a
`SystemExit` ever escapes again.

**What is deliberately not built yet.** The spec's layout also asks for
draggable, resizable, dockable panels; a data grid with sorting and
filtering; a query editor; and a run history view. The first version has
the information architecture, the API, the theming, the i18n and the
accessibility, and the panels are static. That is a real gap and it is
recorded as one rather than implied done.


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

Manual test kit: run the tour by hand, then the Workbench

    python tools/manual_test.py            # the guided tour
    python tools/manual_test.py --list     # just the step names
    aar workbench --open                   # the UI, in a browser

The manual kit is not a duplicate of the suite. The suite proves AAR agrees
with itself; the kit is for the different claim that decides whether an
analyst trusts it — that it agrees with a human. Six steps, real CLI, real
files, and for each one a statement of what a *correct* result looks like,
because the usual failure of a manual test is "it ran, but I could not tell
whether that was right".




