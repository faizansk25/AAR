# Optimisation programme: code, logic, approach

Written after four optimisation rounds, because the rounds taught a
distinction worth making explicit before more of them. Each section answers
the questions asked of it; **[C]**/**[L]**/**[A]** mark whether a claim is
about **code** (a specific implementation), **logic** (a rule the system
obeys), or **approach** (a choice about how to work).

---

## 1. WHAT — the issue, the evidence, the alternatives

### What the issue actually is

Not "the project is slow". Precisely: **AAR has two engines, `arrow` and
`pandas`, documented as fallbacks, and both contained per-row Python paths
on operations every columnar engine vectorises.** A fallback 1,000x slower
than the primary is not a fallback, because nobody waits for it. [L]

### The evidence

From `tools/bench_engines.py --real`: one fresh process per measurement,
1,000,000 rows of real NYC taxi Parquet, median of three, **all spreads
under 0.2%**.

| Defect | Engine | Before | After | Class |
|---|---|---|---|---|
| `filter`, per-row | pandas | 26,723 ms | 12.60 ms | [C] |
| `filter` on `AND` | arrow | 20,114 ms | 9.88 ms | [C] |
| `join`, Python hash | arrow | 4,462 ms | 129 ms | [C] |
| `join`, Python hash | pandas | 4,437 ms | 145 ms | [C] |
| `group_by`/`join` on nulls | arrow, pandas | **failed** | works | [L] |

### Assumptions — and the one that was false

The assumption that every round's premise was sound was wrong once. The
"213x Arrow group-by defect" was **97% a measurement artefact**: operations
timed sequentially in one process, so a preceding 1,000,000-row join left
the allocator holding memory and the next allocation was slow. Two
*identical* group-bys measured 18 ms and 2,975 ms in the same run. [A]

That retraction is in the git history, not quietly dropped. It cost a
plausible optimisation and produced a trustworthy harness.

### Alternatives considered

| Approach | Verdict |
|---|---|
| Write a fast engine | Rejected. Orchestration, not replacement. [L] |
| Delegate to DuckDB/Polars | Rejected: the air-gapped story needs `arrow`/`pandas` alone. [A] |
| Fix the fallbacks to be genuinely fast | **Chosen.** Small, verifiable, makes the air-gapped claim true. [A] |
| Call `pyarrow.compute`/`Table.join` natively | **Chosen** where semantics match; Python path kept where they do not. [A] |

### Risks taken

Speed was bought with a *semantic* risk: a fast wrong answer. Each fast
path is guarded by where the library disagrees with this system's
contract — NULL join keys (SQL says NULL ≠ NULL; Arrow matches them) and
column collisions (Arrow suffixes; this system says right wins). The
fallback stays correct, just slow. [L]

### Next steps

1. **Excel time is bounded by openpyxl, not by AAR.** If large writes
   matter, the lever is a different writer (`xlsxwriter`, or Calamine for
   reading), not a different call pattern. Decide deliberately. [A]
2. Consider `write_only=True` for its **memory** benefit on large writes,
   at the cost of the append/overwrite path no longer being reopenable.
3. Full suite and release checks after each change, never before. [A]
4. One T4 re-run to confirm no GPU regression. [A]

---

## 2. WHERE — locating the problem

| Layer | Status |
|---|---|
| `engines/arrow_engine.py` filter, join | fixed |
| `engines/pandas_engine.py` filter | fixed |
| `engines/_mask.py` (new, shared) | added |
| `interchange/table.py` nullability | fixed |
| **`connectors/excel.py` write** | fixed — and the fix was smaller than claimed (§2) |
| `arrow_engine.py::_python_join` | intentionally slow, guarded |
| `arrow_engine.py::_row_filter` | intentionally slow, fallback |

### The Excel finding — fixed, and the fix was smaller than claimed

```python
for record in table.arrow.to_pylist():
    for c, name in enumerate(table.column_names, start=1):
        ...
        ws.cell(row=row_cursor, column=c, value=value)
```

The same defect *shape* as every other win, on the system's most important
target — principle 13 is "Excel-first". So it was rewritten to call the
library's row API once per row and to iterate column-major.

**Then it was measured, and the claim turned out to be wrong.** At 50,000
rows x 12 columns (`tools/probe_excel_phases.py`):

| form | write ms | save ms | total ms |
|---|---|---|---|
| per-cell | 10,382 | 40,782 | 51,165 |
| append | 7,398 | 40,502 | 47,900 |
| write_only | 47,490 | 2,443 | 49,933 |

**The save dominates, and `ws.cell()` was never the bottleneck.** openpyxl
keeps a `Cell` object per cell in `ws._cells` whichever API writes it, so
`wb.save()` has exactly the same work to do. The change is a real ~6%
improvement and a large cut in call count — it is not the 1000x the comment
originally claimed, and that claim has been corrected in the code rather
than left to flatter the diff. [A]

`write_only=True` is not the answer to time either: it moves the 40s into
the write phase and makes save trivial, landing in the same place. Its
genuine benefit is **memory** — a write-only sheet streams rows out instead
of holding every cell. Time on this path is bounded by XLSX serialisation,
not by how AAR calls the library. [C]

Also removed: an unreachable `return openpyxl` after
`return table.num_rows`, and a latent layout bug. `_sheet_has_content`
called `ws.cell(row=1, column=1)` to ask whether a fresh sheet was empty —
but on a fresh sheet that call *materialises* cell A1, which advances
openpyxl's append cursor and pushes the header to row 2. The file gained a
blank first row, and the reader then took the wrong row as the header and
mis-inferred every type after it. **A probe that is not a read.** [L]

---

## 3. WHY — root causes

**Not four coincidences.** Three were the same mistake: *a per-row Python
path where a vectorised path existed*, written because it is obviously
correct and only measured later. The fourth — the nullability failure — is
that root cause from the other side: a value read from metadata that was
never checked against data. [L]

### Why it was not found sooner

The suite checked **correctness**, and every one of these paths was
correct. A test that passes does not tell you it is slow. Only a benchmark
found them, and the benchmark only became trustworthy once isolated. [A]

### Why this matters beyond speed

`arrow` and `pandas` are the engines the **air-gapped** claim rests on:
`arrow` is what runs when nothing else is installed, and
`pip install aar[arrow]` has no alternative. Before this programme, on real
data, those two engines **could not group_by or join at all** — a user on an
air-gapped machine would hit a `SchemaDriftError` and a 25-second filter. [L]

### Why opinions might differ

Someone could argue the Python paths should be deleted rather than kept.
That is legitimate and cheaper to maintain — but it breaks the air-gapped
promise, which is the project's reason to exist. The trade made here
preserves the promise at the cost of keeping slow code alive. [A]

---

## 4. HOW — implementation and measurement

Three independent gates:

1. `pytest` — 594 passed, 7 skipped. Correctness.
2. `tools/verify_release.py` — 14/14. The **installed wheel**, outside the
   source tree, running a real pipeline against independent Python
   arithmetic. Catches packaging bugs the dev venv hides.
3. `tools/bench_engines.py --real` — every measurement in a **fresh
   process**, spread printed beside the number, a ratio smaller than the
   spread called noise.

Gate 3 exists because gates 1 and 2 both passed while four 1000x defects
sat in the code. [A]

The test that would have caught all three fast-path regressions:

```python
def test_it_does_not_call_a_python_function_per_row(self):
    class Exploding(pd.DataFrame):
        def apply(self, *a, **k):
            raise AssertionError("per-row callback: the 1,400x defect")
```

A frame whose slow-path method raises, proving the fast path is used. [C]

**Reassessment cadence:** after every engine change, run the full suite and
release checks. After any benchmark change, re-measure something whose
number is already known and confirm it did not move — a harness that cannot
reproduce a past result is not measuring. [A]

---

## 5. WHO — people and power

| Who | Impact |
|---|---|
| **Air-gapped analyst** | Main beneficiary. Before: failed joins, 25-second filters. Has no alternative to `arrow`/`pandas`. |
| **Analyst on a normal machine** | Mostly unaffected — the planner picks DuckDB or Polars. Benefits only when those are absent. |
| **Maintainer** | Bears the cost of a guard matrix: every fast path needs a test proving it is used *and* a slow path kept correct. |
| **Downstream integrator** | Depends on metadata surviving every engine. Three of these defects would have silently dropped a `CONFIDENTIAL` tag. |

**Who holds the power:** the **engine abstraction**. It is the seam where a
library's semantics meet this system's contract, and every defect found
sits on it.

**Who might see it differently:** someone building for the common case
would delete the fallbacks and declare exotic predicates unsupported. That
is legitimate and cheaper to maintain — but it breaks the air-gapped
promise, which is the reason this project exists.

**Who benefits:** users on machines with nothing installed, and the
project's credibility. A system that claims air-gapped operation and then
cannot group a real Parquet file has a problem no amount of documentation
fixes. [A]

---

## 6. WHEN — timing

| Round | What | Outcome |
|---|---|---|
| 1 | pandas filter | 1,400x, real defect |
| 2 | cudf engines built | no kernel existed at all |
| 3 | T4 run 1 | detection works, no GPU execution |
| 4 | T4 run 2 | **GPU lost** — transfers dominate at 32 MB |
| 5 | T4 run 3 | cuDF verified end to end |
| 6 | Real data | **arrow and pandas failed outright** |
| 7 | nullability fix | all four engines survive real data |
| 8 | isolation | the 213x defect was 97% harness |
| 9 | native join | 4,462 ms → 129 ms |
| 10 | Excel write | `ws.cell` → `ws.append`; **claim retracted, 1.0×** |

**The cadence that produced those:** roughly one round per exchange, each
ending in a measurement rather than a claim. The longest round was the
nullability fix, because it was a **design change to the interchange
layer** rather than a patch, and had to be right rather than fast. [A]

**When results are expected:** immediately for filter and join — they are
measured. Unknown for Excel, because there is no benchmark for it yet; that
benchmark is the next piece of work. [A]

---

## 7. The three layers, summarised

**[C] Code** — the specific defects: `frame.apply` in pandas,
single-comparison-only Arrow filter, Python hash join, schema-derived
nullability. All fixed, all measured, all guarded by a test.

**[L] Logic** — the rules that made those defects dangerous rather than
merely slow: NULL never equals NULL; a joined column inherits both sides'
tags; a column with nulls must be declared nullable. These are now
*enforced* by the fallback paths, which is why the fast paths can be
trusted when they agree.

**[A] Approach** — the change in how the work is done. Measure before
claiming. Isolate before trusting a number. Test on real data before
believing a suite. Retract in public when a measurement turns out wrong.
The single retraction in this programme was the most valuable thing in it,
because it cost a plausible optimisation and produced a trustworthy
harness. [A]

connector. [C]

### Supporting data

- `data/audit/bench.json` — synthetic, superseded
- `data/audit/bench_real.json` — real taxi data, isolated, trustworthy
- `data/gpu/t4_run1..3.json` — GPU evidence, limits recorded
