<p align="center">
  <img src="logo.svg" alt="AAR - Adaptive Analytics Runtime" width="600">
</p>

# Adaptive Analytics Runtime (AAR)

An orchestration layer for analytical work across Excel, SQL, NoSQL, Python,
local files and large-scale engines. AAR does not replace those tools — it
plans, routes, explains and audits them.

> **Status: AAR runs pipelines.** Nine of nineteen specified layers are built,
> tested and verified against real data on real files. See [`report.md`](report.md)
> for full status and what comes next.
>
> `aar run` executes a pipeline end to end — read, filter, group, UDF, sort,
> write — and reports which engine actually ran each node and anything that
> degraded on the way.

**The goal is minimum total analytical cost — not GPU utilisation.** AAR should
proudly say "CPU selected" or "PostgreSQL selected" when that is optimal.

---

## Implemented ≠ installed

This distinction has caused a capable external reviewer to conclude that two of
AAR's headline capabilities do not exist, so it is stated explicitly rather
than left for a reader to infer from `aar engines` output.

| Engine | Class written? | On this machine? |
|---|---|---|
| Arrow | ✅ | ✅ |
| DuckDB | ✅ | ✅ |
| Polars (CPU) | ✅ | ✅ |
| pandas | ✅ | ✅ |
| Python UDF worker | ✅ | ✅ |
| Excel (read + write) | ✅ | ✅ |
| **cuDF (GPU)** | ✅ `engines/cudf_engine.py` | ❌ cudf not installed |
| **Polars GPU** | ✅ `engines/polars_gpu_engine.py` | ❌ cudf not installed |

`aar engines` printing `[no] cudf  cudf is not installed` means **this
laptop has no GPU stack**, not that AAR cannot execute on one. Both GPU
engines are registered in `ENGINE_FACTORIES`, are constructed and exercised by
`tests/test_gpu_engines.py`, and have been verified end-to-end on a T4
(`data/gpu/`). A plan may therefore legitimately *choose* a GPU engine; on a
machine without cuDF it degrades to CPU and records that it did.

Declared in the capability catalogue but **not** implemented — so the planner
refuses to name them rather than quietly substituting: Ray, Dask,
Spark RAPIDS, Trino, MongoDB engine, and the MySQL/PostgreSQL *engines*
(the SQL **connectors** for SQLite are real and tested against a real
database).


---

## Principles this codebase actually enforces

| Principle | How it is enforced |
|---|---|
| Zero external AI dependency | No network calls anywhere. Egress is `deny` by construction. |
| Empirical, not hardcoded | Cost curves are measured live on the machine, not read from a spec sheet. |
| Never silently fails | `DegradationLedger.assert_clean()` raises on unresolved blocking degradations. |
| Explainable by default | Engine choice, reason, estimate and fallback are fields on the plan node. |
| Orchestration, not replacement | No *required* dependencies. Every engine is optional and probed — but executing a query needs one, and pyarrow is the minimum. |
| Privacy-first by architecture | A `CONFIDENTIAL` tag in PostgreSQL is still there in the Excel output. |

---

## Install

### Conda (recommended on this machine)

Conda is already installed here (`conda 26.7.1`, `C:\Users\fzcit\miniconda3`).
Use a dedicated environment so AAR's engines cannot collide with the `base`
env or with your other work:

```powershell
conda create -n aar python=3.11 -y
conda activate aar
python -m pip install -e ".[arrow,duckdb,polars,excel]"
```

Check it worked — `aar` should be on your path and the engines should report
available:

```powershell
aar version
aar doctor
aar engines
```

`aar doctor` prints this machine's real hardware, and `aar engines` probes
each engine by importing it. If an engine says it is unavailable, the reason
is printed verbatim rather than being greyed out silently.

From then on, **every command in this README is run from the `aar`
environment**, with no `.venv\` prefix:

```powershell
conda activate aar
aar explain pipelines\example_orders.py
aar run    pipelines\example_orders.py
aar workbench                      # the GUI - see below
```

The four commands you need most often:

| Command | What it does |
|---|---|
| `conda activate aar` | Put the right Python on your path. Do this first, in every new terminal. |
| `aar doctor` | What this machine is, and what AAR can actually do on it |
| `aar run <pipeline>` | Execute a pipeline and print the result |
| `aar profile <file>` | Measure a data file: rows, size, per-column statistics |
| `aar workbench` | Open the GUI in your browser |

If `conda activate` is not recognised in a new PowerShell window, run
`conda init powershell` once, then open a new terminal.

### pip / venv (alternative)

If you would rather not use conda:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\aar.exe doctor
```

The core has **no required third-party dependencies** and runs on the standard
library alone. Engines are opt-in extras:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[arrow,duckdb,polars,excel]"
```

---

## GUI — yes, there is one

```powershell
aar workbench
```

That opens `http://127.0.0.1:8765` in your browser. It is a real UI over the
real system, not a demo: the engine list comes from a live probe, `Explain`
returns the actual planner's decision trace, and `Run` executes the pipeline
through the same `PipelineService` the CLI uses. The result grid pages
through real rows.

```powershell
aar workbench --port 9000              # if 8765 is taken
aar workbench --no-browser             # start it without opening a browser
aar workbench --open                   # open a browser (the default)
```

It binds to `127.0.0.1` only, so it is reachable from this machine and
nothing else. **Do not** pass `--host 0.0.0.0` on a shared or untrusted
network: the server has no authentication and `/api/run` executes Python
pipeline files from disk.

## Measure your data first

```powershell
aar profile data\nyc_taxi_2022_03.parquet
aar profile data\nyc_taxi_2022_03.parquet --json
```

```
data\nyc_taxi_2022_03.parquet: 3,627,882 rows, 55,682,369 bytes (15 B/row, parquet-metadata)
  tpep_pickup_datetime, ~3.7 chars
  trip_distance, ~1.4 chars
  passenger_count, 3% null, ~0.2 chars
```

Every profile states its own provenance. `parquet-metadata` is exact and free
— the row count and per-column widths come from the footer, so nothing is
read. `sampled` means a bounded sample was extrapolated, and the numbers are
marked as estimates. A profile that does not say which one it is would be a
guess wearing a measurement's clothes.

`aar run` does this for you automatically, so the planner's arithmetic rests
on measured sizes rather than on the `estimated_bytes` a pipeline declares
about itself.

## Verify

With conda active:

```powershell
python -m pytest -q                   # 992 tests
python tools\check_syntax.py          # parse every module
python tools\smoke_run.py             # build a plan, run it, check the numbers
python tools\manual_test.py           # guided tour, by hand
python tools\debug_calib.py           # per-benchmark timings
```

Or without activating anything, using the environment's interpreter directly:

```powershell
conda run -n aar python -m pytest -q
```

The large-data audit needs a one-off download (~100 MB, NYC taxi
Parquet) and then runs every engine, every CLI command and the
specification's principles against it, exiting non-zero on any failure:

```powershell
python tools\fetch_data.py     # one time; records SHA-256 in data/MANIFEST.json
python tools\audit.py          # writes data/audit/audit.txt
```

## Run

```powershell
# `explain` plans only. It never opens your data, so it works with no file present.
# Both spellings work; the specification writes it with `plan`.
aar explain pipelines\example_orders.py
aar explain plan pipelines\example_orders.py

# `run` needs real data. Generate the sample workbook, then execute.
python tools\make_sample_data.py
aar run pipelines\example_orders.py

# `examples` is the fastest way in: list the bundled pipelines, print one,
# or write a copy you can edit. The examples are embedded in the package,
# so this works on an air-gapped machine.
aar examples
aar examples --show hello
aar examples --write hello my_pipeline.py

# `workbench` opens the local Analyst Workbench. Standard library only.
aar workbench --open

# `doctor` profiles this machine, `engines` lists what is installed,
# `calibrate` measures real operation costs, `policy` inspects a policy
# file, and `version` reports the optional-dependency status.
aar doctor
aar engines
aar calibrate --quick
aar policy show policy.json
aar version
```

```
EXECUTION

  ScanExcel    excel                  0 ->       240 rows    1467.0 ms
  Filter       arrow                240 ->       233 rows       0.6 ms
  GroupBy      arrow                233 ->         4 rows       1.6 ms
  PythonUDF    python_worker          4 ->         4 rows       3.0 ms
  Sort         arrow                  4 ->         4 rows       0.4 ms
  Limit        arrow                  4 ->         4 rows       0.1 ms
  Write        duckdb                 4 ->         4 rows     943.1 ms

  wrote FY26-report.xlsx (4 rows)

  2415.9 ms total, 4 rows out

  No degradations. Full-fidelity execution.
```

`--json` emits the same run as structured output, `--head N` previews rows, and
`--quiet` prints only the summary. A run that degraded is not hidden: every
substitution is recorded in the ledger and shown above.

## Command line

Every subcommand does real work; there are no placeholder commands.

```powershell
aar doctor         # this machine's hardware profile, with budgets
aar engines        # the engine catalogue and what is actually installed
aar calibrate      # measure this machine and write a cost profile
aar explain P.py   # plan only, no data touched
aar run P.py       # plan, execute, and report what actually ran
aar policy show P  # print a policy in readable form
aar version        # version and optional-dependency status
```

`doctor`, `engines`, `explain` and `run` also accept `--json`. Run `aar --help`
for the full list.

```console
$ aar engines
  tier 1 - source_pushdown
    [no ] postgresql     psycopg is not installed
  tier 2 - local_embedded
    [yes] duckdb         cpu engine available (1.5.5)
  tier 3 - gpu_acceleration
    [no ] polars_gpu     cudf is not installed
```

`python -m aar` works identically if the console script is not on your PATH.

---

## Governance

A classification tag is only worth carrying if something acts on it. AAR
enforces four obligations, and the defaults are **deny** — forgetting to
write a policy is the safe mistake.

```python
from aar.sdk import classify, excel, filter_, group_by, sum_

# Formats cannot carry AAR's tags — Parquet has nowhere to put them and an
# Excel header is just a string. `classify` is where you say what a column is.
orders = classify(excel("FY26-orders.xlsx"),
                  "customer", "ssn", tags=["PII"])
by_region = group_by(orders, "region", aggs={"total": sum_("amount")})
# `total` is now CONFIDENTIAL too, because it was computed from classified data.
```

```powershell
aar policy check --write-example example-policy.json
aar run pipeline.py --policy example-policy.json --role analyst --as dana
```

```console
$ aar policy show example-policy.json
Policy 'example-strict'
  egress: deny except ['postgres']
  max sensitivity at a network sink: INTERNAL
  RLS emea: WHERE region = EU
  CLS junior: drop ['ssn', 'national_id']
  CLS analyst: mask email->email
  CLS junior: mask email->email, amount->redact
```

| Obligation | What it does | Default |
|---|---|---|
| Egress | Blocks writes to a network sink | **deny** all |
| Classification | Blocks data above a sensitivity at a network sink | at most `INTERNAL` |
| Row-level | Injects `WHERE` for a role, so restricted rows are not returned | no filter |
| Column-level | Drops or masks a column by role or by sensitivity | mask at `CONFIDENTIAL` |

### Propagation

A derived column is **at least as sensitive as everything it came from**. This
is the part that stops `SUM(salary)` slipping through a policy that trusts
classification:

| Operation | Inherits | Why |
|---|---|---|
| `SUM(x)`, `AVG(x)`, `MIN(x)` | `x`'s tags | the result is a function of `x` |
| `COUNT(x)` | `x`'s tags | conservative; a non-null count is a property of `x` |
| `COUNT(*)` | nothing | a row count is a property of the table, not of a value |
| Group key | its own tags | **the key is the value** — bucketing by a quasi-identifier discloses it |
| UDF output | every column's tags | a function is opaque; assume it read everything |
| Join | both sides | either input can contribute to a joined row |

`declassify()` removes a tag with a written justification, which is stored on
the column. Without that escape hatch, analysts delete the source tags
instead — which is strictly worse.

Four properties worth knowing:

- **Row rules are injected before anything is computed.** RLS is a
  `SECURITY_FILTER` placed directly above each secured source, *before* the
  profiler reads it and before the planner runs. Applied at the write instead,
  `region = 'EU'` would arrive after `AVG(salary)` had already consumed every
  region — which is not a weaker filter but a different, and wrong, query.
- **Row rules are source-scoped.** `{"source": "orders", "predicate": ...}`
  says which input a restriction belongs to. An unscoped rule is accepted only
  when exactly one source could mean it; in a join, AAR refuses rather than
  guessing whose rows the analyst may see.
- **Enforcement happens before the bytes move.** A write that lands and is
  then noticed is a breach that already happened.
- **An unknown key in a policy file is an error, not a no-op.** A misspelled
  `mask_threshold` that were silently ignored would leave a policy that looks
  configured and protects nothing.
- **Masks preserve type.** A masked numeric column stays numeric, so masking
  does not break the next aggregate — which is how masks get removed.
- **Propagation is specific, not blanket.** An aggregate over an unclassified
  column stays unclassified, or the policy would mask everything and get
  switched off.

A run with no `--policy` is **not** unrestricted. AAR applies its baseline:
local processing and local output are allowed, network egress is denied, and no
RLS is configured because none was asked for. The bypass is explicit and
separately named:

```powershell
aar run pipeline.py --unsafe-disable-policy   # turns every check off
```

`aar explain` accepts `--role` and `--policy` too, and shows the *secured*
plan — an explain that printed unrestricted row counts would disclose the very
statistics the restriction exists to withhold.


---

## What exists today

### Canonical types — `aar.types`

The defence against silent cross-engine type corruption.

```python
from aar import types as T

T.from_source("timestamptz", "postgresql")   # Timestamp(us,UTC)
T.from_source("date", "mongodb")             # Timestamp(ms,UTC)
T.from_source("number", "excel")             # Float64
T.from_source("weird", "postgresql")         # raises UnmappableType

T.lossy(T.INT8, T.INT32)            # None  -> lossless
T.lossy(T.INT64, T.INT32)           # "integer int64 does not fit in int32"
T.lossy(T.TIMESTAMP("ns"), T.TIMESTAMP("us"))  # "resolution ns -> us is not exact"
```

Classification travels with the data:

```python
s = T.Schema((T.Field("salary", T.INT64,
                      classification=frozenset({"CONFIDENTIAL"})),))
s.cast({"salary": T.DECIMAL(12, 2)}).get("salary").classification
# frozenset({'CONFIDENTIAL'})   <- survives the cast
```

### Internal IR — `aar.ir`

25 node types and a typed expression algebra. Not Substrait: Substrait is
pre-1.0 and has no vocabulary for Excel ranges, Python UDFs or privacy tags.

```python
from aar.ir import BinOp, Col, Lit, Node, NodeType, ScanSpec, topological_order

scan = Node(NodeType.SCAN_PARQUET,
            scan=ScanSpec(kind="parquet", path="orders.parquet"))
filt = Node(NodeType.FILTER, inputs=[scan],
            predicate=BinOp(Col("amount"), ">", Lit(100.0)))
topological_order(filt)   # deterministic, cycle-detecting
```

Literals are escaped, so a value containing `'; DROP TABLE users; --` renders
as a quoted string and cannot break out of a pushdown.

### Hardware profile — `aar.hardware`

```python
from aar.hardware import HardwareProfile

p = HardwareProfile()
print(p.render())
#   os        Windows 10
#   cpu       Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz | 4P/4L | avx=scalar
#   memory    8.6GB total, 5.2GB available, bandwidth unmeasured
#   gpu       no GPU (no accelerator driver or runtime detected)
#   budgets   memory=4.1GB  vram=0B
```

A missing GPU is an *explained decision*, never an absence of information. What
could not be measured is recorded in each probe's `unknowns` list.

### Calibration — `aar.hardware.calibrate`

Measures this machine instead of assuming anything about it.

```python
from aar.hardware.calibrate import calibrate

store = calibrate(quick=True, fingerprint=profile.fingerprint())
store.best_device("groupby", 1_000_000)      # ('cpu', 0.0003)
store.render()
#   groupby/cpu: 0.0 ms fixed, 3.267e-10 s/byte, R2=0.953
```

A profile from a different machine is discarded, not used: a stale profile is
worse than none, because it is confidently wrong.

### Failure handling — `aar.failures`

All 22 failure modes with detect / handle / fallback / log, plus a ledger that
makes degradation structural rather than aspirational.

```python
from aar.failures import DegradationLedger, FailureKind

ledger = DegradationLedger()
ledger.record(FailureKind.GPU_UNAVAILABLE, "join", "no CUDA device",
              from_engine="polars_gpu", to_engine="polars_cpu")
print(ledger.render())
ledger.assert_clean()   # raises if a blocking degradation is unresolved
```

### Capability registry — `aar.capability`

Which engines can do what, and which are actually installed.

```python
from aar.capability import CapabilityRegistry

reg = CapabilityRegistry()
feasible, rejected = reg.feasible_with_reasons(node)
# feasible: ('duckdb', 'polars_cpu', 'pandas')   in tier order
# rejected: {'polars_gpu': 'no GPU (no accelerator driver or runtime detected)',
#            'cudf':     'cudf is not installed',
#            'ray':      'ray is not installed'}

print(reg.render())
#   tier 1 - source_pushdown
#     [no ] postgresql     cpu engine available
#     ...
```

The probe uses `importlib.util.find_spec`, never an actual import — importing
cuDF or Spark costs seconds and allocates device memory just to answer "is it
there?".

### Cost model — `aar.cost`

The specification's central GPU example, reproduced by arithmetic:

```python
from aar.cost import CostModel

m = CostModel()
cpu, src = m.node_cost(node, "duckdb",     1_000_000_000)
gpu, _   = m.node_cost(node, "polars_gpu", 1_000_000_000)

gpu.kernel_s   # faster on the GPU
cpu.total_s    # faster overall - the transfers win
```

`compute_s` also reports *where* the number came from, so a plan built on
pessimistic priors is visibly distinct from one built on measurements:

```python
value, source = m.compute_s(node, "duckdb", 1_000_000)
# source in {"history", "calibration", "calibration(penalised)", "prior"}
```

---

## Layout

```
src/aar/
  types.py              canonical types, loss detection, schema algebra, drift
  ir/nodes.py           25 node types, expression algebra, DAG ordering
  hardware/detect.py    six stdlib probes + HardwareProfile facade
  hardware/calibrate.py microbenchmarks, cost-curve fitting, profile store
  failures/registry.py  22 failure modes + DegradationLedger
  capability/registry.py 16 engines, 6 tiers, feasible sets, availability probe
  cost/model.py         six-term breakdown, transfers, residency, history
  cli.py                aar doctor / engines / calibrate / explain / run /
                        policy / workbench / version
  workbench/            the Analyst Workbench: local HTTP server, i18n, UI
  connectors/           Excel, Parquet/CSV/JSON, SQL, MongoDB
tests/                  540 behavioural tests
tools/                  check_syntax, smoke_run, manual_test, audit,
                        fetch_data, diag_groupby, debug_calib,
                        make_sample_data, check_assets
report.md               build status, bugs found, what comes next
system.md               the specification
```


## Non-goals

AAR is not a BI tool, not an LLM assistant, not an AutoML tool, and not a
replacement for Excel, SQL or Python. It does not require internet
connectivity, does not force AI into the workflow, and does not chase
"GPU everywhere".

Licence: Apache-2.0.

```
