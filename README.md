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

## Principles this codebase actually enforces

| Principle | How it is enforced |
|---|---|
| Zero external AI dependency | No network calls anywhere. Egress is `deny` by construction. |
| Empirical, not hardcoded | Cost curves are measured live on the machine, not read from a spec sheet. |
| Never silently fails | `DegradationLedger.assert_clean()` raises on unresolved blocking degradations. |
| Explainable by default | Engine choice, reason, estimate and fallback are fields on the plan node. |
| Orchestration, not replacement | Zero required dependencies. Every engine is optional and probed. |
| Privacy-first by architecture | A `CONFIDENTIAL` tag in PostgreSQL is still there in the Excel output. |

---

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The core has **no required third-party dependencies** and runs on the standard
library alone. Engines are opt-in extras:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[arrow,duckdb,polars,excel]"
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q             # 365 tests
.\.venv\Scripts\python.exe tools\check_syntax.py    # parse every module
.\.venv\Scripts\python.exe tools\smoke_run.py       # build a plan, run it, check the numbers
.\.venv\Scripts\python.exe tools\debug_calib.py     # per-benchmark timings
```

## Run

```powershell
# `explain` plans only. It never opens your data, so it works with no file present.
aar explain pipelines\example_orders.py

# `run` needs real data. Generate the sample workbook, then execute.
python tools\make_sample_data.py
aar run pipelines\example_orders.py
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
aar version        # version and optional-dependency status
```

`doctor` and `engines` also accept `--json`. Run `aar --help` for the full
list.

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
  cli.py                aar doctor / engines / calibrate / version
tests/                  222 behavioural tests
tools/                  check_syntax, debug_calib, smoke, truncate
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
