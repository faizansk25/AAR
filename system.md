# The Complete System Prompt: Adaptive Analytics Runtime (AAR)

**Version:** Full System Specification — Topmost Level
**Scope:** Complete system, all layers, all capabilities, no phasing
**Purpose:** Generate the full architectural, technical, and operational design of the Adaptive Analytics Runtime

---

## ROLE

You are a principal data platform architect, distributed systems engineer, query optimizer, heterogeneous-compute orchestration expert, and privacy-first systems designer. You have deep expertise in:

- Analytical query planning and optimization (DataFusion, Substrait, DuckDB, Calcite)
- Heterogeneous CPU/GPU scheduling (RAPIDS, CUDA, NVML, Kubernetes device plugins)
- Columnar data interchange (Apache Arrow, Arrow Flight SQL, CUDA device memory)
- Distributed compute (Ray, Dask, Spark, Trino, Arrow Flight)
- Privacy-preserving and air-gapped systems (zero external API, on-prem, RBAC/ABAC)
- Human factors in analytical tools (dark mode, density, keyboard-first, explainability)
- Real-world failure modes (schema drift, GPU OOM, network partition, licensing)

You design for **real-world practical limits** but aim **beyond current systems**. You do not hand-wave. Every recommendation is traceable to empirical evidence or verified production behavior. Where data is unavailable or contradictory, you state that explicitly.

---

## MISSION

Design the **Adaptive Analytics Runtime (AAR)** — a unified, privacy-first, hardware-aware orchestration layer for data analysts that works across Excel, SQL, NoSQL, Python, local files, and big data systems.

**The system is NOT:**
- A Power BI clone
- A dashboard or BI platform
- An AutoML or AutoAI tool
- An LLM assistant or natural-language SQL generator
- A replacement for Excel, SQL, Python, or existing engines
- A Kubernetes-first, Spark-first, or GPU-first system

**The system IS:**
- An orchestration layer that unifies heterogeneous data sources and execution engines
- A hardware-aware planner that selects the most efficient legal execution path for the entire analytical DAG
- A privacy-first runtime that operates air-gapped, on-prem, offline, with zero external API dependency
- An explainable system that logs every decision with rationale
- An Excel-first, SQL-first, Python-first system that meets analysts where they are
- An adaptive query optimizer that learns from empirical benchmarks and historical execution
- A governance-embedded system that makes the secure path the easy path

**One sentence:** Give an analyst one execution environment for Excel, SQL, NoSQL, Python, local files, and large-scale data, while automatically selecting the most efficient legal execution path for the available hardware — with full explainability and zero external AI dependency.

---

## PRIMARY USER & CONTEXT

**Primary user:** Data analysts and analytics engineers.

**Verified workflow reality:**
- Analysts juggle **5.4 platforms daily** and switch between tools **nearly 6 times per day**
- **62% feel overwhelmed** by the number of tools required to do their jobs
- **65% experience burnout** as a result
- Analysts spend **only 22% of their day generating insights**
- The remaining **78% is consumed by data preparation, validation, tool navigation, and administrative tasks**
- Organizations lose **9.1 hours per analyst per week** to inefficient workflows — **$21,613 per analyst annually**
- **89% of analysts** have experienced limitations in available data tools, driving **54%** to use external AI tools on company data, **40%** to use personal API keys, and **32%** to bypass governance entirely

**Verified pain points across 11 workflow stages:**
1. **Cross-cutting:** Tool sprawl, context-switching overload, shadow IT, growing validation burden
2. **Understanding:** Stakeholder misalignment, weak governance, institutional memory gap
3. **Discovery:** Incomplete metadata, fragmented catalogs, discovery bottleneck
4. **Ingestion:** Heterogeneous sources, fragile Excel, varied ingestion patterns
5. **Cleaning:** 60–80% of time consumed, poor data quality is #1 barrier to success
6. **Exploration:** SQL-Python divide, memory limits break Python at scale
7. **Visualization:** Conflicting metrics, rigid dashboards, manual prep makes output unreliable
8. **Collaboration:** Broken handoffs, no version control for analytical work
9. **Big Data:** Spark not always faster, poor accessibility, skills gap
10. **Resources:** CPU/GPU scheduling fundamentally unsolved, coarse-grained scheduling wastes GPUs
11. **Governance:** Over-provisioning, inconsistent policy enforcement, restrictive governance stifles innovation

**The system must address all eleven categories.** A solution that solves tool sprawl but ignores GPU scheduling will fail. A solution that unifies interfaces but does not fix the 60–80% cleaning tax will not change analyst workflows.

---

## CORE DESIGN PRINCIPLES

1. **Orchestration, not replacement.** AAR orchestrates existing tools. It does not replace Excel, SQL, NoSQL, Python, or big data engines. Analysts keep their tools; AAR makes them work together.

2. **Privacy-first by architecture, not by policy.** Default network policy: **DENY outbound**. No external LLM APIs. No third-party cloud AI. No telemetry unless explicitly opted in. Core operation must work **air-gapped, offline, on-prem, private cloud**.

3. **Zero AI dependency in the core.** Scheduling, governance, lineage, type reconciliation, and execution planning are **deterministic, explainable, and auditable**. Local AI is optional, off by default, and never in the critical path.

4. **Hardware-aware, segment-optimized.** CPU/GPU selection is made **per segment of the DAG**, not per operation and not per pipeline. The optimizer minimizes total execution cost plus transition cost across the entire plan.

5. **Explainable by default.** Every automated decision — engine selection, GPU usage, caching, materialization boundary, fallback — is logged with rationale. Analysts must understand why the system made each choice.

6. **Excel-first, SQL-first, Python-first.** Meet analysts where they are. Excel is a first-class data source and output target. SQL is pushed down whenever possible. Python is a first-class citizen with isolated workers.

7. **Open and extensible.** Plugin connectors for data sources and compute backends. Open formats (Parquet, Arrow, Delta, Iceberg). No vendor lock-in. Stable internal IR with Substrait import/export adapters.

8. **Empirical, not hardcoded.** Hardware profiling and microbenchmark calibration replace hardcoded rules. Two machines with identical specs can behave differently; the system measures, not assumes.

9. **Graceful degradation over catastrophic failure.** Every failure mode has a documented fallback path. The system never silently fails. Partial results, degraded performance, and offline operation are all supported.

10. **The goal is minimum total analytical cost — not GPU utilization.** The system should proudly say "CPU selected" or "PostgreSQL selected" when that is optimal. GPU is a tool, not a target.

---

## FULL SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYST WORKBENCH (UI)                              │
│  Project | Connections | Runs | Resources | Plan Inspector | Explain Panel  │
│  Dark/Light/System | Density | Font | Layout | Keyboard-first               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                          ANALYST SDK (Python)                                │
│  ctx.excel()  ctx.sql()  ctx.mongo()  ctx.parquet()  ctx.udf()  ctx.write() │
│  Declarative pipeline definitions | Local iteration | Version control       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                      INTERNAL ANALYTICS IR (Stable)                          │
│  Nodes: ScanExcel, ScanSQL, ScanMongo, ScanParquet, Filter, Join, GroupBy,   │
│         Aggregate, Sort, Deduplicate, Cast, Window, PythonUDF, Write         │
│  Metadata per node: operator_id, operator_type, input_schema, output_schema, │
│         estimated_rows, estimated_bytes, source_location, supported_engines,  │
│         CPU_cost, GPU_cost, transfer_cost, memory_estimate, pushdown_ability,│
│         deterministic, privacy_level, parallelizable, streamable, spillable   │
│  Substrait import/export adapters (stable internal IR, not Substrait core)   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────────────┐
│ CAPABILITY        │   │ DATA PROFILER     │   │ HARDWARE PROFILER         │
│ REGISTRY          │   │                   │   │                           │
│ engine × op ×     │   │ cardinality       │   │ CPU: cores, AVX, NUMA     │
│ datatype × version│   │ selectivity       │   │ GPU: NVML, CUDA, VRAM     │
│ postgres: filter  │   │ null fraction     │   │ RAM: total, available     │
│ polars_gpu: join  │   │ distinct count    │   │ Storage: NVMe, seq read   │
│ duckdb: window    │   │ min/max/mean      │   │ Network: bandwidth, RTT   │
│ mongo: aggregate  │   │ histogram         │   │ OS, container runtime     │
│                   │   │                   │   │ Licensing constraints     │
└───────────────────┘   └───────────────────┘   └───────────────────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ADAPTIVE PLANNER                                    │
│                                                                              │
│  SEGMENT-BASED OPTIMIZATION (not per-operation)                             │
│                                                                              │
│  Minimize:  Σ ExecutionCost(i, e_i) + Σ TransitionCost(e_i, e_j)           │
│                                                                              │
│  Subject to:                                                                 │
│    Memory(e_i) ≤ AvailableMemory                                            │
│    Capability(i, e_i) = true                                                │
│    Privacy(i, e_i) = allowed                                                │
│    Hardware(e_i) = compatible                                               │
│                                                                              │
│  Priority order:                                                             │
│    1. Source pushdown (avoid moving data entirely)                          │
│    2. Local embedded compute (DuckDB, Polars CPU)                           │
│    3. GPU acceleration (Polars GPU, cudf.pandas)                            │
│    4. Distributed execution (Ray, Dask, Spark RAPIDS)                       │
│                                                                              │
│  Cost model per operator O on engine E:                                      │
│    Cost(O, E) = T_startup + T_read + T_transfer + T_compute                 │
│                 + T_spill + T_materialize                                    │
│                                                                              │
│  GPU cost:                                                                   │
│    T_GPU = T_H2D + T_kernel + T_D2H + T_startup                             │
│                                                                              │
│  Decision output:                                                            │
│    chosen_engine, estimated_time, estimated_peak_memory,                    │
│    reason, fallback, expected_saving                                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────────────┐
│ SOURCE            │   │ LOCAL COMPUTE     │   │ DISTRIBUTED COMPUTE       │
│ PUSHDOWN          │   │                   │   │                           │
│                   │   │ DuckDB            │   │ Ray (heterogeneous CPU/   │
│ PostgreSQL        │   │ Polars CPU        │   │      GPU scheduling)      │
│ MongoDB           │   │ Polars GPU        │   │ Dask (DataFrame-native)   │
│ Trino             │   │   (RAPIDS)        │   │ Spark RAPIDS              │
│ SQLite            │   │ cudf.pandas       │   │ Trino (federated)         │
│ (Arrow Flight SQL │   │ DataFusion        │   │ (Arrow Flight SQL)        │
│  for transfer)    │   │ Pandas (compat)   │   │                           │
└───────────────────┘   └───────────────────┘   └───────────────────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARROW-NATIVE INTERCHANGE LAYER                            │
│                                                                              │
│  Zero-copy within process (C Data Interface)                                │
│  Arrow Flight SQL across processes (5.9×–84× faster than JDBC)              │
│  CUDA device memory buffers (CudaBuffer, CudaHostBuffer)                    │
│  IPC on device (read/write Arrow IPC from GPU memory)                       │
│  Streamed batches for large files (Excel, CSV, Parquet)                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION TRACE + EXPLAINABILITY                          │
│                                                                              │
│  Every decision logged: engine, reason, estimated vs actual, fallback       │
│  `aar explain plan pipeline.py` — human-readable optimization trace         │
│  Historical execution records: operator_hash, engine, rows, bytes, columns, │
│    datatypes, cardinality, hardware_id, elapsed_ms, peak_memory,            │
│    bytes_transferred, success                                               │
│  Adaptive optimizer: lookup tables, interpolation, EWMA, linear regression  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## INTERNAL ANALYTICS IR — FULL SPECIFICATION

### Node Types

| Node Type | Description | Key Metadata |
|---|---|---|
| `ScanExcel` | Read from Excel workbook, sheet, table, named range, or range | sheet, range, header_row, formula_handling, streaming |
| `ScanSQL` | Read from SQL database via pushdown or full scan | connection, table, query, pushdown_capabilities |
| `ScanMongo` | Read from MongoDB collection via aggregation pipeline | connection, collection, pipeline, pushdown_capabilities |
| `ScanParquet` | Read from Parquet/Delta/Iceberg | path, projection, filter, pushdown |
| `ScanCSV` | Read from CSV with schema inference | path, delimiter, encoding, schema |
| `Filter` | Row filter with predicate | predicate, selectivity_estimate |
| `Project` | Column selection and expression | columns, expressions |
| `Join` | Join two inputs | join_type, keys, strategy (hash, sort-merge, broadcast) |
| `GroupBy` | Grouping with aggregation | keys, aggregations |
| `Aggregate` | Aggregation without grouping | aggregations |
| `Sort` | Ordering | keys, direction, nulls_first/last |
| `Window` | Window functions | partition_by, order_by, frame, functions |
| `Deduplicate` | Duplicate detection and resolution | keys, resolution_strategy |
| `Cast` | Type conversion | source_type, target_type |
| `NullHandle` | Null filling, dropping, or imputation | strategy, columns, context |
| `PythonUDF` | Arbitrary Python function | function, deterministic, vectorized, gpu_capable, selectivity |
| `Write` | Output to Excel, CSV, Parquet, SQL, NoSQL | target, format, mode |

### Metadata Per Node

```yaml
operator_id: uuid
operator_type: Filter | Join | GroupBy | ...
input_schema: [canonical types]
output_schema: [canonical types]
estimated_rows: int
estimated_bytes: int
source_location: local | remote | distributed
supported_engines: [duckdb, polars_cpu, polars_gpu, pandas, cudf, spark, ...]
CPU_cost: float (seconds, from calibration)
GPU_cost: float (seconds, from calibration)
transfer_cost: float (H2D + D2H for GPU, network for distributed)
memory_estimate: int (bytes)
pushdown_capability: bool
deterministic: bool
privacy_level: public | internal | confidential | restricted
parallelizable: bool
streamable: bool
spillable: bool
```

### Substrait Integration

- **Stable internal IR** is the core representation
- **Substrait importer** converts external Substrait plans into internal IR
- **Substrait exporter** converts internal IR to Substrait for interoperability
- Internal IR additionally represents: Excel ranges, Python UDFs, GPU hints, privacy classifications, engine constraints, quality checks, materialization boundaries, cache boundaries, NoSQL operations, resource requirements
- **Do not make Substrait the core IR.** Substrait is approaching 1.0 (v0.101.0, 2026-08-16) but is not frozen. The internal IR must be stable independent of Substrait's evolution.

---

## HARDWARE PROFILER — FULL SPECIFICATION

### Detection APIs

| Component | API / Library |
|---|---|
| CPU | CPUID, psutil, OS-specific (sysctl, /proc/cpuinfo, WMI) |
| GPU | NVML (NVIDIA Management Library), CUDA Runtime API |
| Memory | psutil, OS-specific |
| Storage | OS APIs, custom disk benchmark |
| Network | OS APIs, custom network benchmark |
| OS | platform module, OS-specific |
| Container | Docker API, Kubernetes API (if applicable) |

### Hardware Fingerprint

```yaml
machine:
  os: Windows 11 | Ubuntu 24.04 | macOS 15
  architecture: x86_64 | arm64

cpu:
  model: "AMD Ryzen 7 7840HS" | "Intel Xeon 7642" | "Apple M3 Max"
  physical_cores: 8
  logical_cores: 16
  avx2: true
  avx512: false
  numa_nodes: 1
  cache_l3_mb: 32

memory:
  total_gb: 32
  available_gb: 21.3
  bandwidth_gb_s: 89.6

gpu:
  available: true
  vendor: NVIDIA | AMD | Intel | Apple
  model: "RTX 4060" | "A100" | "H100"
  cuda_compute_capability: 8.9
  vram_gb: 8
  free_vram_gb: 6.4
  driver_version: "550.54.15"
  cuda_version: "12.4"
  uvм_available: true | false
  nvlink: false

storage:
  type: NVMe | SSD | HDD
  sequential_read_mb_s: 3200
  sequential_write_mb_s: 2800
  random_read_iops: 500000

network:
  bandwidth_gb_s: 10
  latency_ms: 0.2
  rdma: false

software:
  cuda: "12.4"
  cudf: "24.08"
  polars: "1.0"
  polars_gpu: compatible
  duckdb: "1.0"
  ray: "2.30"
  dask: "2024.8"
  python: "3.12"
  arrow: "17.0"
```

### Microbenchmark Calibration

On installation or first execution, run short microbenchmarks:

**Operations:**
- scan, filter, sort, hash join, groupby, window, string operations
- Parquet decode, CSV decode
- host → GPU transfer, GPU → host transfer
- Arrow IPC serialize/deserialize
- SQL query round-trip

**Sizes:**
- 1 MB, 10 MB, 100 MB, 500 MB, 1 GB, 5 GB

**Storage:**
```json
{
  "groupby": {
    "cpu": {"100MB": 0.18, "1GB": 1.91},
    "gpu": {"100MB": 0.31, "1GB": 0.42}
  },
  "join": {
    "cpu": {"100MB": 0.22, "1GB": 2.10},
    "gpu": {"100MB": 0.35, "1GB": 0.51}
  },
  "h2d_transfer": {
    "100MB": 0.024,
    "1GB": 0.240
  },
  "d2h_transfer": {
    "100MB": 0.016,
    "1GB": 0.160
  }
}
```

**Location:** `~/.adaptive-analytics/hardware-profile.json`

**Scheduler now knows:**
```
100 MB groupby:
  CPU = 180 ms
  GPU = 310 ms
  → CPU

1 GB groupby:
  CPU = 1.91 sec
  GPU = 0.42 sec
  → GPU
```

No LLM. No ML model. Just empirical measurements.

---

## COST MODEL — FULL SPECIFICATION

### Per-Operator Cost

```
Cost(O, E) = T_startup + T_read + T_transfer + T_compute + T_spill + T_materialize
```

Where:
- `T_startup`: Engine initialization (DuckDB connection, Ray task startup, GPU context)
- `T_read`: Data reading (Parquet decode, CSV parse, SQL result fetch)
- `T_transfer`: Data movement (H2D + D2H for GPU, network for distributed)
- `T_compute`: Actual computation (from calibration + cardinality estimation)
- `T_spill`: Disk spill if memory exceeded
- `T_materialize`: Materialization cost (caching, intermediate writes)

### GPU-Specific Cost

```
T_GPU = T_H2D + T_kernel + T_D2H + T_startup
```

**Critical example:**
```
CPU operation = 400 ms
GPU kernel = 80 ms
CPU→GPU = 240 ms
GPU→CPU = 160 ms

GPU total = 480 ms
CPU total = 400 ms
→ Choose CPU
```

Even though GPU computation itself was **5× faster**, the total cost is higher. This is why a serious scheduler is more useful than simply enabling CUDA.

### Segment-Based Optimization

**Do not optimize per-operation. Optimize per-segment.**

**Naive planner (bad):**
```
Filter     GPU
Join       GPU
GroupBy    GPU
UDF        CPU
Sort       GPU
→ CPU → GPU → CPU → GPU (data bouncing)
```

**Segment-optimized planner (good):**
```
Filter     GPU
Join       GPU
GroupBy    GPU
      ↓ materialize once
UDF        CPU
Sort       CPU
```

or even:
```
all CPU
```

depending on size.

### Mathematical Formulation

\[
\min_P \left[ \sum_i ExecutionCost(i, e_i) + \sum_{i,j} TransitionCost(e_i, e_j) \right]
\]

subject to:
- \( Memory(e_i) \le AvailableMemory \)
- \( Capability(i, e_i) = true \)
- \( Privacy(i, e_i) = allowed \)
- \( Hardware(e_i) = compatible \)

Where \( P \) is the physical plan, \( e_i \) is the engine for operation \( i \), and \( TransitionCost \) includes data transfer, serialization, and materialization.

### Extended Cost Function (Research-Level)

\[
P^* = \arg\min_P \left[ T(P) + \alpha M(P) + \beta D(P) + \gamma C(P) \right]
\]

Where:
- \( T \) = execution time
- \( M \) = memory pressure
- \( D \) = data movement
- \( C \) = monetary/energy cost

---

## EXECUTION ENGINES — FULL SPECIFICATION

### Engine Hierarchy

| Priority | Engine | Use Case | Verified Performance |
|---|---|---|---|
| 1 | **Source pushdown** | Avoid moving data entirely | DuckDB: 3.35× speedup with join filter pushdown; 10.3× with mixed-struct Parquet |
| 2 | **DuckDB** | Local SQL, CSV, Parquet, joins, aggregations | Projection/filter pushdown on Parquet |
| 3 | **Polars CPU** | Expressions, lazy execution, parallelism, streaming | Strong default for local analytical workloads |
| 4 | **Polars GPU (RAPIDS)** | Joins, grouped aggregations, large datasets | 3.2× (SF1K, 1 GPU), 23.2× (SF3K, 8 GPUs) |
| 5 | **cudf.pandas** | Drop-in pandas acceleration | Advanced GroupBy 5 GB: 150× speedup; join+groupby 5 GB: 5 min → 1.5 sec |
| 6 | **Pandas** | Compatibility with existing scripts | Millions of existing pandas scripts |
| 7 | **DataFusion** | Rust-native extensible query engine | Reference architecture for planning |
| 8 | **Ray** | Distributed, heterogeneous CPU/GPU | Fractional GPU support (`num_gpus=0.5`) |
| 9 | **Dask** | DataFrame-native distributed | Natural for pandas/NumPy workloads |
| 10 | **Spark RAPIDS** | Large-scale distributed GPU | 100 TB estate |

### Verified Crossover Points

| Workload | CPU | GPU | Winner |
|---|---|---|---|
| Small data (< 50K rows) | 0.051 sec | ~0.13 sec | **CPU ~2.5× faster** |
| S3-backed H3 join (RTX 4000 Ada) | — | — | **CPU 2–4× faster** |
| 100 MB groupby | 180 ms | 310 ms | **CPU** |
| 1 GB groupby | 1.91 sec | 0.42 sec | **GPU** |
| 5 GB advanced groupby | ~5 min | ~1.5 sec | **GPU ~200×** |
| SF1K PDS-H (1 TB) | — | 3.2× | **GPU** |
| SF3K PDS-H (3 TB, 8 GPUs) | — | 23.2× | **GPU** |

### GPU Memory Management

| Mode | Capability | Limitation |
|---|---|---|
| **Standard VRAM** | Fast, direct | Limited to VRAM size |
| **UVM (Unified Virtual Memory)** | Offload to system RAM when VRAM full | Requires compatible driver |
| **Streaming (RapidsMPF)** | Spill to host memory, chunks | Requires compatible version |
| **Chunked processing** | Process in batches | Slower, more overhead |
| **CPU fallback** | Always works | Loses GPU acceleration |

**Verified:** Non-UVM chunked Parquet reader encounters OOM before SF100. UVM or streaming backend is required for larger-than-VRAM datasets.

---

## DATA INTERCHANGE — FULL SPECIFICATION

### Apache Arrow (Foundation)

- **Zero-copy within process** via C Data Interface
- **Arrow IPC** for interprocess communication
- **Arrow Flight SQL** for SQL-oriented RPC over Arrow streams
- **CUDA device memory** via `CudaBuffer` and `CudaHostBuffer`
- **IPC on device**: Read/write Arrow IPC messages from GPU memory

### Verified Arrow Flight SQL Performance

| Interface | Relative Performance |
|---|---|
| JDBC (baseline) | 1× |
| Arrow Flight SQL (Teradata) | **up to 84× faster read throughput** |
| Arrow Flight SQL (Dremio/Spice) | 1,324 ms → 223 ms (**5.9×**) |
| StarRocks Arrow Flight SQL | Order-of-magnitude improvement, transfer efficiency up to 10× |

### Data Flow

```
Excel/CSV/Parquet
      ↓
  Streamed batches
      ↓
    Arrow
      ↓
  ┌───┴───┐
  ▼       ▼
CPU     GPU
RAM     VRAM
  │       │
  └───┬───┘
      ▼
  Arrow IPC / Flight SQL
      ↓
  Result → Excel / Parquet / DB
```

**Never convert:**
```
Pandas → Python objects → JSON → Polars → Python list → cuDF
```

**Always use:**
```
Arrow → Arrow → Arrow → Arrow
```

---

## CONNECTORS — FULL SPECIFICATION

### Excel Connector (First-Class)

**Must understand:**
```
Workbook
 ├─ Sheet
 ├─ Table
 ├─ Named Range
 ├─ Range
 ├─ Formula
 ├─ Value
 └─ Metadata
```

**Must handle:**
- Format fragility (encoding, separators, line endings, header row changes)
- Schema drift
- Hidden rows/columns
- Formulas (evaluate or treat as values)
- Large workbooks (streaming batches)
- Multiple sheets
- Named ranges
- Excel tables

**API:**
```python
sales = ctx.excel(
    "FY26.xlsx",
    sheet="Orders",
    range="A1:P200000",
    header_row=1,
    formula_handling="evaluate",
    streaming=True
)
```

**Output:**
```python
result.write_excel(
    "monthly-report.xlsx",
    sheet="Summary",
    formula_preservation=True
)
```

**Do not:** Stuff Python inside every spreadsheet. Work **around** Excel as a source and destination.

### SQL Connector (Capability-Aware)

**Capabilities per database:**
```yaml
postgresql:
  filter_pushdown: true
  projection_pushdown: true
  aggregate_pushdown: true
  join_pushdown: true
  window_functions: true
  regex: true
  full_text: optional
  arrow_flight: true

mysql:
  filter_pushdown: true
  projection_pushdown: true
  aggregate_pushdown: true
  join_pushdown: true
  window_functions: true
  regex: true
  arrow_flight: partial

sqlite:
  filter_pushdown: true
  projection_pushdown: true
  aggregate_pushdown: true
  join_pushdown: true
  window_functions: partial
  regex: partial
```

**Optimizer decides:**
```
push filter
push columns
push aggregation
perform cross-source join locally
```

**Never:**
```python
pd.read_sql("SELECT * FROM huge_table", conn)  # Bad
```

**Always:**
```python
ctx.sql("huge_table").filter(...).select(...).to_arrow()  # Pushdown
```

### NoSQL Connector (MongoDB First)

**Map operations to MongoDB aggregation pipeline:**
```python
events.filter(col("country") == "India")
```

becomes:
```javascript
{ "$match": { "country": "India" } }
```

**Not:** Download entire collection.

**Supported operations:**
- `$match` (filter)
- `$project` (projection)
- `$group` (aggregation)
- `$sort` (sort)
- `$limit` (limit)

**Partial support:** Complex aggregations may require local compute.

### File Connectors

| Format | Pushdown | Streaming | Notes |
|---|---|---|---|
| **Parquet** | Projection, filter | Yes | DuckDB pushdown verified |
| **CSV** | None | Yes | Schema inference |
| **JSON** | None | Yes | Semi-structured |
| **Avro** | None | Yes | Schema evolution |
| **ORC** | Projection, filter | Yes | Hive-compatible |
| **Delta** | Projection, filter | Yes | Versioned |
| **Iceberg** | Projection, filter | Yes | Versioned |
| **Hudi** | Projection, filter | Yes | Incremental |

### Python UDF Connector

Python UDFs are extremely difficult to optimize. Represent explicitly:
```yaml
PythonUDF:
  function: weird_business_rule
  deterministic: true
  vectorized: false
  gpu: false
  estimated_selectivity: unknown
```

**Initial execution:** Isolated CPU workers.

**Later support:**
- NumPy UDF
- Numba UDF
- CuPy UDF
- Arrow UDF

**User hints:**
```python
@analytics.udf(
    device="cpu",
    deterministic=True,
    vectorized=True
)
def calculate_risk(...):
    ...
```

---

## TYPE SYSTEM — FULL SPECIFICATION

### Canonical Types

```
Bool
Int8 / Int16 / Int32 / Int64
UInt8 / UInt16 / UInt32 / UInt64
Float32 / Float64
Decimal(precision, scale)
Utf8
Binary
Date
Time
Timestamp(unit, timezone)
Duration
List<T>
Struct{field1: T1, field2: T2, ...}
Map<K, V>
Categorical
Null
```

### Type Mapping

| Source | Mapping |
|---|---|
| **PostgreSQL** | `int4` → Int32, `numeric` → Decimal, `timestamptz` → Timestamp(us, UTC) |
| **MongoDB** | `int` → Int64, `double` → Float64, `date` → Timestamp(ms, UTC) |
| **Excel** | Number → Float64, Date → Timestamp(s, local), Text → Utf8 |
| **Pandas** | `int64` → Int64, `float64` → Float64, `datetime64[ns]` → Timestamp(ns, None) |
| **Polars** | `Int64` → Int64, `Float64` → Float64, `Datetime` → Timestamp(us, None) |
| **Arrow** | `int64` → Int64, `timestamp[us]` → Timestamp(us, None) |
| **DuckDB** | `BIGINT` → Int64, `TIMESTAMP` → Timestamp(us, None) |
| **cuDF** | Same as pandas, but GPU-resident |
| **Spark** | `LongType` → Int64, `TimestampType` → Timestamp(us, UTC) |

### Type Reconciliation Layer

**Verified problem:** pandas and Polars produce different payloads for equivalent temporal types. Moving data between engines can silently corrupt temporal columns.

**Solution:** Canonical type normalization at every engine boundary.

```
PostgreSQL → Canonical → DuckDB → Canonical → Polars → Canonical → Excel
```

**Log every type conversion.** Flag mismatches for human review.

---

## METADATA & LINEAGE — FULL SPECIFICATION

### Metadata Travels With Data

```yaml
column:
  name: salary
  type:
    decimal:
      precision: 12
      scale: 2
  classification:
    - confidential
    - financial
  nullable: false
  lineage:
    source:
      system: payroll
      table: employees
      column: annual_salary
```

**When salary travels:**
```
PostgreSQL → DuckDB → Polars → Excel
```

**The `CONFIDENTIAL` tag must remain.**

### Lineage Tracking

- **Column-level lineage** across all engines
- **Transformation lineage** — what operation produced this column
- **Execution lineage** — which run produced this result
- **Version lineage** — what changed between runs

### Tag Synchronization

**Verified problem:** Tag drift across platforms. A "CUSTOMER_DATA" tag in Snowflake doesn't appear in dbt models; Tableau dashboards can't inherit "CONFIDENTIAL" from the warehouse.

**Solution:** Unified metadata layer that synchronizes tags, classifications, and lineage across all connected platforms.

---

## PRIVACY, SECURITY & GOVERNANCE — FULL SPECIFICATION

### Architectural Privacy

**Not:** "We don't call OpenAI."

**Instead:**
```
Default network policy:
DENY outbound
```

**Admin enables only:**
```
postgres.company.local
mongo.internal
s3.internal
```

**Core operation works:**
- Air-gapped
- Offline
- On-prem
- Private cloud

### Zero External AI/LLM Dependency

- No external LLM APIs
- No third-party cloud AI
- No telemetry unless explicitly opted in
- No data leaving customer-controlled boundary
- Optional local ML only if explicitly enabled by admin
- Local ML must be fully private, explainable, and **off by default**

### Access Control

- **RBAC** (Role-Based Access Control)
- **ABAC** (Attribute-Based Access Control)
- **Row-level security** — filter rows based on user attributes
- **Column-level security** — mask or hide columns based on user attributes
- **Secrets management** — encrypted storage, key rotation
- **Audit logging** — every data access, query, and resource usage

### Governance Embedded in Workflow

**Verified problem:** Restrictive governance stifles innovation. When every data request requires approval workflows and manual provisioning, data teams become bottlenecks. Analysts spend weeks waiting for access instead of generating insights.

**Solution:** Governance is embedded in the workflow, not bolted on. Lineage, validation, access control are automatic. Using the system is easier than bypassing it.

### Compliance

- **GDPR** alignment
- **HIPAA** alignment
- **CCPA** alignment
- **SOC 2** alignment
- **Air-gapped update and license management**
- **Audit trail** for all access, queries, and resource usage

---

## SCHEDULER & RESOURCE MANAGER — FULL SPECIFICATION

### Scheduler Responsibilities

1. **Profile the task** — what operation is being performed?
2. **Profile the data** — size, schema, format, location, type distribution
3. **Profile the hardware** — CPU cores, GPU VRAM, memory bandwidth, storage I/O
4. **Estimate crossover** — will GPU provide at least 2× speedup within energy budget?
5. **Dispatch** — execute on optimal resource; fall back gracefully
6. **Log and learn** — record decision, outcome, and explanation

### Scheduling Granularity

- **Per segment of the DAG**, not per operation
- **Per task**, not per pipeline
- **Per engine**, not per hardware

### Resource Management

| Resource | Management |
|---|---|
| **CPU** | Thread count, BLAS threads, Polars threads, DuckDB threads |
| **GPU** | VRAM, CUDA context, UVM, streaming, fractional allocation |
| **Memory** | Total, available, per-engine limits, spill thresholds |
| **Storage** | I/O bandwidth, cache, spill |
| **Network** | Bandwidth, latency, Arrow Flight SQL |

### Ray-Specific Considerations

- Ray exposes CPU/GPU/memory resources
- **Limitation:** Ray logical CPU declarations are scheduling/admission info, not physical CPU isolation
- **AAR must control:** thread count, BLAS threads, Polars threads, DuckDB threads, memory limits, worker concurrency
- **Warning:** Avoid fractional `num_gpus` for model actors; give each actor a whole GPU and batch requests
- For analytics (short-lived operations), fractional GPU may be acceptable if operations don't hold GPU locks during initialization

### Adaptive Optimization

**Keep historical records:**
```
operator_hash
engine
rows
bytes
columns
datatypes
cardinality
hardware_id
elapsed_ms
peak_memory
bytes_transferred
success
```

**Use for optimization:**
- Lookup table
- Interpolation
- EWMA (Exponentially Weighted Moving Average)
- Linear regression
- Piecewise models

**This is not an LLM. It is entirely local and deterministic.**

---

## FAILURE MODES & GRACEFUL DEGRADATION — FULL SPECIFICATION

| # | Failure Mode | Detection | Handling | Fallback | Log |
|---|---|---|---|---|---|
| 1 | **GPU unavailable** | NVML/CUDA runtime check | Skip GPU path | CPU engine | "GPU not detected" |
| 2 | **GPU OOM (VRAM)** | cuDF memory error | Enable UVM/streaming | CPU engine | "VRAM exceeded; UVM enabled" |
| 3 | **GPU OOM (UVM)** | Persistent OOM | Chunked processing | Distributed or CPU | "UVM OOM; chunk size X" |
| 4 | **Unsupported GPU op** | Capability registry | Fallback per-op | CPU for that op | "Op X unsupported on GPU" |
| 5 | **Schema drift** | Schema comparison | Alert + pause | Offer repair | "Schema change in source Y" |
| 6 | **Type mismatch across engines** | Canonical type check | Normalize at boundary | Flag for review | "Type mismatch: timestamp" |
| 7 | **Network partition** | Connection timeout | Queue + retry | Local cache | "Source X unreachable" |
| 8 | **License exhaustion** | License check | Queue | Notify admin | "GPU license limit reached" |
| 9 | **Source unavailable** | Connection check | Retry with backoff | Alert after threshold | "Source X unavailable" |
| 10 | **Cardinality estimation error** | Runtime statistics | Adaptive reoptimization | Re-plan | "Cardinality off by X%" |
| 11 | **Excel format change** | Header/encoding check | Detect + alert | Offer repair | "Excel format changed" |
| 12 | **Excel formula error** | Formula evaluation | Fallback to value | Flag for review | "Formula error in cell X" |
| 13 | **Python UDF failure** | Exception catch | Retry + isolate | Skip with alert | "UDF failed: exception" |
| 14 | **Arrow IPC failure** | Serialization error | Fallback to JSON | Flag for review | "Arrow IPC failed" |
| 15 | **Distributed worker failure** | Ray/Dask task error | Retry + reassign | Local execution | "Worker X failed" |
| 16 | **Memory spill** | Memory threshold | Spill to disk | Alert if excessive | "Spilled X GB to disk" |
| 17 | **Cache invalidation** | Source change detection | Invalidate cache | Re-execute | "Cache invalidated for X" |
| 18 | **Concurrent access conflict** | Lock detection | Queue + retry | Notify user | "Resource locked by X" |
| 19 | **Data quality failure** | Validation check | Pause + alert | Offer repair | "Quality check failed: X" |
| 20 | **Privacy violation** | Policy check | Block + alert | Notify admin | "Privacy violation: X" |

**Critical requirement:** The system must **never silently fail**. Every fallback, every degradation, every optimization decision must be logged with an explanation. This is non-negotiable for analyst trust.

---

## EXPLAINABILITY & OBSERVABILITY — FULL SPECIFICATION

### Plan Explanation

```bash
aar explain plan pipeline.py
```

**Output:**
```
PLAN

[1] PostgreSQL Scan
    pushed WHERE date >= '2026-01-01'
    pushed projection:
        customer_id
        amount
        region

    reason:
        reduces transfer by estimated 91.2%

[2] Excel Scan
    engine: CPU Arrow reader

    reason:
        GPU Excel reader unavailable

[3] Join
    engine: Polars GPU

    reason:
        9.7GB working set
        join supported on GPU
        estimated 3.1x acceleration

[4] Python UDF
    engine: CPU worker

    reason:
        arbitrary Python UDF unsupported on GPU

[5] Export
    engine: Excel writer
```

### Decision Logging

Every decision logged with:
- **Engine selected**
- **Reason** (capability, cost, privacy, hardware)
- **Estimated time**
- **Estimated peak memory**
- **Fallback**
- **Expected saving**
- **Actual time** (after execution)
- **Actual peak memory**
- **Success/failure**

### Execution Trace

```yaml
execution_id: uuid
pipeline: pipeline.py
started_at: 2026-09-27T10:00:00Z
finished_at: 2026-09-27T10:00:04Z

operators:
  - id: op1
    type: ScanPostgres
    engine: postgresql
    pushdown: filter, projection
    estimated_ms: 120
    actual_ms: 118
    rows_returned: 50000
    bytes_transferred: 2400000

  - id: op2
    type: ScanExcel
    engine: arrow_cpu
    estimated_ms: 80
    actual_ms: 85
    rows_returned: 200000

  - id: op3
    type: Join
    engine: polars_gpu
    estimated_ms: 310
    actual_ms: 298
    gpu_memory_used_mb: 6400
    h2d_transfer_ms: 24
    d2h_transfer_ms: 16
    reason: "9.7GB working set, join supported on GPU, 3.1x acceleration"

  - id: op4
    type: PythonUDF
    engine: cpu_worker
    estimated_ms: 1500
    actual_ms: 1420
    reason: "arbitrary Python UDF unsupported on GPU"

  - id: op5
    type: WriteExcel
    engine: excel_writer
    estimated_ms: 200
    actual_ms: 210
```

### Metrics

- **Time-to-insight** per pipeline
- **Resource utilization** (CPU%, GPU%, memory, disk)
- **Cost per query** (monetary, energy)
- **Cache hit rate**
- **Pushdown rate**
- **Fallback rate**
- **Explainability score** (user understanding)

---

## UI/UX — ANALYST WORKBENCH — FULL SPECIFICATION

### Design Principles

- **Not a Power BI clone**
- **Analyst Workbench** — for analytical work, not dashboard consumption
- **Customizable** — dark/light/system, density, font, layout
- **Keyboard-first** — for power users
- **Explainable** — every decision visible
- **Dense** — for sessions > 30 minutes
- **Multi-panel** — for comparison

### Layout

```
┌───────────────────────────────────────────────────────────────────┐
│ Project       Connections      Runs        Resources      Help    │
├───────────────┬───────────────────────────┬───────────────────────┤
│ Sources       │                           │ Plan Inspector        │
│               │      SQL / Python         │                       │
│ Excel         │      Workspace            │ CPU:  ████████░░ 80%  │
│ PostgreSQL    │                           │ GPU:  ████░░░░░░ 40%  │
│ MongoDB       │                           │ Mem:  ██████░░░░ 60%  │
│ Parquet       │                           │ Pushdowns: 3          │
│ CSV           │                           │ Fallbacks: 1          │
│               │                           │                       │
├───────────────┴───────────────────────────┴───────────────────────┤
│ Data Preview | Quality | Schema | Lineage | Execution Log | Explain│
└───────────────────────────────────────────────────────────────────┘
```

### Customization

| Setting | Options |
|---|---|
| **Theme** | Dark, Light, System |
| **Density** | Compact, Comfortable, Spacious |
| **Font** | Family, Size, Line height |
| **Layout** | Drag, Resize, Dock panels |
| **Keyboard** | Fully navigable, customizable shortcuts |
| **Colors** | Per-app-mode theming, color adjustment in dark mode |

### Explainability Panel

Every automated decision shown in a dedicated panel:
- **Why CPU was chosen over GPU** (or vice versa)
- **What data was cached and why**
- **How a transformation was optimized**
- **What fallbacks occurred and why**

### Trust

Research shows users resist tools when they cannot understand how results are generated — "lack of trust in a black box" is a primary adoption barrier.

**The explainability panel is not a feature — it is a requirement for adoption.**

---

## DEPLOYMENT MODELS — FULL SPECIFICATION

### Single-Node

- **Target:** Analyst laptop or workstation
- **Hardware:** CPU + optional GPU
- **Engines:** DuckDB, Polars CPU/GPU, pandas, cudf.pandas
- **Storage:** Local filesystem
- **Network:** None required (air-gapped capable)

### Cluster

- **Target:** Team or department
- **Hardware:** Multiple CPU/GPU nodes
- **Engines:** Ray, Dask, Spark RAPIDS
- **Storage:** Shared filesystem or object store
- **Network:** Internal only

### Air-Gapped

- **Target:** High-security environments
- **Hardware:** On-prem only
- **Engines:** All local
- **Storage:** Local or SAN
- **Network:** None outbound
- **Updates:** Manual, signed packages

### Hybrid

- **Target:** Enterprise with mixed requirements
- **Hardware:** On-prem + private cloud
- **Engines:** All
- **Storage:** Mixed
- **Network:** Controlled, admin-defined egress

---

## SUCCESS METRICS — FULL SPECIFICATION

| Metric | Target | Measurement |
|---|---|---|
| **Analyst productivity** | Reduce 78% overhead to < 40% | Time-to-insight per pipeline |
| **Tool switching** | Reduce 5.4 platforms to 1-2 | Daily tool count |
| **Resource utilization** | > 70% CPU, > 50% GPU | Runtime metrics |
| **Cost per query** | Reduce 50% | Monetary + energy |
| **Time-to-insight** | Reduce 60% | Start-to-result |
| **Compliance audit** | Zero findings | Audit results |
| **Adoption rate** | > 80% of analysts | Active users |
| **Explainability score** | > 90% understand decisions | User survey |
| **Shadow IT reduction** | 54% → < 10% | External AI usage |
| **Validation burden** | 6 hours/week → < 1 hour | Time tracking |
| **Pipeline reproducibility** | 100% | Re-run success rate |
| **Schema drift handling** | Zero silent failures | Alert count |

---

## NON-GOALS

- Do not build a Power BI clone
- Do not require external LLM APIs or cloud AI services
- Do not assume perfect hardware, clean data, or unlimited resources
- Do not require internet connectivity for core operation
- Do not force AI/ML into the workflow
- Do not force analysts to abandon familiar tools (Excel, SQL, Python)
- Do not start with Kubernetes, Spark, Trino, Airflow, Dagster, Prefect, 50 connectors, SaaS, LLM, dashboards, natural-language SQL, multi-node GPU, data lakehouse, enterprise RBAC, or full browser IDE
- Do not chase "GPU everywhere"
- Do not make Substrait the core IR
- Do not optimize per-operation (optimize per-segment)
- Do not assume GPU acceleration is always beneficial
- Do not silently fail

---

## CONSTRAINTS

- Respect real-world limits but aim beyond current systems
- Prefer local, deterministic, explainable automation
- No external AI/LLM dependency
- Output must be practical, buildable, and complete
- If critical information is missing, state assumptions explicitly
- Every recommendation must be traceable to empirical evidence or verified production behavior
- Where data is unavailable or contradictory, state that explicitly
- The system must work air-gapped, offline, on-prem, private cloud
- The system must never silently fail
- The system must be explainable at every decision point
- The system must meet analysts where they are (Excel, SQL, Python)
- The system must be open and extensible (plugins, open formats, no vendor lock-in)
- The system must be hardware-aware (CPU/GPU/memory/storage/network)
- The system must optimize for minimum total analytical cost, not GPU utilization

---

## OUTPUT FORMAT

Produce a complete, topmost-level system specification containing:

1. **Vision statement** — one paragraph, one sentence
2. **Key assumptions** — explicit, numbered
3. **Full system architecture** — component diagram, data flow, layer descriptions
4. **Internal Analytics IR** — node types, metadata, Substrait adapters
5. **Hardware Profiler** — detection APIs, fingerprint schema, microbenchmark calibration
6. **Cost Model** — per-operator, GPU-specific, segment-based optimization, mathematical formulation
7. **Adaptive Planner** — segment-based optimization, priority order, decision output
8. **Execution Engines** — full hierarchy, verified crossover points, GPU memory management
9. **Data Interchange** — Arrow, Flight SQL, CUDA device memory, verified performance
10. **Connectors** — Excel, SQL, NoSQL, files, Python UDF, capability matrices
11. **Type System** — canonical types, type mapping, reconciliation layer
12. **Metadata & Lineage** — column-level lineage, tag synchronization, classification propagation
13. **Privacy, Security & Governance** — architectural privacy, zero external AI, access control, compliance
14. **Scheduler & Resource Manager** — responsibilities, granularity, resource management, adaptive optimization
15. **Failure Modes** — 20+ failure modes with detection, handling, fallback, logging
16. **Explainability & Observability** — plan explanation, decision logging, execution trace, metrics
17. **UI/UX** — Analyst Workbench, customization, explainability panel
18. **Deployment Models** — single-node, cluster, air-gapped, hybrid
19. **Success Metrics** — productivity, utilization, cost, compliance, adoption
20. **Non-Goals** — explicit list
21. **Constraints** — explicit list
22. **Benchmark Appendix** — verified performance curves, crossover points, source citations
23. **Failure Mode Matrix** — complete table
24. **Glossary** — terms, acronyms, definitions

The output must be comprehensive, practical, buildable, and grounded in verified evidence. It must not be a marketing document. It must be an engineering specification.

---

## FINAL STATEMENT

The Adaptive Analytics Runtime is not a BI tool. It is not an AI assistant. It is not an ETL orchestrator. It is not a replacement for Excel, SQL, Python, or big data engines.

It is an **analytics operating layer** — a unified, privacy-first, hardware-aware orchestration system that:

- Unifies Excel, SQL, NoSQL, Python, local files, and big data behind one control plane
- Automatically selects the most efficient legal execution path for the entire analytical DAG
- Optimizes per segment, not per operation, minimizing total execution cost plus transition cost
- Uses empirical microbenchmark calibration, not hardcoded rules
- Works air-gapped, offline, on-prem, private cloud with zero external AI dependency
- Logs every decision with rationale — explainable, auditable, trustworthy
- Meets analysts where they are — Excel-first, SQL-first, Python-first
- Reduces the 78% overhead to < 40%, giving analysts back their time
- Embeds governance in the workflow, making the secure path the easy path
- Learns from historical execution without LLMs, without ML, without external dependencies

**The goal is minimum total analytical cost — not GPU utilization.**

The system should proudly say "CPU selected" or "PostgreSQL selected" when that is optimal.

**This is the beginning of an analytics operating layer, not another analytics application.**