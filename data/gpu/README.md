# GPU evidence

GPU results are kept here so that a claim about GPU behaviour can be
traced to the machine that produced it. A number in a report with no file
behind it is a rumour.

## `cpu_baseline.json`

The **CPU-only** reference run, produced on a Windows host with no CUDA
device. Establishes:

* AAR runs correctly with no GPU, degrading rather than failing.
* Three independent CPU engines (Arrow, DuckDB, Polars) produce **exactly
  identical** group-by checksums on 2,000,000 rows - the cross-engine
  correctness claim does not depend on having a GPU.
* The cost model already prefers CPU over GPU in all nine sampled
  size/parallelism combinations, which is the correct answer for a machine
  with no GPU at all.

It is deliberately *not* evidence about GPU performance. The `cudf` and
`polars_gpu` entries in its `benchmark` block are marked `skipped` with the
engine that actually ran instead - there was none. `cudf` and `polars_gpu`
are declared in the capability registry but have no implementation class, so
`create_engine` can only hand back a CPU fallback. See `report.md` 8.2.

## What a GPU run would and would not prove

**A GPU run today would prove almost nothing about cudf.** It would confirm
that AAR's *detection*, *cost model* and *degradation* logic behave
correctly on real hardware - genuinely worth having - but it will not
produce a cudf benchmark until a `CudfEngine` exists.

Run `tools/gpu_verification.py` (or `notebooks/aar_gpu_verification.ipynb`
on Colab with a T4) and save the result here as `t4.json`.

Be careful about what a T4 establishes even then. It is compute capability
7.5: **no bfloat16**, and FP64 at 1/64 of FP32. A workload that wins on a
T4 can lose on an A100.
