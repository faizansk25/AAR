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
`polars_gpu` rows are marked `skipped`, because both engines are now
implemented but *decline to execute* on a host with no CUDA device - which
is the honest answer, and a different one from "no engine exists".

## What a GPU run would and would not prove

A GPU run now exercises real code: `aar/engines/cudf_engine.py` and
`aar/engines/polars_gpu_engine.py` exist, and `tools/gpu_verification.py`
constructs them with `allow_degradation=False` so that a row filed under
"cudf" must have been produced by cudf.

Be careful about what a T4 establishes even then. It is compute capability
7.5: **no bfloat16**, and FP64 at 1/64 of FP32. A workload that wins on a
T4 can lose on an A100. A successful run demonstrates that the path *works*
and that the planner chooses sensibly; it does not make the numbers portable
to other hardware, and it does not calibrate the cost model.
