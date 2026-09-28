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
engine that actually ran instead, because `create_engine` degrades silently
unless a failure ledger is passed.

## What a GPU run should add

Run `tools/gpu_verification.py` (or `notebooks/aar_gpu_verification.ipynb`
on Colab with a T4) and save the result here as `t4.json`, then record what
it actually shows.

Be careful about what a T4 establishes. It is compute capability 7.5: **no
bfloat16**, and FP64 at 1/64 of FP32. A workload that wins on a T4 can
lose on an A100. A successful run demonstrates that the GPU path *works*
and that the planner chooses sensibly; it does not make the numbers
portable to other hardware.
