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

**cuDF is verified end to end.** `t4_run3.json` records a real group-by
(613.8 ms) and a real filter (80.1 ms) on the device, both agreeing with
every CPU engine on the same data. `polars_gpu` does not: Polars' own GPU
backend rejects the grouped plan, so that engine declines with a recorded
reason and the executor falls back.

> **The run-3 artifact is not in this directory.** Only `t4_run1.json` and
> `cpu_baseline.json` are committed. The figures above are a transcript of a
> Colab run whose JSON was not saved back to the repository, so today a
> reader can verify the CPU baseline and run 1 from these files and cannot
> verify run 3 from anything here. That is exactly the state this file
> exists to prevent — a number in a report with no file behind it is a
> rumour — so it is stated here rather than left for someone to discover.
> The fix is to re-run `tools/gpu_verification.py` on a T4 and commit the
> output, not to keep the claim and lose the evidence.

**The GPU lost, twice.** cuDF took 364 ms and 613.8 ms against DuckDB's
53.7 ms and 293.6 ms on 2M rows / 512 groups. This is the specification's
counter-example measured rather than asserted: 2M x 2 columns is about
32 MB, and PCIe transfer dominates a group-by that small. A GPU wins only
when the compute is large enough to hide the move.

**Read section 5 of any run with care.** A Colab T4 is a shared VM, and
the run-to-run spread is large - DuckDB's group-by was 53.7 ms in one run
and 293.6 ms in the next, a 5.5x swing on identical code and data. One
measurement is an anecdote. The *direction* is stable across runs (the GPU
loses by 2-7x on this workload); the individual numbers are not. Do not
quote a figure from a single run as if it were a benchmark.

**What a T4 cannot settle.** Compute capability 7.5: no bfloat16, FP64 at
1/64 of FP32. A workload that wins on a T4 can lose on an A100. And one
GPU is one data point - the cost model is not calibrated, and calibration
needs several machines.
