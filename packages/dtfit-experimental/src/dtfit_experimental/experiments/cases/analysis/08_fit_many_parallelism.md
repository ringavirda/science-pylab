# `fit_many` — independent-fit fan-out (process / thread backends)

**Verdict: MIXED — an honest negative for fine-grained fits.** On this platform,
process-pool fan-out of millisecond-scale fits runs *slower* than serial because
spawn + pickle + per-fit setup costs dwarf the work. It pays off only for
coarse/heavy tasks or via the threading backend on GIL-released kernels.

Source: [`../../../../../../dtfit/src/dtfit/scale/_parallel.py`](../../../../../../dtfit/src/dtfit/scale/_parallel.py).
Tested in: [Parallel scaling (7)](../07_parallel_scaling/07_parallel_scaling.ipynb),
used as orchestration in [Noise (3)](../03_noise_robustness/03_noise_robustness.ipynb).

## What it is

`fit_many(problems, n_jobs, backend)` fans a list of independent fit problems
(`FittingProblem(x, y, expr, var, method, kwargs)`) across workers via joblib —
either **loky** (true processes, Windows-spawn-safe because it ships picklable
expr *strings* and arrays, not lambdified callables) or **threading** (which
benefits from the GIL-released kernels).

## Measured results

A batch of independent EAC fits across loky workers (Exp 7):

| workers P | time (s) | speedup |
|---|---|---|
| 1 | 0.82 | 1.00 |
| 2 | 2.14 | 0.38 |
| 4 | 1.83 | 0.45 |
| 8 | 1.96 | 0.42 |
| 16 | 2.24 | 0.37 |

→ **Slower than serial** for these fine-grained fits (≤0.45×).

In contrast, in the noise sweep (Exp 3) `fit_many` was used to fan a **grid of
fits** across cores and was a useful orchestration tool — because there the per-
task work is larger and the batch is big.

## Why it loses on fine-grained fits (the cost model)

Each fit here is ~milliseconds. The per-task overhead the process pool adds is
*larger* than that:

1. **Process spawn (Windows).** loky uses `spawn`, which starts a fresh Python
   interpreter and re-imports the world per worker. That fixed cost is amortized
   only over long-lived, heavy tasks.

2. **Pickling round-trips.** Every problem's arrays and the result must be
   serialized to the worker and back. For small fits the serialization can cost as
   much as the fit.

3. **Per-fit SymPy `lambdify`.** Each worker re-`lambdify`s the model expression
   (the price of shipping picklable strings instead of callables). For a
   millisecond fit, building the callable dominates.

Add these up and `T_parallel ≈ T_serial/P + overhead·P`, where `overhead` per task
exceeds the per-fit work — so more workers make it *worse*, which is exactly the
monotone decline in the table. **This is a platform/granularity ceiling, not an
algorithmic flaw**: the fits are embarrassingly parallel; they are just too small
to pay the process tax.

## When it actually helps

- **Coarse/heavy tasks:** each fit takes ≫ the spawn+pickle+lambdify overhead
  (large N, expensive global-search bounds, many channels per task). Batch small
  fits into fewer, bigger tasks.
- **The threading backend** for kernel-bound work: threads have no spawn/pickle
  cost and the GIL-released kernels ([07_gil_released_kernels.md](07_gil_released_kernels.md))
  run concurrently — that is the path that scales for dtfit's hot loops.
- **Reuse the pool** across many batches so the spawn cost amortizes.

## The takeaway for dtfit's parallel strategy

The suite's three parallel paths rank cleanly by overhead:
**threaded GIL-released kernels (9.3×) > threaded map-reduce (2.1×, memory-bound)
≫ fine-grained process fan-out (≤1×).** The lesson is to keep parallelism
**in-process** and **coarse**: parallelize the *kernels* and the *reduce*, not
thousands of tiny independent `lambdify`-bearing fits.

## Related

- The path that *does* scale: [07_gil_released_kernels.md](07_gil_released_kernels.md).
- The exact distributed estimator it should fan instead of tiny fits:
  [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md).
