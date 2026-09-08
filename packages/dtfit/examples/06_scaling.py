"""Scaling out -- parallel fits and the streaming image accumulator.

Per-problem independence makes dtfit embarrassingly parallel. Two tools:

- fit_many (+ FittingProblem)  -- fan independent fits across CPU cores.
- ImageStream -- accumulate a signal's image in fixed O(order) memory as it
  arrives; merge() reduces per-shard accumulators, so a stream (or a
  partitioned dataset) too big for memory still fits in one pass.

Run headless:   python examples/06_scaling.py
"""

import numpy as np

from dtfit import fit_many, ImageStream, fit
from dtfit.image import FittingProblem


def parallel_fits(rng) -> None:
    # Each FittingProblem is a self-contained, picklable spec. A failed fit is
    # captured as .error rather than aborting the batch (use n_jobs=-1 for all
    # cores). fit_many returns a FittingResult per problem, tagged with .label.
    problems = []
    for i, b in enumerate([0.4, 0.6, 0.8, 1.0, 1.2]):
        x = np.linspace(0, 3, 200)
        y = (1 + 0.2 * i) * np.exp(b * x) + rng.normal(0, 0.05, x.size)
        problems.append(FittingProblem(
            x=x, y=y, expr="a*exp(b*t)", var="t",
            method="lsi", kwargs={"p0": [1.0, 1.0]}, label="ch{}".format(i)))
    print("== fit_many: independent fits ==")
    for r in fit_many(problems, n_jobs=1):
        msg = r.error if r.error else "coeffs={}".format(np.round(r.coeffs, 3))
        print("  {}: {}".format(r.label, msg))


def one_pass(rng) -> None:
    # Fold chunks of a stream into an additive image accumulator, then fit
    # once. Consecutive update() calls are exactly additive (equal to a
    # single whole-domain image).
    acc = ImageStream("legendre", 6, domain=(0, 5))
    for x_chunk in np.array_split(np.linspace(0, 5, 5000), 10):
        y_chunk = (1.3 * np.exp(0.7 * x_chunk)
                   + rng.normal(0, 0.05, x_chunk.size))
        acc.update(x_chunk, y_chunk)
    res = fit("a*exp(b*t)", acc.image(), "t", p0=[1.0, 1.0])
    print("\n== ImageStream: one pass, fixed memory ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()})


def map_reduce(rng) -> None:
    # Workers each accumulate over a contiguous shard, then the partials are
    # reduced with merge() -- the distributed estimator.
    def shard(x_shard):
        a = ImageStream("legendre", 6, domain=(0, 5))
        y = 1.3 * np.exp(0.7 * x_shard) + rng.normal(0, 0.05, x_shard.size)
        a.update(x_shard, y)
        return a

    shards = np.array_split(np.linspace(0, 5, 5000), 4)
    partials = [shard(s) for s in shards]  # the "map" (parallelizable)
    reduced = partials[0]
    for a in partials[1:]:
        reduced = reduced.merge(a)          # the "reduce"
    res = fit("a*exp(b*t)", reduced.image(), "t", p0=[1.0, 1.0])
    print("\n== map-reduce with merge() ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()})


def main() -> None:
    rng = np.random.default_rng(0)
    parallel_fits(rng)
    one_pass(rng)
    map_reduce(rng)


if __name__ == "__main__":
    main()
