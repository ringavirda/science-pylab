# API: streams and scale

Run the additive image at scale: accumulate a stream in fixed memory, split a
dataset across workers and merge, hold many channels on a shared grid, or fan
many independent fits across cores. Pick by your situation
([../guides/choosing-a-method.md Sec.3](Guides-Choosing-a-Method)):

| situation | tool |
|---|---|
| accumulate a stream in **fixed memory** | [`ImageStream`](#imagestream) accumulator mode |
| split a dataset **across workers**, then merge | [`ImageStream.merge`](#merge) |
| many **channels on a shared grid** | [`ImageStream`](#imagestream) channels |
| **block** streams with retention, assembly and drift flags | [`ImageStream`](#imagestream) block mode |
| many **independent** fits | [`fit_many`](#fit_many) |

---

<a name="fit_many"></a>
## `fit_many`

```python
fit_many(problems, *, n_jobs=-1, backend="loky", verbose=0) -> list[FittingResult]
```

Fit many **independent** problems in parallel (different series and/or models).

| arg | default | meaning |
|---|---|---|
| `problems` | -- | a sequence of [`FittingProblem`](#fittingproblem) specs |
| `n_jobs` | `-1` | workers (`-1` = all cores; `1` = serial, no pool) |
| `backend` | `"loky"` | `"loky"` (processes), `"threading"` (rides GIL-released kernels), or `"multiprocessing"` |
| `verbose` | `0` | forwarded to `joblib.Parallel` |

Returns [`FittingResult`](API-Types) objects **in input order**, each carrying
`.label` and `.error`. A failed problem has its `error` set (and empty `coeffs`)
rather than aborting the batch.

```python
from dtfit import fit_many
from dtfit.image import FittingProblem
problems = [FittingProblem(x, y, "a*exp(b*t)", "t", label=name)
            for name, (x, y) in series.items()]
results = fit_many(problems, n_jobs=-1)
for r in results:
    print(r.label, r.error or r.coeffs)
```

<a name="fittingproblem"></a>
### `FittingProblem`

A picklable spec for one fit (a dataclass):

| field | default | meaning |
|---|---|---|
| `x`, `y` | -- | observed samples |
| `expr`, `var` | -- | model and main variable |
| `method` | `"lsi"` | `"lsi"` or `"eac"` |
| `kwargs` | `{}` | method-specific keywords (`p0`, `bounds`, ...) |
| `label` | `None` | tag carried through to the result (channel name, etc.) |

### Return type

`fit_many` returns plain [`FittingResult`](API-Types) objects -- batch and
single fits share the **same** type. `FittingResult` is itself picklable (it
drops its lazily-built callable on pickling and rebuilds it from `expr`/`coeffs`
on the caller side), so it survives a process-pool round trip and carries:

- `coeffs`, `expr`, `var`, `cov`, `label`, `error` (set instead of `coeffs` when
  the fit raised).
- `model` -- the fitted callable, rebuilt lazily from `expr`/`coeffs`.
- `predict(x) -> ndarray` -- evaluate the model (broadcasts scalars).

Because they are real `FittingResult`s, the uncertainty helpers (`stderr`,
`confidence_intervals`, prediction bands) are available when a covariance was
produced.

---

<a name="imagestream"></a>
## `ImageStream`

```python
ImageStream(basis, order=None, *, domain=None, block=None, channels=1,
            grid="uniform", keep_fine=64, fold=16, backend="numpy",
            detect=None)
```

A running image over a fixed domain, block images with local domains, or a
channel batch over one shared Gram -- three uses of the same running
statistic `S = Phi^T w y`, `G = Phi^T diag(w) Phi`.

| arg | default | meaning |
|---|---|---|
| `basis` | -- | `"legendre"`, `"block"` or a `Basis` instance |
| `order` | `None` | basis order, at least 1 (a `Basis` instance carries its own) |
| `domain` | `None` | `(x0, x1)` the images are taken over; required unless `block` gives a domain length |
| `block` | `None` | `None` for the accumulator; an `int` for blocks of that many samples, at least `order + 2` (not with `basis="block"`, whose count-block domain never aligns with the hull windows retention and assembly need); a `float` for blocks of that domain length, counted from the domain's start. A length block that closes with fewer than `order + 2` samples is dropped -- no image, `dropped_` counts it. Block mode needs `channels == 1` |
| `channels` | `1` | number of signals sharing the sample positions; `update`'s `y` then has shape `(n, channels)` |
| `grid` | `"uniform"` | `"uniform"` tracks positions as endpoints and spacing and needs evenly spaced, increasing chunks that continue each other; `"explicit"` keeps every position (and weights) |
| `keep_fine`, `fold` | `64`, `16` | block retention, each at least 1: once there are more than `keep_fine` fine blocks, the oldest `fold` are folded into one coarse block at the stream's order; when `fold > keep_fine` the peak retained count is `max(keep_fine, fold)` |
| `backend` | `"numpy"` | `"numpy"`, `"cupy"` or `"torch"` for the `S` projection; the Gram update stays host numpy either way, accumulation is float64 regardless of backend |
| `detect` | `None` | block-level drift detection, block mode only: `None` for none, `"previous"` to compare each finished block to the one before it, or `(model, params)` / `(model, params, var)` to compare it to that model's image |

Raises `ValueError` on a missing domain, `x1 <= x0`, `order < 1`,
`channels < 1`, an unknown grid, basis or backend name, `keep_fine < 1` or
`fold < 1`; in block mode, `channels != 1`, a `block` below `order + 2` samples
or non-positive length, `basis="block"` with a count block, or an
unrecognised `detect`; `detect` given without `block`.

### Accumulator mode

`block=None`: `update(x, y, w=None)` folds one chunk into the running
statistic, `image()` reads it off, and `fit(model, image, var)` takes it
directly.

```python
acc = ImageStream("legendre", 6, domain=(0, 4))
for x_chunk, y_chunk in stream:
    acc.update(x_chunk, y_chunk)
res = fit("a0 + a1*exp(a2*x)", acc.image(), "x")
print({k: round(v, 3) for k, v in res.params.items()})
```

`n` is the sample count accumulated so far (always 0 in block mode, where
samples live in the finished blocks instead).

<a name="merge"></a>
### `merge`

`merge(other) -> ImageStream` returns a new stream holding both sample sets.
Both streams need the same configuration; uniform streams must be
contiguous (one starts one spacing after the other ends). This is the
distributed step: each worker accumulates over its own shard, and the
partial streams are reduced by repeated `merge`.

### `checkpoint` and `resume`

`checkpoint() -> dict` is the complete state as JSON-serializable data.
`resume(state) -> self` loads a checkpoint into a freshly constructed stream
built with the same arguments and continues from it exactly -- discarding
whatever samples that stream already held. A checkpoint from
`detect=(model, params[, var])` needs the same `detect` argument again on
resume; only the accumulated detector state travels with the checkpoint.

### Block mode

`block` a sample count or a domain length: `update` returns the block
images finished by this chunk (a sample on a block's upper edge belongs to
the next block, the domain's end to the last), `close()` finishes a partial
block, `blocks(t0, t1)` lists the stored block images inside a range, and
`assemble(t0, t1, order=None)` merges the whole blocks inside `[t0, t1]`
onto the hull of their domains through the transfer.

```python
blk = ImageStream("legendre", 3, domain=(0, 4), block=1.0)
for x_chunk, y_chunk in stream:
    for img in blk.update(x_chunk, y_chunk):
        print("finished block:", img.n, "samples", img.domain)
blk.close()
print("stored blocks:", len(blk.blocks(0, 4)))
whole = blk.assemble(0, 4)
print("assembled:", whole.n, "samples over", whole.domain)
```

### Channels

`channels=B`: projections `(B, K+1)` over one shared Gram, one GEMM per
chunk through the `numpy`, `cupy` or `torch` backend. `update(x, Y)` takes
`Y` of shape `(n, B)`; `image(channel)` reads one channel's running image,
`images()` all of them.

```python
ch = ImageStream("legendre", 4, domain=(0, 4), channels=Y.shape[1])
ch.update(x, Y)
for img in ch.images():
    print(img.S[:2])
```

### `detect`

`detect=(model, params)` or `detect="previous"` runs a `DriftDetector` on
successive block images and records flagged boundaries as
`(block_index, domain)` pairs in `flags_`.

```python
det = ImageStream("legendre", 3, domain=(0, 4), block=1.0, detect="previous")
for x_chunk, y_chunk in stream:
    det.update(x_chunk, y_chunk)
det.close()
print(len(det.flags_))
```

Four blocks from 300 samples never flag here: the detector's default
`warmup=20` spends the first 20 block innovations building baselines
before it can raise one.

---

<a name="assemble"></a>
## `assemble`, `legendre_transfer`, `block_transfer`

From `dtfit.image`, the transfer machinery behind `ImageStream.assemble`:

```python
assemble(images, *, domain=None, order=None) -> Image
```

The image of the union of the given images on one coarse domain: every
image is transferred with `Image.transfer` and the results are merged.
`images` share one basis name; their sample sets may be disjoint or not,
merging adds them either way. `domain` defaults to the hull of the image
domains, `order` to the smallest image order.

<a name="legendre_transfer"></a>
### `legendre_transfer`

```python
legendre_transfer(local_domain, coarse_domain, local_order, coarse_order) -> ndarray
```

Change-of-basis matrix `A` with `Phi_coarse(x) = Phi_local(x) @ A` for every
`x` in `local_domain`, shape `(local_order + 1, coarse_order + 1)`. **The
exactness rule:** a coarse Legendre polynomial restricted to the local
domain is a polynomial of the same degree in the local variable, so the
identity is exact to rounding whenever `coarse_order <= local_order`.

<a name="block_transfer"></a>
### `block_transfer`

```python
block_transfer(local_domain, coarse_domain, local_order, coarse_order) -> ndarray
```

Aggregation matrix `A` (entries 0 or 1) with the same relation, shape
`(local_order, coarse_order)`, for the block basis. **The union rule:** a
coarse window must be a union of fine blocks, or the transfer raises
`ValueError` -- membership is half-open, `[x0, x1)`, except that the last
block of the coarse domain also claims its right endpoint.

---

The map-reduce and GEMM-batched estimators of the experiment notebooks --
`PartitionedLSI`, `PartitionedEAC`, `PartitionedBatchLSI`, `fit_lsi_batched`,
`project_spectra` -- live in `dtfit_experimental.scale`.
