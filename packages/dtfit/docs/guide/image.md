# Image -- the discrete differential transform

!!! note
    Adapted from the project [wiki](https://github.com/ringavirda/science-nonline/wiki/Methods-Image). The wiki has the full set of method, domain and case-study pages.
> Numeric core every batch method fits on. Source:
> [`image/original.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/original.py),
> [`image/image.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/image.py),
> [`image/bases.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/bases.py),
> [`image/fit.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/fit.py),
> [`image/transfer.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/transfer.py).
> Invoke via `Original(x, y)`, `.image(basis, order)`, `fit(model, data)`.
> API: [../api/fitting.md](../api/batch-fitting.md).

Every batch method in `dtfit` -- [LSI](lsi.md), [EAC](eac.md), the
streaming filters, the stochastic tier -- runs its fit on the **image** of a
signal rather than on the raw samples. An `Original` holds the sampled signal;
its `Image` in a basis at an order is a fixed-size statistic that carries
everything a least-squares fit in that basis needs and nothing else. This page
is the base the other Methods pages point to.

## Mathematical grounding

The differential-transformation lineage pairs a signal with its spectrum on a
system of functions: DSB balances the two spectra symbolically, LSI and EAC
relax that to a least-squares match. The image is the discrete form of that
spectrum, built directly from the samples rather than from a fitted
continuous model.

Given a basis `Phi = (phi_0, ..., phi_K)` and samples `(x_i, y_i, w_i)` mapped
to `u_i = 2 (x_i - x0) / (x1 - x0) - 1` on the domain `[x0, x1]`, the image is

$$
S = \Phi^\top (w \odot y), \qquad G = \Phi^\top \operatorname{diag}(w)\, \Phi,
$$

`S` of length `K+1`, `G` of shape `(K+1) x (K+1)`, together with the sample
count `n` and `sumsq = sum(w y^2)`. The least-squares coefficients in the
basis are `beta = G^+ S`, `G^+` the Hermitian pseudo-inverse (singular values
below `1e-15` of the largest are dropped, so an ill-conditioned Gram never
returns a silently wrong answer); `beta` is derived on demand and never
stored.

`S` and `G` are the discrete statistic of the samples, not a quadrature of an
integral: the model is projected on the **same grid**, `S_f(theta) = Phi^T (w
* f(x; theta))`, so the residual `S - S_f(theta)` is exactly the least-squares
residual of `y ~ Phi(x) theta` restricted to the span of `Phi` -- an
estimator that is exact NLLS in that span, with zero bias on a uniform,
clustered or random grid alike, and parameter RMSE 1.00 to 1.03 times NLLS in
the Legendre basis at `order_for`, 1.02 to 1.06 times NLLS in the block basis
at four windows per parameter. A quadrature rule would integrate the model
against a continuous approximation of the data and inherit whatever bias that
approximation carries on an irregular grid; matching the discrete projections
of the same samples the model is evaluated on has none.

`G` carries the correlation of the test functions on the sample grid exactly,
so orthogonality is never required: the Legendre polynomials and the block
indicators are two members of one family of test functions, distinguished
only by their span and locality, not by a different estimator.

## Properties

An image is additive over sample sets: `Image.merge` adds `S`, `G`, `n` and
`sumsq` (and `sumy`, `wsum`) whatever the two sample sets are, and joins the
sample positions in the grid. It is nested in order: `Image.truncate(K')`
for `K' < K` keeps the leading `K'+1` rows and columns of `S` and `G`, exact
for the Legendre basis. It is serializable: `to_dict` and `from_dict`
round-trip the whole image, and two images compare equal when their basis,
domain, grid, counts and arrays agree.

What an image does not contain is the samples themselves. That is what buys
the properties above: a size fixed by the order on a uniform grid (an
explicit grid keeps its positions), a statistic streams can accumulate
chunk by chunk ([ImageStream](https://github.com/ringavirda/science-nonline/wiki/Methods-Scaling)), a map-reduce that only
ever adds sums, and a checkpoint that is the sums.

## The projected estimator

`fit(model, data, ...)` takes an `Original` or an `Image`. An `Original` is
imaged first, at `order` (default `order_for(model)`, below). The model is
projected on the image's own grid with the image's own weights, and the
whitened residual

$$
r(\theta) = L^{-1} \big(S - S_f(\theta)\big), \qquad G = L L^\top,
$$

is minimized: nonlinear least squares restricted to the span of the basis.
Levenberg-Marquardt runs when there are no bounds; trust-region runs with
bounds, followed by a differential-evolution stage only when every bound is
finite and the local solve fails, returns a non-finite cost, or explains less
than half the weighted total sum of squares -- announced with a `UserWarning`
when it triggers.

Covariance comes from the SVD of the Jacobian of `r`, scaled by `RSS / (n -
p)` unless `absolute_sigma=True`; a parameter with a component in a null
direction of the Jacobian is not identified and is reported with `inf` on
its diagonal entry and `nan` off it, rather than a spuriously small
variance. `RSS` from an `Original` is the raw residual sum of squares over
the samples; from an `Image` it is the identity

$$
\text{RSS} = \text{sumsq} - S^\top G^+ S + (S - S_f)^\top G^+ (S - S_f),
$$

exact when the model lies in the span of the basis -- the result records
which one was used as `rss_source`, plus `image_order` and `basis_name`.

`order_for(model, params, domain, tol=0.02, max_order=64)` is the smallest
Legendre order at which every parameter sensitivity `df/dtheta` is
represented to relative L2 error `tol`, floored at `n_params - 1`.
`coverage(model, params, image)` is the largest truncation error of the
sensitivities at the image's own order, measured against a reference
expansion of `max(2K, 64)` modes; `fit` raises a `UserWarning` when that
exceeds `tol`, meaning the image is too coarse to identify the model's
parameters.

## The robust image

Robustness is a property of the image, decided once at construction, before
any model is involved. `robust=True` on `fit`, `fit_lsi`, `fit_eac` or
`Original.image` runs Huber IRLS on the basis regression `y ~ Phi beta`:
each of five passes fits the regression at the current weights, reads its
residual and MAD scale `s`, and sets sample `i`'s weight to `w_i * min(1, c s
/ |r_i|)` with `c = 1.345`. Those weights enter `S` and `G` directly; fitting,
merging and the analytics that follow are unchanged, because they only ever
see `S`, `G`, `n` and `sumsq`.

Measured with 10 percent outliers at ten sigma, the robust image gives
parameter RMSE 0.34 to 0.38 of a plain NLLS fit at order 12 -- the same
reduction scipy's `soft_l1` loss gives. It is built at `order_for(model)`
rather than at a generous order: a higher-order regression has enough
freedom to bend around an outlier and absorb part of it, which weakens the
robust weighting. Because robustness lives in the image and not in `fit`,
`fit(robust=True)` given an already-built `Image` raises: the reweighting
needs the samples, which an `Image` does not carry.

## Bases

`legendre` is the LSI basis: Legendre polynomials `P_0 .. P_order` on `[-1,
1]`, `order` the polynomial degree. `block` is the EAC basis: indicator
functions of `order` equal, half-open windows in `u`. A `Basis` exposes
`evaluate(u) -> Phi` (shape `(len(u), n_coef)`) and the property `n_coef`;
`Image.transfer` dispatches on the basis name to `legendre_transfer` and
`block_transfer` (below). The experimental package carries more families
on its own spectral machinery -- Fourier, Chebyshev, Laguerre -- see
[the experimental adaptations API](https://github.com/ringavirda/science-nonline/wiki/Experimental-Adaptations-API).

## Transfer and assembly

For the Legendre basis, a coarse polynomial restricted to a local domain
inside it is a polynomial of the same degree in the local variable, so
`Phi_coarse = Phi_local @ A` exactly, with `A` depending only on the two
domains (`legendre_transfer`, local order at least the coarse order):

$$
S_{\text{coarse}} = A^\top S_{\text{local}}, \qquad
G_{\text{coarse}} = A^\top G_{\text{local}} A,
$$

`n` and `sumsq` add. For the block basis a coarse window must be a
union of fine windows, and the transfer (`block_transfer`) is a 0/1
aggregation, exact whenever each coarse window is a union of fine windows.
`assemble(images, domain,
order)` transfers every image onto one coarse domain and sums them --
[the streams page](https://github.com/ringavirda/science-nonline/wiki/Methods-Scaling) builds streams and block map-reduce on
top of exactly this.

## Worked example

```python
rng = np.random.default_rng(0)
x = np.linspace(0.0, 3.0, 200)
y = 1.0 * np.exp(0.8 * x) + rng.normal(0, 0.05, x.size)

original = Original(x, y)
order = order_for("a*exp(b*x)", [1.0, 0.8], original.domain, var="x")
image = original.image("legendre", order)

res_orig = fit("a*exp(b*x)", original, var="x", basis="legendre", order=order, p0=[1.0, 0.8])
res_img = fit("a*exp(b*x)", image, var="x", p0=[1.0, 0.8])
print(dict(zip(res_orig.names, res_orig.coeffs)), res_orig.rss_source)
print(dict(zip(res_img.names, res_img.coeffs)), res_img.rss_source)

half = x.size // 2
left = Original(x[:half], y[:half], domain=original.domain).image("legendre", order)
right = Original(x[half:], y[half:], domain=original.domain).image("legendre", order)
res_merged = fit("a*exp(b*x)", left.merge(right), var="x", p0=[1.0, 0.8])
print(dict(zip(res_merged.names, res_merged.coeffs)))

y_out = y.copy()
y_out[::20] += 5.0
outlier = Original(x, y_out)
res_plain = fit("a*exp(b*x)", outlier.image("legendre", order), var="x", p0=[1.0, 0.8])
res_robust = fit("a*exp(b*x)", outlier.image("legendre", order, robust=True), var="x", p0=[1.0, 0.8])
print(dict(zip(res_plain.names, res_plain.coeffs)))
print(dict(zip(res_robust.names, res_robust.coeffs)))
```

`order_for` picks order 3 for this model on this domain. The fit from the
`Original` and the fit from its `Image` agree to the last printed digit,
`a=1.0025, b=0.7988`, with `rss_source` `"samples"` for the first and
`"image"` for the second. Imaging the two halves of the samples separately
and merging before fitting gives the same `a=1.0025, b=0.7988`: the image is
additive regardless of how the samples were split. With one sample in twenty
pushed 5.0 high, a hundred times the noise scale, the plain image is pulled
to `a=1.188, b=0.743`; the
robust image recovers `a=1.006, b=0.798`, close to the outlier-free fit.

## Where it is used

[LSI](lsi.md) and [EAC](eac.md) are presets of `fit` in the
Legendre and block bases; [ImageStream](https://github.com/ringavirda/science-nonline/wiki/Methods-Scaling) runs the same `S`
and `G` update over a running accumulator, block images or channel batches;
the [EACFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Equal-Areas-Filter) and
[LSIFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Legendre-Filter) read a window image each step; and the
stochastic tier fits the functionals of a random process through the same
Legendre image -- see [the stochastic tier](https://github.com/ringavirda/science-nonline/wiki/Methods-Stochastic).
