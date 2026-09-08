# Image -- the discrete differential transform

!!! note
    Adapted from the project [wiki](https://github.com/ringavirda/science-nonline/wiki/Methods-Image). The wiki has the full set of method, domain and case-study pages.
> Numeric core every batch method fits on. Source:
> [`image/original.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/original.py),
> [`image/image.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/image.py),
> [`image/bases.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/bases.py),
> [`image/fit.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/fit.py),
> [`image/transfer.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/transfer.py),
> [`image/analytics.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/analytics.py).
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

## What an image says about itself

An image carries enough to answer questions about the signal without going
back to the samples. Every read-out below is a method on `Image` and a
function in `dtfit.image.analytics`; they read only `beta = G^+ S` and the
coefficient covariance per unit noise variance `V = G^+`.

`effective_order()` is the largest `j` whose coefficient stands three sigma
above the noise, `|beta_j| > 3 sqrt(s2 V_jj)`. The noise scale `s2` is
estimated from the orders above the answer itself: it starts as the median
of `beta_j^2 / V_jj` over the upper half of the orders, rescaled by the
chi-square median so no single large coefficient drags it, and is refined
from the mean of the tail the current answer leaves. On a Legendre
polynomial of known degree plus noise at order 24 the answer is the degree
in 79 to 88 percent of draws and never below it.

`noise_sigma()` reads the noise off those tail orders,
`sqrt(mean_j(beta_j^2 / V_jj))` for `j` above the effective order, and
returns `None` (with a `RuntimeWarning`) when fewer than eight orders are
left, since a short tail estimates nothing. It is an upper bound when the
image order sits close to the signal's own: on a damped oscillation it reads
1.76 times the true sigma at order 24, 1.10 at order 32 and 1.04 at order
40.

`decay()` fits both a geometric law `|beta_j| ~ a r^j` and an algebraic law
`|beta_j| ~ c j^-p` to the coefficients between order 2 and the effective
order, and reports both rates with both `r2` values and the name of the
better fit. The `r2` is the part to read: an oscillatory signal's Legendre
coefficients rise before they fall, and neither law describes them (a
damped cosine gives 0.099 and 0.019, an exponential 0.974 and 0.845).

`test_equal(other)` asks whether two images are of the same signal:
`d = beta_a - beta_b` against `d^T (s2_a V_a + s2_b V_b)^+ d`, chi-square
with the rank of the combined covariance. The two images may hold any
sample sets, weights and sample counts -- only basis, order and domain
must agree. Measured false-alarm rate at `alpha = 0.05` over 2000
replicates: 0.046 under Gaussian noise, 0.052 under Student-t with three
degrees of freedom, 0.048 under Laplace, unaffected by mixing a weighted
and an unweighted image (0.050 over 400 replicates).
`test_structure(model, params)` asks the other question, whether a model
explains everything the basis resolves: the leftover projections
`d = S - S_f` against `d^T (s2 G)^+ d`, which is the drop in residual sum
of squares between the model and the best fit in the span, in units of
the noise variance. `test_structure` takes its noise scale from the
basis-regression residual `(sumsq - S^T beta) / (n - rank(G))`, which
exists at any order and in any basis; `test_equal` takes each image's own
such residual instead, `s2_a` and `s2_b`, since the two images may carry
different weights. Both return a `ChiSquareTest` (`statistic`, `dof`,
`pvalue`, `alpha`, `reject`). In a test module, reach both as `Image`
methods or as `analytics.test_equal`/`analytics.test_structure`, not by
importing the bare names -- pytest collects a module-level `test_equal`
as a test.

`simulate(n=None, sigma=None, rng=None)` goes the other way: the
reconstruction plus Gaussian noise, on the image's own grid when `n` is
`None` and on `n` evenly spaced positions otherwise, with `sigma` defaulting
to `noise_sigma()`.

```python
rng = np.random.default_rng(0)
x = np.linspace(0.0, 10.0, 500)
clean = np.exp(0.9 * x)
y = clean + rng.normal(0, 0.05, x.size)
img = Original(x, y).image("legendre", 40)

print(round(img.noise_sigma(), 4), img.effective_order())

d = img.decay()
print(d.kind, round(d.ratio, 3), round(d.geometric_r2, 3), round(d.algebraic_r2, 3))

same = Original(x, clean + rng.normal(0, 0.05, x.size)).image("legendre", 40)
warm = Original(x, 1.02 * clean + rng.normal(0, 0.05, x.size)).image("legendre", 40)
print(round(img.test_equal(same).pvalue, 3), img.test_equal(warm).reject)

res = fit("a*exp(b*t)", img, "t", p0=[1.0, 1.0])
left = img.test_structure("a*exp(b*t)", res.coeffs, "t")
print(round(left.statistic, 2), left.dof, round(left.pvalue, 3), left.reject)
print(img.test_structure("a + b*t", [0.0, 1000.0], "t").reject)

xs, ys = img.simulate(200, rng=np.random.default_rng(1))
print(xs.shape, round(float(np.std(ys - img.reconstruct(xs))), 4))

resid = Original(x, y).diagnostics("a*exp(b*t)", res.coeffs, "t")
print(round(resid["durbin_watson"], 3), round(resid["lag1_autocorr"], 3))
```

The noise level comes back as `0.048` against the 0.05 that was added, and
twelve orders carry signal; the coefficients fall geometrically by `0.321`
per order, a law that explains 0.974 of their spread against the algebraic
law's 0.845. The two clean records are not distinguished (`p = 0.895`), the
two-percent-warmer one is. The exponential the data came from leaves nothing
in the span (`34.63` on 39 degrees of freedom, `p = 0.669`); a straight line
is rejected. `simulate(200, rng=...)` draws a record on a fresh grid with
the noise the image measured (`0.0443`), and `Original.diagnostics`
confirms the residuals are white (Durbin-Watson `2.043`, lag-1
autocorrelation `-0.022`).

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
stochastic tier builds its own second-order image -- lagged sums, dyadic block
sums and a fixed-grid DFT -- and fits the functionals it reads off that with
the same LSI and EAC presets -- see
[the stochastic tier](https://github.com/ringavirda/science-nonline/wiki/Methods-Stochastic).
