# EAC -- Equal-Areas Criterion

!!! note
    Adapted from the project [wiki](https://github.com/ringavirda/science-nonline/wiki/Methods-EAC). The wiki has the full set of method, domain and case-study pages.
> Numeric batch method, successor to the symbolic DSBE. Source:
> [`image/fit.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/fit.py),
> [`image/bases.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/image/bases.py).
> Invoke via `fit_eac(x, y, expr, var, ...)`, `fit(model, data, basis="block",
> order=n_windows)`, or `NonlineRegressor(..., basis="block")`.

EAC is [`fit`](image.md) in the **block** basis: the basis is the set of
indicator functions of `n_windows` equal windows in the normalized variable
`u`, so the image `S` is the vector of window sums -- the areas -- and the
Gram `G` is diagonal, each entry the sum of the sample weights in its window.
The model is projected on the same grid and matched window sum for window
sum. EAC is the numeric successor of the symbolic DSBE, and its block image
is the basis of the streaming [EACFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Equal-Areas-Filter).

## Mathematical grounding

For a model $f(x;\theta)$ with $m$ unknown parameters, the block image splits
the domain into $M \ge m$ equal windows $W_1,\dots,W_M$ and requires the
model's window sum to equal the data's on each:

$$
S_{f,i}(\theta) \;=\; \sum_{k \in W_i} w_k\, f(x_k;\theta)
\qquad = \qquad
S_i \;=\; \sum_{k \in W_i} w_k\, y_k,
\qquad i = 1,\dots,M .
$$

That is $M$ equations in $m$ unknowns -- residuals

$$
r_i(\theta) \;=\; S_i - S_{f,i}(\theta) .
$$

**Connection to the differential spectrum.** The area of a signal over
$[0,H]$ is the integral of its inverse transform,

$$
\int_{0}^{H} x(t)\,dt
   = \sum_{k} X(k)\!\int_0^H\!\Big(\tfrac{t}{H}\Big)^{k}\!dt
   = H\sum_{k} \frac{X(k)}{k+1},
$$

so an area is a **moment of the differential spectrum**. Matching areas over
$M$ shifted windows is matching $M$ independent integral functionals of the
spectrum -- a weak-form (Galerkin, piecewise-constant test function)
identification. Two analytic functions sharing $m$ such independent moments
agree where the model has $m$ degrees of freedom, so the parameters are
recovered.

**What it is for.** A window sum is a moment of the differential spectrum,
so matching M shifted window sums is a piecewise-constant (Haar) Galerkin
identification: the h-version of the same weak-form fit LSI runs with a
global polynomial basis. EAC owns the h-version's regime -- a jump or a
regime change aligned to a window edge -- and the diagonal block Gram makes
it the cheap, well-conditioned image at high order and the basis of the
streaming EACFilter and the map-reduce tiers. It is not more robust to
noise or outliers than LSI: robustness is a property of the robust image
(below), not of the block basis.

## Windows

`n_windows` defaults to **four per parameter** (`4 * n_params`); an explicit
value must still leave at least as many windows as parameters, since
[`fit`](image.md) raises when the image has fewer coefficients than the
model has parameters. The windows are equal in the normalized variable `u`
and half-open: a sample exactly on a window's upper edge belongs to the
**next** window; a sample at the domain's end belongs to the last window.
Measured on the model catalog, this recovers parameter RMSE 1.02 to 1.06
times NLLS at four windows per parameter.

`fit_eac` recovers a sharp sigmoid step with its windows spread evenly across
`x` (dotted edges): the block preset uses four uniform windows per parameter,
and the estimate comes from the projection on those windows, not from where
the curve bends.

![EAC with uniform windows on a sharp sigmoid step](figures/eac_adaptive.png)

## Algorithm

1. **Parse** the model; collect the $m$ free parameters $\theta$.
2. **Image** the data in the block basis at `n_windows` (default $4m$):
   $S_i = \sum_{k \in W_i} w_k y_k$, and $G$ diagonal with $G_{ii} =
   \sum_{k \in W_i} w_k$.
3. **Project** the model and its analytic sensitivities
   $\partial f/\partial\theta_j$ onto the same grid:
   $S_{f,i}(\theta) = \sum_{k \in W_i} w_k f(x_k;\theta)$, and likewise for
   the Jacobian columns.
4. **Whiten** by the diagonal Gram: $r(\theta) = L^{-1}\big(S -
   S_f(\theta)\big)$ with $G = LL^\top$ -- for the diagonal block Gram this
   is dividing each window residual by the square root of its weight sum.
5. **Solve**: Levenberg-Marquardt when unbounded, trust-region when `bounds`
   is given, followed by a differential-evolution stage only when every
   bound is finite and the local solve fails, returns a non-finite cost, or
   explains less than half the weighted total sum of squares.
6. **Return** the fitted $\theta$ and, when the degrees of freedom allow, a
   parameter covariance from the SVD of the whitened Jacobian.

## Relation to classical (Western) methods

The Pukhov differential-transformation lineage is largely absent from the Western
literature, but EAC has exact, well-known counterparts there -- naming them makes
the method legible to a signal-processing or system-identification audience and
explains *why* the defaults are what they are.

- **Galerkin weighted residuals (Haar test functions).** Requiring the window-
  sum residual to vanish on each window is exactly a **Galerkin / method-of-
  weighted-residuals** identification with **piecewise-constant (Haar /
  indicator) test functions** $\phi_i$: $\sum_k \phi_i(x_k)\, w_k\,
  [y_k - f(x_k;\theta)] = 0$. The exactly-determined case ($M = m$) is the
  classical square Galerkin system.
- **Over-identified method of moments (GMM).** `fit_eac`'s default of $M =
  4m$ windows is an **over-identified moment system** -- more moment
  conditions than unknowns, solved by least squares. This is the lens that
  justifies windows beyond the parameter count: Hansen's GMM theory says
  extra moment conditions reduce estimator variance and supply the residual
  covariance an exactly-determined system cannot. The robust image is then
  a **robust GMM / M-estimator** on those moment conditions.
- **Alternative routes to the same parameters.** For the special case of sums of
  exponentials / sinusoids, the classical *algebraic* route is **Prony's method**
  and its SVD-robust successors **Matrix Pencil / ESPRIT** (recover the modes as
  roots / subspace eigenvalues rather than by area matching); the *separable*
  route is **variable projection** (Golub-Pereyra). dtfit's experimental suite runs
  these head-to-head as baselines -- see [the baselines page](https://github.com/ringavirda/science-nonline/wiki/Experimental-Baselines).

In one line: **EAC is the $h$-version (local Haar) Galerkin / GMM** counterpart of
[LSI](lsi.md)'s $p$-version (global spectral) projection -- two faces of the
same weighted-residual identification.

## The h/p crossover: when EAC beats LSI

LSI (Legendre, p-refinement) and EAC (block, h-refinement) are two bases on
the same image, so the classical h/p rule decides between them: p-refinement
wins wherever the target is globally smooth (spectral convergence), h-
refinement wins at a local non-smoothness a polynomial cannot follow without
ringing. Image reconstruction RMSE at an equal coefficient budget K
(n = 2000) makes the crossover concrete:

| target | LSI (Legendre) | EAC (block) |
|---|---|---|
| smooth (exp, Gaussian) | 5e-16 at K>=16 | ~1e-2 (stalls) |
| kink \|x-0.5\|, K=8 | 7.9e-3 | 3.6e-2 |
| kink \|x-0.5\|, K=64 | 4.3e-4 | 4.5e-3 |
| two-piece slope (continuous) | wins at every K | -- |
| step at 0.5 on a window edge | ~5e-2 (Gibbs, K=64) | 0.00e+00 (exact) |

LSI dominates smooth and continuous-but-non-smooth targets; EAC is exact
only on a true discontinuity that lands on a block boundary. That step is
EAC's home, together with the conditioning and streaming reasons above.

## Robustness to outliers -- the robust image

Robustness is a property of the **image construction, not of the basis**.
`robust=True` (or any `loss` other than `"linear"`) runs Huber IRLS on the
image regression before the model is involved -- each sample weighted by
`min(1, c s / |r_i|)`, `c = 1.345`, `s` the MAD scale, five passes -- and it
applies identically to the Legendre and the block image. Measured over 60
seeds with 10 percent outliers at ten sigma, RMSE-to-truth relative to a
clean-data NLLS fit:

* Plain images are no more robust than NLLS: block 3.41, Legendre 3.39,
  NLLS 3.39 under scattered outliers.
* The robust image is basis-agnostic under scattered outliers: LSI-robust
  1.17 and EAC-robust 1.17, matching a dedicated robust loss.
* Under a contiguous burst the robust image on Legendre (1.41) beats it on
  blocks (3.65): a burst fills whole windows the per-window reweighting
  cannot isolate, so the global basis, not the block basis, is the robust
  choice there.

Reach for `robust=True` on whichever basis the signal's shape already
chose; do not choose EAC for robustness.

## Optimizations and guards

- **Diagonal Gram** -- the block basis's windows never overlap, so `G` is
  diagonal and whitening the residual amounts to a per-window division by
  the square root of its weight sum.
- **Per-sample sensitivity mask** -- a transcendental sensitivity can be
  singular at an isolated sample while its window sum stays finite elsewhere
  (e.g. $\partial_n\, x^{n} = x^{n}\ln x$ is `NaN` at $x=0$, with limit $0$).
  Such a sample is zeroed before the projection so it cannot poison a window
  sum or a Jacobian column.
- **Sample-count guard** -- an image at `n_windows` windows needs at least
  `n_windows + 1` samples; `fit_eac` raises otherwise.
- **Coverage does not apply** -- [`coverage`](image.md) measures
  Legendre truncation error and returns `0.0` for the block basis; `fit`
  never runs the coverage check on an EAC image.

## Worked example

`y = a.arctan(w.x)` (truth `a=2.0, w=3.0`), a transcendental **non-Taylor**
saturation curve, with 8 % noise. **Left:** EAC recovers `a~=2.00, w~=2.99` from the
noisy cloud; the shaded bands are the per-parameter integration windows. **Right:**
the equal-areas criterion -- the cumulative integral of the fitted model (dashed)
tracks the cumulative integral of the data, which is the quantity EAC actually
matches.

![EAC fit and the equal-areas criterion](figures/eac_fit.png)

## Comparison

**Model data -- `y = a.exp(b.x)`, ground truth a=1.0, b=1.2, 5 % noise, n=80.**
Error is against the *clean* signal.

| method | recovered params | R^2 | RMSE | MAPE % | fit (ms) |
|---|---|---|---|---|---|
| LSI | a=1.000, b=1.203 | 0.9999 | 0.01133 | 0.24 | 15.3 |
| **EAC** | a=1.002, b=1.203 | 0.9999 | 0.01641 | 0.41 | 3.4 |
| SciPy `curve_fit` | a=1.000, b=1.204 | 0.9999 | 0.01305 | 0.25 | 0.1 |
| numpy.polyfit (deg 5) | -- | 0.9997 | 0.02302 | 0.85 | 0.1 |

EAC recovers the parameters essentially as well as LSI and the NLS gold standard
while being the **fastest** of the dtfit methods (~=3 ms here -- roughly 5x LSI),
because it solves a small area-matching system instead of a spectral least-squares
problem. That speed and its derivative-free, O(1)-per-sample form are why EAC is the
basis of the streaming [EACFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Equal-Areas-Filter).

**Real data -- COVID-19 Ukraine** (28-day take-off, 548->8617 cases),
`y = a.exp(b.t)`:

| method | R^2 | RMSE | MAPE % |
|---|---|---|---|
| LSI | 0.9877 | 275.1 | 13.00 |
| **EAC** | 0.8506 | 960.1 | 9.39 |
| SciPy `curve_fit` | 0.9879 | 273.5 | 13.34 |

## Where it is best applied

**Use EAC for:** a jump or regime change aligned to a window edge (the h-
version's exact case), high-order conditioning (the diagonal block Gram --
the showcase's 1,275-station Legendre Gram hit condition 1e19 where the
block Gram stays diagonal), and the streaming and map-reduce tiers where the
block image is `n_windows` sums, the batch form of the streaming
[EACFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Equal-Areas-Filter)'s measurement and of the MCU block
images. It is also a fast, stable initializer for a slower method. For peaks
and cycles the Legendre preset at [`order_for`](image.md) is the more
efficient image, and `Model.fit` and the `auto` route send peaks there
rather than to the block basis (1.04 to 1.22 times the Legendre parameter
error on the peaked families). Outlier-prone data reach for the robust image
(`robust=True`) on whichever basis the shape chose, scattered or bursty.

**Caveats.** EAC's window sums partly cancel **oscillations** -- for a cycle
use [LSI](lsi.md)'s oscillatory recipe or the streaming
[LSIFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Legendre-Filter). For real-time tracking of
*time-varying* parameters, use the recursive [EACFilter](https://github.com/ringavirda/science-nonline/wiki/Methods-Equal-Areas-Filter).
