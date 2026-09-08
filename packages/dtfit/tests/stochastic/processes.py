"""Ground-truth process generators for the stochastic tier's tests.

Defined here rather than imported from the experimental harness; stable
dtfit must not depend on it."""

import numpy as np


def gen_ar1(n, phi, rng, sigma=1.0, burn=200):
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def gen_arfima(n, d, rng, ntrunc=1200):
    psi = np.empty(ntrunc)
    psi[0] = 1.0
    for j in range(1, ntrunc):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    e = rng.standard_normal(n + ntrunc)
    return np.convolve(e, psi)[ntrunc:ntrunc + n]


def gen_garch(n, omega, alpha, beta, rng, burn=500):
    N = n + burn
    z = rng.standard_normal(N)
    s2 = np.empty(N)
    r = np.empty(N)
    s2[0] = omega / max(1e-9, 1.0 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, N):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:]


def gen_ar2(n, p1, p2, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = p1 * x[t - 1] + p2 * x[t - 2] + rng.standard_normal()
    return x


def gen_ar2_cycle(n, period, damping, rng, burn=300):
    phi1 = 2.0 * damping * np.cos(2.0 * np.pi / period)
    phi2 = -(damping ** 2)
    N = n + burn
    e = rng.standard_normal(N)
    x = np.zeros(N)
    for t in range(2, N):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x[burn:]


def gen_trend_cycle(n, slope, period, amp, noise_sd, rng):
    t = np.arange(n, dtype=float)
    y = (slope * t + amp * np.sin(2.0 * np.pi * t / period)
         + rng.normal(0.0, noise_sd, n))
    return t, y
