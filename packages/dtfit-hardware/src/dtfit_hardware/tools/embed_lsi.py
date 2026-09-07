"""Freeze a streaming-LSI config into embeddable form plus a golden reference.

The on-MCU filter is a fixed-size specialization of
``dtfit.streaming.LSIFilter``: one model, a fixed window ``W`` and Legendre
``order``, full-window only, with no adaptive-window, drift or robust paths.
This module bridges the Python method and the C firmware.

* :func:`tables` precomputes every constant the hot path needs, being the
  whitened basis matrix B, the noise variance and the process-noise
  diagonal. On the MCU these live in read-only flash.
* :func:`golden_run` reimplements the C hot path in float64, operation for
  operation. It is the host reference the embedded float32 filter is checked
  against.
* :func:`cross_check` shows the golden matches the real ``LSIFilter``
  configured to the same fixed-window subset, which is what makes the embedded
  filter demonstrably the dtfit method rather than a lookalike.
* :func:`emit_header` writes ``lsi_tables.h`` for the firmware.

The frozen model is a per-axis constant velocity ``y = c0 + c1*t`` on a
monomial basis. Being linear in the parameters, it stays well-conditioned in
float32 even at large absolute ``t``, where the condition number goes as
t0/span.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.polynomial import legendre as L

# Frozen configuration.
W = 15                  # sliding-window length
ORDER = 5               # Legendre spectral order -> M = ORDER + 1 coefficients
DEGREE = 1              # model degree: y = sum_{k=0..DEGREE} c_k t^k  (N = DEGREE+1)
R0 = 1.0                # base measurement-noise variance
Q_DIAG = (0.01, 0.01)   # process-noise variance per parameter (len == N)
P0_DIAG = 10.0          # initial covariance diagonal (LSIFilter uses eye*10)
F_CPU_HZ = 64_000_000   # nRF52840 core clock, for cycles -> microseconds

M = ORDER + 1
N = DEGREE + 1
HERE = Path(__file__).resolve().parent
FIRMWARE = HERE.parent / "firmware"
# Every sketch dir that #includes the generated tables. Arduino demands a
# sketch-local copy of each header, so the generator writes lsi_tables.h into
# all of them in one pass. Regenerating just one would silently leave the
# other's tables stale, and mismatched firmware would ship.
FIRMWARE_TARGETS = ("nano_lsi_onboard", "nano_lsi_log")


def tables() -> dict:
    """Every constant the on-MCU hot path needs; these become flash tables.

    ``B = Phi L^-T`` with ``Phi`` the Legendre basis on the uniform window
    mapped to ``[-1, 1]`` and ``L`` the Cholesky factor of ``Phi^T Phi``:
    ``B^T y`` is the whitened window image of ``y``.
    """
    tau = np.linspace(-1.0, 1.0, W)
    phi = L.legvander(tau, ORDER)                            # (W, M)
    chol = np.linalg.cholesky(phi.T @ phi)                   # (M, M)
    b = np.linalg.solve(chol, phi.T).T                       # (W, M)
    return {
        "B": b, "s2": R0, "q_diag": np.asarray(Q_DIAG, float),
        "p0_diag": P0_DIAG,
    }


def golden_run(t: np.ndarray, y: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Float64 reference for the C hot path, one (t, y) sample at a time.

    Returns:
        The per-sample parameter estimate ``p`` (N,), shape ``(len(t), N)``.
        Rows before the window fills hold the initial ``p0``.
    """
    tb = tables()
    B, s2, q_diag = tb["B"], tb["s2"], tb["q_diag"]
    p = np.array(p0, float)
    P = np.eye(N) * tb["p0_diag"]
    Q = np.diag(q_diag)
    tw: list[float] = []
    yw: list[float] = []
    out = np.empty((len(t), N))
    for s in range(len(t)):
        tw.append(float(t[s]))
        yw.append(float(y[s]))
        if len(tw) > W:
            tw.pop(0)
            yw.pop(0)
        if len(tw) == W:
            ta = np.asarray(tw)
            ya = np.asarray(yw)
            z = B.T @ ya                             # whitened window image
            H = np.column_stack(
                [B.T @ ta ** k for k in range(N)]
            )                                         # (M, N)
            e = z - H @ p
            A = H.T @ H / s2
            bvec = H.T @ e / s2
            P_post = np.linalg.inv(np.linalg.inv(P) + A)
            p = p + P_post @ bvec
            P = P_post + Q
        out[s] = p
    return out


def dtfit_run(t: np.ndarray, y: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """The real ``LSIFilter`` constrained to the embedded fixed-window subset."""
    from dtfit.streaming import LSIFilter

    expr = " + ".join(["c0"] + [f"c{k}*t**{k}" if k > 1 else "c1*t"
                                 for k in range(1, N)])
    f = LSIFilter(
        expr, "t", window_size=W, order=ORDER, min_window=W,
        noise_var=R0, q_diag=list(Q_DIAG), p0=list(p0),
        adaptive_window=False, robust=False,
        cusum_k=float("inf"), alpha=1e-15,
    )
    out = np.empty((len(t), N))
    for s in range(len(t)):
        f.partial_fit(float(t[s]), float(y[s]))
        out[s] = f.p
    return out


def cross_check() -> float:
    """Golden against the real LSIFilter on a synthetic ramp plus noise.

    Returns:
        The largest absolute parameter difference over the run.
    """
    rng = np.random.default_rng(0)
    t = np.arange(80) * 0.1
    y = 3.0 - 1.5 * t + rng.normal(0, 0.05, t.size)
    p0 = np.array([y[0]] + [0.0] * (N - 1))
    g = golden_run(t, y, p0)
    d = dtfit_run(t, y, p0)
    return float(np.max(np.abs(g - d)))


def _fc(v: float) -> str:
    """Format a float as a valid C float literal, decimal or exponent.

    Anything under ``1e-12`` in magnitude is snapped to ``0.0``. Those are
    projection entries that are analytically zero but carry ~1e-17 of roundoff
    which differs between BLAS and NumPy builds; snapping them keeps the
    emitted tables byte-identical across machines. Otherwise the checked-in
    tables sync test goes flaky and the firmware ships meaningless noise.
    """
    if abs(v) < 1e-12:
        v = 0.0
    s = f"{v:.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"          # "1" -> "1.0", or the 'f' suffix is invalid C++
    return s + "f"


def _carr(name: str, a: np.ndarray, dims: str) -> str:
    flat = np.asarray(a, float).ravel()
    body = ", ".join(_fc(v) for v in flat)
    return f"static const float {name}{dims} = {{{body}}};"


def render_header() -> str:
    """Render the firmware's ``lsi_tables.h`` text from the frozen config."""
    tb = tables()
    lines = [
        "// Generated by tools/embed_lsi.py -- do not edit by hand.",
        "// Frozen streaming-LSI config for the on-MCU filter (flash tables).",
        "#pragma once",
        "",
        f"#define LSI_W {W}",
        f"#define LSI_ORDER {ORDER}",
        f"#define LSI_M {M}        // ORDER + 1 Legendre coefficients",
        f"#define LSI_N {N}        // model parameters (degree {DEGREE})",
        f"#define LSI_DEGREE {DEGREE}",
        f"#define LSI_P0_DIAG {P0_DIAG}f",
        f"#define LSI_S2 {_fc(R0)}",
        f"#define LSI_F_CPU_HZ {F_CPU_HZ}u",
        "",
        _carr("LSI_B", tb["B"], "[LSI_W][LSI_M]"),
        _carr("LSI_QDIAG", tb["q_diag"], "[LSI_N]"),
        "",
    ]
    return "\n".join(lines)


def emit_header(path: Path | None = None) -> list[Path]:
    """Write the firmware's ``lsi_tables.h`` from the frozen config.

    Args:
        path: Write this single file instead, such as a temp file in a test.
            With no ``path`` the header goes into every sketch dir that
            consumes it (:data:`FIRMWARE_TARGETS`), which is what stops a
            config change leaving one sketch's tables stale.

    Returns:
        The paths written.
    """
    text = render_header()
    targets = ([path] if path is not None
               else [FIRMWARE / t / "lsi_tables.h" for t in FIRMWARE_TARGETS])
    for out in targets:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return targets


def load_sample(col: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Load (t_seconds, y) from the newest recorded BLE sample CSV.

    Time is re-referenced to the first sample and converted to seconds. With no
    sample recorded yet this falls back to a synthetic ramp plus noise.

    Args:
        col: Signal column to take; 4 is longitude.
    """
    data_dir = HERE.parent / "data"
    samples = sorted(data_dir.glob("sample_*.csv"))
    if not samples:
        rng = np.random.default_rng(1)
        t = np.arange(120) * 1.0
        return t, -77.05 + 1e-3 * t + rng.normal(0, 2e-4, t.size)
    rows = samples[-1].read_text(encoding="utf-8").strip().splitlines()[1:]
    t_ms, y = [], []
    for ln in rows:
        p = ln.split(",")
        if len(p) == 14:
            t_ms.append(float(p[0]))
            y.append(float(p[col]))
    t_ms = np.asarray(t_ms)
    return (t_ms - t_ms[0]) / 1000.0, np.asarray(y)


def emit_testvec(t: np.ndarray, y: np.ndarray, path: Path | None = None) -> Path:
    """Emit ``lsi_testvec.h``, the on-boot self-validation vector."""
    out = path or (FIRMWARE / "nano_lsi_onboard" / "lsi_testvec.h")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "// Generated by tools/embed_lsi.py -- on-boot validation vector.",
        "#pragma once",
        f"#define LSI_NT {len(t)}",
        _carr("LSI_T", t, "[LSI_NT]"),
        _carr("LSI_Y", y, "[LSI_NT]"),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    diff = cross_check()
    print(f"golden vs dtfit.LSIFilter  max|dp| = {diff:.3e}")
    for hdr in emit_header():
        print("wrote", hdr)
    t, y = load_sample()
    p0 = np.array([y[0]] + [0.0] * (N - 1))
    tv = emit_testvec(t, y)
    print("wrote", tv, f"({len(t)} samples)")
    g = golden_run(t, y, p0)
    print(f"config: W={W} order={ORDER} M={M} N={N} degree={DEGREE}")
    print(f"golden final estimate p = {g[-1]}")
