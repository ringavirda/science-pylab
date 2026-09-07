"""Exact transfer of images between domains and orders, and assembly of
block images onto a coarse domain."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.polynomial.legendre as leg

from .bases import LegendreBasis, u_of
from .image import Image


def legendre_transfer(
    local_domain: tuple[float, float],
    coarse_domain: tuple[float, float],
    local_order: int,
    coarse_order: int,
) -> np.ndarray:
    """Change-of-basis matrix ``A`` with ``Phi_coarse(x) = Phi_local(x) @
    A`` for every ``x`` in ``local_domain``.

    A coarse Legendre polynomial restricted to the local domain is a
    polynomial of the same degree in the local variable, so the identity is
    exact when ``coarse_order <= local_order``; ``A`` is found by
    interpolation at ``local_order + 1`` Gauss-Legendre nodes.

    Args:
        local_domain: ``(x0, x1)`` of the local basis, ``x0 < x1``.
        coarse_domain: ``(x0, x1)`` of the coarse basis, ``x0 < x1``.
        local_order: Legendre order of the local image, at least
            ``coarse_order``.
        coarse_order: Legendre order on the coarse domain, at least 1.

    Returns:
        Array of shape ``(local_order + 1, coarse_order + 1)``.

    Raises:
        ValueError: ``coarse_order`` above ``local_order`` or below 1, or a
            degenerate domain.
    """
    if coarse_order < 1 or coarse_order > local_order:
        raise ValueError(
            f"coarse order {coarse_order} must be between 1 and the local "
            f"order {local_order}"
        )
    a0, a1 = float(local_domain[0]), float(local_domain[1])
    if not a1 > a0 or not coarse_domain[1] > coarse_domain[0]:
        raise ValueError("domains must be non-degenerate intervals")
    nodes, _ = leg.leggauss(local_order + 1)
    x = a0 + (a1 - a0) * (nodes + 1.0) / 2.0
    phi_l = LegendreBasis(local_order).evaluate(u_of(x, a0, a1))
    phi_c = LegendreBasis(coarse_order).evaluate(
        u_of(x, float(coarse_domain[0]), float(coarse_domain[1]))
    )
    return np.linalg.solve(phi_l, phi_c)


def block_transfer(
    local_domain: tuple[float, float],
    coarse_domain: tuple[float, float],
    local_order: int,
    coarse_order: int,
) -> np.ndarray:
    """Aggregation matrix ``A`` (entries 0 or 1) with
    ``Phi_coarse(x) = Phi_local(x) @ A`` for the block basis: fine window
    ``i`` maps to the coarse window that contains it.

    Callers must treat block membership as half-open on the right,
    ``[x0, x1)``, except for the last block of the coarse domain: this
    matrix always assigns a shared edge to the window it opens, but
    :class:`~dtfit.image.bases.BlockBasis` clamps a sample exactly at
    the local domain's right endpoint into its last (closed) window, so
    the two disagree there when that endpoint is an interior coarse
    edge. A caller cutting fine blocks (as in :func:`assemble`) must
    keep each block's samples inside ``[x0, x1)`` to avoid the seam.

    Domain and window containment are checked with an absolute
    tolerance of ``1e-9 * (coarse_domain[1] - coarse_domain[0])``, so a
    local domain or fine-window edge within that distance of a coarse
    edge is silently snapped to it.

    Args:
        local_domain: ``(x0, x1)`` of the fine windows.
        coarse_domain: ``(x0, x1)`` of the coarse windows, containing the
            local domain.
        local_order: number of fine windows, at least 1.
        coarse_order: number of coarse windows, at least 1.

    Returns:
        Array of shape ``(local_order, coarse_order)``.

    Raises:
        ValueError: ``local_order`` or ``coarse_order`` below 1; a local
            domain outside the coarse one; or a fine window that is not
            inside one coarse window (the coarse window is then not a
            union of fine blocks).
    """
    if local_order < 1 or coarse_order < 1:
        raise ValueError(
            f"local_order and coarse_order must be at least 1, got "
            f"{local_order} and {coarse_order}"
        )
    a0, a1 = float(local_domain[0]), float(local_domain[1])
    c0, c1 = float(coarse_domain[0]), float(coarse_domain[1])
    tol = 1e-9 * (c1 - c0)
    if a0 < c0 - tol or a1 > c1 + tol:
        raise ValueError(
            "the local domain must lie inside the coarse domain"
        )
    fine = np.linspace(a0, a1, local_order + 1)
    coarse = np.linspace(c0, c1, coarse_order + 1)
    A = np.zeros((local_order, coarse_order))
    for i in range(local_order):
        lo, hi = fine[i], fine[i + 1]
        inside = np.flatnonzero(
            (coarse[:-1] <= lo + tol) & (hi <= coarse[1:] + tol)
        )
        if inside.size == 0:
            raise ValueError(
                "a coarse block window must be a union of fine blocks; "
                f"fine window [{lo:g}, {hi:g}] straddles a coarse edge"
            )
        A[i, inside[0]] = 1.0
    return A


def assemble(
    images: Sequence[Image],
    *,
    domain: tuple[float, float] | None = None,
    order: int | None = None,
) -> Image:
    """The image of the union of the given images on one coarse domain:
    every image is transferred with :meth:`Image.transfer` and the
    results are merged. ``images`` are the pieces (windows or blocks) of
    one series, in any of the image bases; the coarse basis matches
    theirs.

    Args:
        images: images with the same basis name; their sample sets are
            disjoint or not, merging adds them either way.
        domain: coarse domain; default the hull of the image domains.
        order: coarse order; default the smallest image order. For the
            block basis this coarsens by the number of images (three
            order-4 blocks default to order 4 over their hull, windows
            three times as wide as the fine ones); pass ``order``
            explicitly to keep the fine window width.

    Returns:
        One :class:`Image` on ``domain`` at ``order``.

    Raises:
        ValueError: no images, mixed basis names, or any error of
            :meth:`Image.transfer`.
    """
    imgs = list(images)
    if not imgs:
        raise ValueError("assemble needs at least one image, got no images")
    names = {img.basis.name for img in imgs}
    if len(names) != 1:
        raise ValueError(f"images must share one basis, got {sorted(names)}")
    if domain is None:
        domain = (
            min(img.domain[0] for img in imgs),
            max(img.domain[1] for img in imgs),
        )
    if order is None:
        order = min(img.order for img in imgs)
    out = imgs[0].transfer(domain, order)
    for img in imgs[1:]:
        out = out.merge(img.transfer(domain, order))
    return out
