"""Matplotlib setup and shared figure helpers (headless / Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"figure.dpi": 110, "savefig.dpi": 300, "font.size": 12, "axes.grid": True}
)


def fit_overlay(ax, x, y, yhat, *, truth=None, title="", label="fit",
                color="tab:blue", train_split=None):
    """Scatter samples + fitted curve (+ optional ground truth / holdout split)."""
    if train_split is not None:
        ax.scatter(x[:train_split], y[:train_split], s=12, c="0.6",
                   label="train")
        ax.scatter(x[train_split:], y[train_split:], s=12, c="tab:orange",
                   label="held-out")
        ax.axvline(x[train_split], color="0.7", ls=":")
    else:
        ax.scatter(x, y, s=12, c="0.6", label="data")
    if truth is not None:
        ax.plot(x, truth, "k--", lw=1, label="ground truth")
    ax.plot(x, yhat, color=color, lw=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)


def residuals(ax, x, y, yhat, *, title="Residuals", color="tab:blue"):
    ax.plot(x, np.asarray(yhat) - np.asarray(y), color=color, lw=1)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("residual")


def error_bars(ax, names, values, *, ylabel="", title="", colors=None,
               annotate="{:.3g}"):
    """A labelled bar chart of one metric per method (lower-is-better style)."""
    colors = colors or [f"C{i}" for i in range(len(names))]
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, annotate.format(v),
                    ha="center", va="bottom", fontsize=8)
    return bars


def panel_labels(fig, axes, labels=("(а)", "(б)", "(в)", "(г)", "(д)"),
                 pad=0.015, fontsize=11):
    """Place panel tags (а)/(б)/... below each panel, clear of its x-axis and
    legend, where ДСТУ figure layout wants them rather than in the title.

    A grid gets one baseline per row; a single figure-wide baseline would drop
    the top rows' tags to the bottom of the whole figure. Call this after
    ``fig.tight_layout()`` and save with ``bbox_inches="tight"``, or the tags
    fall outside the saved area.
    """
    axs = [ax for ax in np.ravel(axes) if ax.get_visible()]
    fig.canvas.draw()  # realise a renderer so tight bounding boxes are valid
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    spans = []
    for ax in axs:
        bb = ax.get_tightbbox(rend)
        (x0, y0), (x1, _) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        rowkey = round(ax.get_position().y0, 3)  # panels in one row share this
        spans.append((x0, x1, y0, rowkey))
    base = {}                        # lowest content in each row
    for _x0, _x1, y0, rk in spans:
        base[rk] = min(base.get(rk, y0), y0)
    for (x0, x1, _y0, rk), lab in zip(spans, labels):
        fig.text(0.5 * (x0 + x1), base[rk] - pad, lab, ha="center", va="top",
                 fontsize=fontsize)


def savefig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
