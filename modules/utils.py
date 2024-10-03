import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Any
from scipy.interpolate import interp1d
from time import perf_counter


def size_equal(func: Callable[[np.ndarray, np.ndarray], Any]):
    def inner(left: np.ndarray, right: np.ndarray) -> Any:
        if left.size != right.size:
            raise RuntimeError("Array sizes must be equal.")
        else:
            return func(left, right)

    return inner


@size_equal
def lin_div(left: np.ndarray, right: np.ndarray) -> np.float64:
    div: np.float64 = 0
    for i, v in np.ndenumerate(left):
        div += np.abs(v - right[i])
    return div


@size_equal
def concord(left: np.ndarray, right: np.ndarray) -> np.float64:
    return np.abs(
        2
        * np.cov(left, right, bias=True)[0][1]
        / (
            np.var(left)
            + np.var(right)
            + np.power(np.median(left) - np.median(right), 2)
        )
    )


@size_equal
def rss(left: np.ndarray, right: np.ndarray) -> np.float64:
    res: np.float64 = 0.0
    for i, l in np.ndenumerate(left):
        res += np.square(l - right[i])
    return res


def tss(arr: np.ndarray) -> np.float64:
    res: np.float64 = 0.0
    mean = arr.mean()
    for y in np.nditer(arr):
        res += np.square(y - mean)
    return res


@size_equal
def rse(left: np.ndarray, right: np.ndarray) -> np.float64:
    return np.sqrt(rss(left, right) / (left.size - 2))


@size_equal
def r_sq(left: np.ndarray, right: np.ndarray) -> np.float64:
    return 1 - rss(left, right) / tss(right) 


@size_equal
def mse(left: np.ndarray, right: np.ndarray) -> np.float64:
    return (np.square(left - right)).mean()


def period(arr: np.ndarray) -> np.float64:
    period = 0
    passes = 0
    origin = arr[period]
    for next in np.nditer(arr):
        if next == origin:
            passes += 1
        elif passes == 2:
            return period
        else:
            period += 1


def growth(arr: np.ndarray) -> np.float64:
    gr = np.empty(arr.size - 1)
    for i in np.arange(arr.size):
        if i < arr.size - 1:
            gr[i] = arr[i + 1] - arr[i]
    return gr.mean()

def statistics(data: np.ndarray, *fitted: tuple[np.ndarray, str]) -> None:
    keys = ("d_med", "d_var", "std. div.", "mse", "lin. div.", "conc")
    labels = []
    frame: dict[str, list] = {k: [] for k in keys}
    fitted = list(fitted)
    fitted.append((data, "data"))
    for fit in fitted:
        frame[keys[0]].append(np.abs(np.median(data) - np.median(fit[0])))
        frame[keys[1]].append((np.var(np.median(data) - fit[0])))
        frame[keys[2]].append(np.std(fit[0]))
        frame[keys[3]].append(mse(data, fit[0]))
        frame[keys[4]].append(lin_div(data, fit[0]))
        frame[keys[5]].append(concord(data, fit[0]))
        labels.append(fit[1])
    pd.set_option("display.width", 1000)
    print(pd.DataFrame(frame, index=labels))


def statistics_n(*fitted: tuple[np.ndarray, np.ndarray, str]) -> None:
    keys = ("d_med", "d_var", "std. div.", "mse", "lin. div.", "conc")
    labels = []
    frame: dict[str, list] = {k: [] for k in keys}
    fitted = list(fitted)
    for fit in fitted:
        frame[keys[0]].append(np.abs(np.median(fit[0]) - np.median(fit[1])))
        frame[keys[1]].append(np.abs(np.median(fit[0]) - np.var(fit[1])))
        frame[keys[2]].append(np.std(fit[1]))
        frame[keys[3]].append(mse(fit[0], fit[1]))
        frame[keys[4]].append(lin_div(fit[0], fit[1]))
        frame[keys[5]].append(concord(fit[0], fit[1]))
        labels.append(fit[2])
    pd.set_option("display.width", 1000)
    print(pd.DataFrame(frame, index=labels))


def timed(proc: Callable[(...), Any], label: str = None) -> Any:
    time = perf_counter()
    res = proc()
    time = perf_counter() - time
    label = proc.__name__ if not label else label
    print(f"[{label}] took {time*1000:.5f}ms to complete.")
    return res


def scale(data: np.ndarray, coeff: float) -> np.ndarray:
    range_new = np.arange(0, data.size, 1 / coeff)
    inter = interp1d(np.arange(data.size), data)
    return inter(range_new)


def multi_plot(
    data: np.ndarray,
    *args: tuple[np.ndarray, str],
    ylims: tuple[float, float] = None,
    row_size: int = 3,
    fig_scale: int = 5,
) -> None:
    rows = (
        int(np.floor(len(args) / row_size))
        if int(np.floor(len(args) / row_size)) >= 1
        else 1
    )
    fig, ax = plt.subplots(rows, row_size)
    for i in range(len(args)):
        if ax.ndim > 1:
            row = int(np.floor(i / row_size))
            col = i - row * row_size
            ax[row, col].plot(data, label="data")
            ax[row, col].plot(args[i][0], label=args[i][1])
            ax[row, col].legend(loc="lower right")
            if ylims:
                ax[row, col].ylim(ylims)
        else:
            ax[i].plot(data, label="data")
            ax[i].plot(args[i][0], label=args[i][1])
            ax[i].legend(loc="lower right")
            if ylims:
                ax[i].set_ylim(ylims)
    fig.set_figwidth(row_size * fig_scale)
    fig.set_figheight(rows * fig_scale)
    plt.show()


def multi_plot_n(
    *args: tuple[np.ndarray, np.ndarray, str],
    ylims: tuple[float, float] = None,
    row_size: int = 3,
    fig_scale: int = 5,
) -> None:
    rows = (
        int(np.floor(len(args) / row_size))
        if int(np.floor(len(args) / row_size)) >= 1
        else 1
    )
    fig, ax = plt.subplots(rows, row_size)
    for i in range(len(args)):
        if ax.ndim > 1:
            row = int(np.floor(i / row_size))
            col = i - row * row_size
            ax[row, col].plot(args[i][0], label="data")
            ax[row, col].plot(args[i][1], label=args[i][2])
            ax[row, col].legend(loc="lower right")
            if ylims:
                ax[row, col].ylim(ylims)
        else:
            ax[i].plot(args[i][0], label="data")
            ax[i].plot(args[i][1], label=args[i][2])
            ax[i].legend(loc="lower right")
            if ylims:
                ax[i].set_ylim(ylims)
    fig.set_figwidth(row_size * fig_scale)
    fig.set_figheight(rows * fig_scale)
    plt.show()
