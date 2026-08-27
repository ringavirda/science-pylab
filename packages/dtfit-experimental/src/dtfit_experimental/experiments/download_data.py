"""Download real-world datasets for validating the dtfit methods.

Two series are the domain evidence. They match the dissertation's stated domain
(nonlinear smoothing and forecasting on currency, economic and pandemic time
series) and they have the shape the methods target: models nonlinear in their
parameters, exponential or transcendental.

1. USD/UAH official exchange rate (National Bank of Ukraine open API).
   Window: the 2014-2015 hryvnia crisis, a sustained, roughly exponential
   depreciation from ~8 to ~30 UAH/USD. Currency-domain test.

2. COVID-19 cumulative confirmed cases (JHU CSSE time series, GitHub).
   The early-2020 growth phase is the textbook exponential-in-parameters
   signal. Pandemic-domain test.

The LTSF benchmark sets follow, for the long-horizon forecasting comparison
against the published baselines.

Run:  python -m dtfit_experimental.experiments.download_data
The two series land in experiments/data/ as plain CSV (date,value); the LTSF
sets keep their published layout under experiments/data/ltsf/.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

NBU_URL = (
    "https://bank.gov.ua/NBU_Exchange/exchange_site"
    "?start={start}&end={end}&valcode=usd&sort=exchangedate&order=asc&json"
)
JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dtfit-experiments/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_usd_uah(start: str = "20140101", end: str = "20151231") -> Path:
    """USD/UAH daily official rate over the 2014-2015 hryvnia crisis."""
    raw = _get(NBU_URL.format(start=start, end=end))
    rows = json.loads(raw)
    out = DATA_DIR / "usd_uah_2014_2015.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "rate_uah_per_usd"])
        for r in rows:
            d = datetime.strptime(r["exchangedate"], "%d.%m.%Y").date()
            # NBU historically quoted USD per 100 units. rate_per_unit is the
            # rate for a single USD (7.993 and the like), the one we want.
            w.writerow([d.isoformat(), r["rate_per_unit"]])
    print(f"  USD/UAH: {len(rows)} daily rates -> {out.relative_to(DATA_DIR.parent)}")
    return out


def download_covid_ukraine() -> Path:
    """COVID-19 cumulative confirmed cases for Ukraine (JHU CSSE)."""
    raw = _get(JHU_URL).decode("utf-8")
    reader = csv.reader(io.StringIO(raw))
    header = next(reader)
    dates = header[4:]  # columns after Province/State, Country/Region, Lat, Long
    ua = next(row for row in reader if row[1] == "Ukraine")
    values = ua[4:]
    out = DATA_DIR / "covid_ukraine_confirmed.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "cumulative_confirmed"])
        for d, v in zip(dates, values):
            iso = datetime.strptime(d, "%m/%d/%y").date().isoformat()
            w.writerow([iso, int(v)])
    print(f"  COVID UA: {len(dates)} daily points -> {out.relative_to(DATA_DIR.parent)}")
    return out


LTSF_DIR = DATA_DIR / "ltsf"

# ETT family: small, reliably hosted on the canonical GitHub repo.
ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
ETT_FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]

# The larger LTSF sets live in the Autoformer dataset bundle (Google Drive /
# Tsinghua cloud), which is awkward to fetch programmatically. Each file gets a
# list of candidate mirrors; a file that fetches from none of them is skipped
# with its canonical source printed, and the benchmark runs on whatever subset
# made it down.
_HF_TSLIB = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
LTSF_MIRRORS = {
    "weather.csv": [_HF_TSLIB + "weather/weather.csv"],
    "exchange_rate.csv": [_HF_TSLIB + "exchange_rate/exchange_rate.csv"],
    "electricity.csv": [_HF_TSLIB + "electricity/electricity.csv"],
    "traffic.csv": [_HF_TSLIB + "traffic/traffic.csv"],
}
AUTOFORMER_CANONICAL = (
    "https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy "
    "(Autoformer dataset bundle); place the *.csv into experiments/data/ltsf/")


def _save(url: str, path: Path, timeout: int = 180) -> bool:
    try:
        data = _get(url, timeout=timeout)
        path.write_bytes(data)
        print(f"  {path.name}: {len(data) / 1e6:.1f} MB -> {path.relative_to(DATA_DIR.parent)}")
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort, report and continue
        print(f"  {path.name}: FAILED ({type(exc).__name__})")
        return False


def download_ltsf() -> None:
    """Fetch the LTSF benchmark datasets (ETT reliably; others best-effort)."""
    LTSF_DIR.mkdir(parents=True, exist_ok=True)
    print("LTSF benchmark datasets:")
    for f in ETT_FILES:
        if not (LTSF_DIR / f).exists():
            _save(ETT_BASE + f, LTSF_DIR / f)
    missing = []
    for fname, urls in LTSF_MIRRORS.items():
        if (LTSF_DIR / fname).exists():
            continue
        if not any(_save(u, LTSF_DIR / fname) for u in urls):
            missing.append(fname)
    if missing:
        print(f"  (could not auto-fetch {', '.join(missing)} -- get them from "
              f"{AUTOFORMER_CANONICAL})")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading datasets into {DATA_DIR} ...")
    for fn in (download_usd_uah, download_covid_ukraine, download_ltsf):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__}: FAILED ({type(exc).__name__}: {exc})")
    print("Done. (sunspots / CO2 for the forecasting experiment come from "
          "statsmodels.datasets at run time -- no download needed.)")


if __name__ == "__main__":
    main()
