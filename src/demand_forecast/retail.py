"""UCI Online Retail II: download, verify, clean and aggregate to daily units per product.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D, licensed CC BY 4.0. All transactions of a UK-based online
gift-ware retailer between 2009-12-01 and 2011-12-09; many customers are wholesalers.

Aggregation rules (each one is unit-tested on a hand-made frame):

* the workbook has one sheet per year and the sheets overlap on 2010-12-01..09, so the first
  sheet is cut at 2010-12-01;
* cancellations (invoice starting with ``C``), non-positive quantities or prices and
  non-product codes (postage, fees, adjustments, gift vouchers) are dropped;
* the panel keeps the ``n_series`` products with the most selling days in the *first year*
  only (before ``selection_end``), so the choice of products does not look at the backtest
  period and products that were later discontinued stay in;
* days without sales are zeros; ``holiday`` marks England and Wales bank holidays.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.panel import DATE, SERIES, TARGET

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
WORKBOOK = "online_retail_II.xlsx"
FIRST_SHEET_END = "2010-12-01"
PRODUCT_CODE = r"\d{5}[A-Za-z]{0,2}"

# England and Wales bank holidays (gov.uk), including substitute days and 2011-04-29.
BANK_HOLIDAYS = pd.to_datetime(
    [
        "2009-12-25", "2009-12-28",
        "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30",
        "2010-12-27", "2010-12-28",
        "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30",
        "2011-08-29", "2011-12-26", "2011-12-27",
        "2012-01-02",
    ]
)  # fmt: skip


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dest: Path, url: str = URL, sha256: str = SHA256, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` unless a file with the expected checksum is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == sha256:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256_of(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
    tmp.replace(dest)
    return dest


def read_workbook(zip_path: Path) -> pd.DataFrame:
    """Read both yearly sheets from the zipped workbook and remove their overlap."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(WORKBOOK) as fh:
        sheets = pd.read_excel(
            io.BytesIO(fh.read()),
            sheet_name=None,
            dtype={"Invoice": str, "StockCode": str},
        )
    first, second = sheets.values()
    first = first[first["InvoiceDate"] < FIRST_SHEET_END]
    return pd.concat([first, second], ignore_index=True)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep real product sales; return columns day, StockCode, Quantity."""
    invoice = raw["Invoice"].astype(str)
    code = raw["StockCode"].astype(str).str.strip()
    keep = (
        ~invoice.str.startswith("C")
        & (raw["Quantity"] > 0)
        & (raw["Price"] > 0)
        & code.str.fullmatch(PRODUCT_CODE)
    )
    out = pd.DataFrame(
        {
            "day": pd.to_datetime(raw.loc[keep, "InvoiceDate"]).dt.normalize(),
            "StockCode": code[keep].str.upper(),
            "Quantity": raw.loc[keep, "Quantity"].astype(float),
        }
    )
    return out.reset_index(drop=True)


def daily_panel(
    sales: pd.DataFrame,
    n_series: int = 100,
    selection_end: str = "2010-12-01",
    future_days: int = 28,
) -> pd.DataFrame:
    """Daily units for the top products, in the long format of :mod:`demand_forecast.panel`."""
    first_year = sales[sales["day"] < selection_end]
    selling_days = first_year.groupby("StockCode")["day"].nunique()
    # ties broken by code so the selection is deterministic
    ranked = selling_days.reset_index().sort_values(["day", "StockCode"], ascending=[False, True])
    chosen = ranked["StockCode"].head(n_series).tolist()

    observed = pd.date_range(sales["day"].min(), sales["day"].max(), freq="D")
    units = (
        sales[sales["StockCode"].isin(chosen)]
        .pivot_table(index="day", columns="StockCode", values="Quantity", aggfunc="sum")
        .reindex(index=observed, columns=chosen)
        .fillna(0.0)
    )
    future = pd.date_range(observed[-1], periods=future_days + 1, freq="D")[1:]
    all_dates = observed.append(future)
    values = np.vstack([units.to_numpy(), np.full((future_days, len(chosen)), np.nan)])
    holiday = all_dates.isin(BANK_HOLIDAYS).astype(float)
    return pd.DataFrame(
        {
            DATE: np.repeat(all_dates.values, len(chosen)),
            SERIES: np.tile(np.array(chosen, dtype=object), len(all_dates)),
            "holiday": np.repeat(holiday, len(chosen)),
            TARGET: values.ravel(),
        }
    )


def build(raw_dir: Path, n_series: int = 100, future_days: int = 28) -> pd.DataFrame:
    """Download (if needed) and aggregate the dataset."""
    zip_path = download(raw_dir / "online_retail_ii.zip")
    return daily_panel(
        clean_transactions(read_workbook(zip_path)), n_series=n_series, future_days=future_days
    )
