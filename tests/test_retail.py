import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from demand_forecast import retail
from demand_forecast.panel import panel_from_long


def _raw(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"])


def test_clean_transactions_keeps_only_product_sales() -> None:
    raw = _raw(
        [
            ("500001", "22423", 4, "2010-01-04 10:00", 2.5),  # kept
            ("500001", " 85123a ", 6, "2010-01-04 10:00", 2.5),  # kept, code normalised
            ("C500002", "22423", -4, "2010-01-05 09:00", 2.5),  # cancellation
            ("500003", "22423", -2, "2010-01-05 09:00", 2.5),  # negative quantity
            ("500004", "22423", 3, "2010-01-05 09:00", 0.0),  # free / adjustment line
            ("500005", "POST", 1, "2010-01-05 09:00", 18.0),  # postage
            ("500006", "gift_0001_20", 1, "2010-01-05 09:00", 20.0),  # voucher
            ("500007", "BANK CHARGES", 1, "2010-01-05 09:00", 15.0),
        ]
    )
    out = retail.clean_transactions(raw)
    assert out["StockCode"].tolist() == ["22423", "85123A"]
    assert out["Quantity"].tolist() == [4.0, 6.0]
    assert (out["day"] == pd.Timestamp("2010-01-04")).all()


def test_daily_panel_selects_on_first_year_only_and_fills_zeros() -> None:
    days = pd.date_range("2010-01-01", "2011-06-30", freq="D")
    early = days[days < "2010-12-01"][::3]  # 'A' sells regularly in the first year
    late = days[days >= "2010-12-01"]  # 'B' sells every day, but only later
    sales = pd.concat(
        [
            pd.DataFrame({"day": early, "StockCode": "A", "Quantity": 2.0}),
            pd.DataFrame({"day": late, "StockCode": "B", "Quantity": 9.0}),
            pd.DataFrame({"day": early[:5], "StockCode": "C", "Quantity": 1.0}),
        ]
    )
    frame = retail.daily_panel(sales, n_series=1, future_days=5)
    assert frame["series_id"].unique().tolist() == ["A"]
    panel = panel_from_long(frame)
    assert panel.n_obs == len(pd.date_range(early.min(), late.max(), freq="D"))
    assert panel.observed.sum() == 2.0 * len(early)
    assert (panel.observed == 0).any()  # gaps became zeros
    assert np.isnan(panel.y[panel.n_obs :]).all() and len(panel.dates) == panel.n_obs + 5
    holidays = panel.dates[panel.known["holiday"][:, 0] == 1]
    assert pd.Timestamp("2010-12-27") in holidays and pd.Timestamp("2011-04-29") in holidays
    assert set(holidays) <= set(retail.BANK_HOLIDAYS)


def test_read_workbook_removes_the_sheet_overlap(tmp_path) -> None:
    first = _raw(
        [("1", "10001", 1, pd.Timestamp("2010-11-30 10:00"), 1.0),
         ("2", "10001", 2, pd.Timestamp("2010-12-02 10:00"), 1.0)]
    )  # fmt: skip
    second = _raw(
        [("2", "10001", 2, pd.Timestamp("2010-12-02 10:00"), 1.0),
         ("3", "10001", 3, pd.Timestamp("2011-01-04 10:00"), 1.0)]
    )  # fmt: skip
    xlsx = tmp_path / retail.WORKBOOK
    with pd.ExcelWriter(xlsx) as writer:
        first.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        second.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(xlsx, retail.WORKBOOK)
    out = retail.read_workbook(archive)
    assert out["Invoice"].tolist() == ["1", "2", "3"]


def test_download_verifies_checksum(tmp_path) -> None:
    source = tmp_path / "remote.bin"
    source.write_bytes(b"retail data")
    good = hashlib.sha256(b"retail data").hexdigest()
    dest = tmp_path / "raw" / "file.zip"
    assert retail.download(dest, url=source.as_uri(), sha256=good).read_bytes() == b"retail data"
    # already present with the right checksum: no network access needed
    assert retail.download(dest, url="http://invalid.invalid/x", sha256=good) == dest
    other = tmp_path / "raw" / "other.zip"
    with pytest.raises(ValueError, match="checksum mismatch"):
        retail.download(other, url=source.as_uri(), sha256="0" * 64)
    assert not other.exists() and not other.with_suffix(".zip.part").exists()
