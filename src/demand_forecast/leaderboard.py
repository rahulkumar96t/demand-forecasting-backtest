"""Turn backtest points into per-series and overall leaderboards.

Only rows of scored (``evaluation``) origins count. Per series, MAE / RMSE / sMAPE / bias are
taken over all its scored points; MASE divides each absolute error by the scale of its own
origin. The overall table averages scale-free metrics (MASE, sMAPE, MSIS) over series, so a
large series does not dominate, and reports pooled MAE / RMSE / bias for the volume view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_forecast.backtest import EVALUATION
from demand_forecast.metrics import interval_score, smape_terms


def point_terms(points: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Scored rows with per-point error terms added."""
    df = points[points["role"] == EVALUATION].copy()
    err = df["forecast"] - df["actual"]
    df["error"] = err
    df["abs_error"] = err.abs()
    df["sq_error"] = err**2
    df["scaled_abs_error"] = df["abs_error"] / df["scale"]
    df["smape"] = smape_terms(df["actual"].to_numpy(), df["forecast"].to_numpy())
    for suffix in ("", "_native"):
        lo, hi = df[f"lower{suffix}"], df[f"upper{suffix}"]
        has = lo.notna() & hi.notna()
        df[f"covered{suffix}"] = np.where(has, (df["actual"] >= lo) & (df["actual"] <= hi), np.nan)
        score = interval_score(df["actual"].to_numpy(), lo.to_numpy(), hi.to_numpy(), alpha)
        df[f"scaled_is{suffix}"] = np.where(has, score / df["scale"], np.nan)
        df[f"scaled_width{suffix}"] = np.where(has, (hi - lo) / df["scale"], np.nan)
    return df


def per_series(terms: pd.DataFrame) -> pd.DataFrame:
    g = terms.groupby(["series_id", "model"], sort=True)
    out = pd.DataFrame(
        {
            "mae": g["abs_error"].mean(),
            "rmse": np.sqrt(g["sq_error"].mean()),
            "smape": g["smape"].mean(),
            "mase": g["scaled_abs_error"].mean(),
            "bias": g["error"].mean(),
            "coverage": g["covered"].mean(),
            "msis": g["scaled_is"].mean(),
        }
    ).reset_index()
    out["rank"] = out.groupby("series_id")["mase"].rank(method="min").astype(int)
    return out.sort_values(["series_id", "rank", "model"]).reset_index(drop=True)


def overall(terms: pd.DataFrame, series_table: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    by_series = series_table.groupby("model")
    g = terms.groupby("model")
    # CPU time of fit + predict in this process (wall time also counts other load and sleep)
    scored_runs = runs[runs["role"] == EVALUATION].groupby("model")["cpu_seconds"].mean()
    total_actual = g["actual"].sum()
    out = pd.DataFrame(
        {
            "mase": by_series["mase"].mean(),
            "mase_median": by_series["mase"].median(),
            "smape": by_series["smape"].mean(),
            "mae": g["abs_error"].mean(),
            "rmse": np.sqrt(g["sq_error"].mean()),
            "bias_pct": 100.0 * g["error"].sum() / total_actual.where(total_actual != 0),
            "coverage": g["covered"].mean(),
            "msis": by_series["msis"].mean(),
            "wins": series_table.assign(win=series_table["rank"] == 1)
            .groupby("model")["win"]
            .sum(),
            "cpu_s_per_origin": scored_runs,
        }
    )
    out["wins"] = out["wins"].astype(int)
    return out.sort_values("mase").rename_axis("model").reset_index()


def interval_table(terms: pd.DataFrame) -> pd.DataFrame:
    """Coverage, scaled width and MSIS for every model and interval method that has values."""
    rows = []
    for model, df in terms.groupby("model", sort=False):
        for method, suffix in (("conformal", ""), ("quantile", "_native")):
            covered = df[f"covered{suffix}"].dropna()
            if covered.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "coverage": covered.mean(),
                    "scaled_width": df[f"scaled_width{suffix}"].mean(),
                    "msis": df[f"scaled_is{suffix}"].mean(),
                }
            )
    columns = ["model", "method", "coverage", "scaled_width", "msis"]
    return pd.DataFrame(rows, columns=columns).sort_values("msis").reset_index(drop=True)


def by_horizon(terms: pd.DataFrame, width: int = 1) -> pd.DataFrame:
    """MASE per horizon bucket of ``width`` days (rows) and model (columns).

    With a step that is a multiple of 7, horizon day ``h`` falls on the same weekday at every
    origin, so single days mix horizon and weekday effects; weekly buckets separate them.
    """
    last = int(terms["h"].max())
    start = (terms["h"] - 1) // width * width + 1
    label = start.astype(str) + "-" + np.minimum(start + width - 1, last).astype(str)
    if width == 1:
        label = terms["h"]
    table = terms.assign(days_ahead=label).pivot_table(
        index="days_ahead", columns="model", values="scaled_abs_error", aggfunc="mean", sort=False
    )
    return table.loc[sorted(table.index, key=lambda v: int(str(v).split("-")[0]))]


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Small GitHub-flavoured markdown table without extra dependencies."""

    def fmt(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "-" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    align = (
        "|"
        + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns)
        + "|"
    )
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, align, *body])
