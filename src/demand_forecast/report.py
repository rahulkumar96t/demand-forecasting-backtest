"""Write leaderboards, a metrics JSON and the chart for one backtest run."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from demand_forecast.backtest import BacktestResult
from demand_forecast.leaderboard import (
    by_horizon,
    interval_table,
    overall,
    per_series,
    point_terms,
    to_markdown,
)
from demand_forecast.panel import Panel
from demand_forecast.plots import example_series, results_chart

PACKAGES = ("numpy", "pandas", "scikit-learn", "statsmodels")


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": str(os.cpu_count()),
        **{pkg: version(pkg) for pkg in PACKAGES},
    }


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    clean = df.astype(object).where(df.notna(), None)
    return [
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def write_report(
    result: BacktestResult,
    panel: Panel,
    dataset: str,
    out_dir: Path,
    wall_seconds: float,
    save_points: bool = False,
    cpu_seconds: float | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    terms = point_terms(result.points, cfg.alpha)
    series_table = per_series(terms)
    board = overall(terms, series_table, result.runs)
    intervals = interval_table(terms)
    horizon = by_horizon(terms)
    weekly = by_horizon(terms, width=7)

    series_table.to_csv(out_dir / "per_series.csv", index=False, float_format="%.4f")
    board.to_csv(out_dir / "leaderboard.csv", index=False, float_format="%.4f")
    if save_points:
        result.points.to_csv(out_dir / "points.csv.gz", index=False, float_format="%.4f")

    best = str(board.iloc[0]["model"])
    example_id = example_series(series_table, best)
    observed = pd.Series(
        panel.observed[:, panel.series_ids.index(example_id)], index=panel.dates[: panel.n_obs]
    )
    results_chart(result.points, weekly, series_table, best, observed,
                  f"{dataset}: rolling-origin backtest", out_dir / "results.png")  # fmt: skip

    scored = result.points[result.points["role"] == "evaluation"]
    summary: dict[str, object] = {
        "dataset": dataset,
        "series": panel.n_series,
        "observed_days": panel.n_obs,
        "first_date": str(panel.dates[0].date()),
        "last_observed_date": str(panel.dates[panel.n_obs - 1].date()),
        "scored_origins": [str(d.date()) for d in sorted(scored["origin"].unique())],
        "scored_points_per_model": int(len(scored) / max(scored["model"].nunique(), 1)),
        "config": cfg.as_dict(),
        "wall_seconds": round(wall_seconds, 1),
        "cpu_seconds": None if cpu_seconds is None else round(cpu_seconds, 1),
        "environment": environment(),
        "leaderboard": _records(board),
        "intervals": _records(intervals),
        "mase_by_horizon_day": {
            str(m): [round(float(v), 4) for v in horizon[m].to_numpy()] for m in horizon.columns
        },
        "mase_by_horizon_week": {
            str(m): dict(zip(weekly.index.astype(str), weekly[m].round(4).tolist(), strict=True))
            for m in weekly.columns
        },
        "diagnostics": _records(
            result.runs.drop(columns=["origin", "role"]).groupby("model").sum().reset_index()
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"# Backtest: {dataset}",
        "",
        f"{panel.n_series} series, {panel.n_obs} observed days "
        f"({summary['first_date']} to {summary['last_observed_date']}); "
        f"horizon {cfg.horizon}, step {cfg.step}, {cfg.n_origins} scored origins after "
        f"{cfg.calibration_origins} calibration origins; "
        f"{int((1 - cfg.alpha) * 100)} % intervals. Run time {wall_seconds:.0f} s wall"
        + ("" if cpu_seconds is None else f", {cpu_seconds:.0f} s CPU")
        + ".",
        "",
        "## Overall",
        "",
        to_markdown(board),
        "",
        "## Prediction intervals",
        "",
        to_markdown(intervals),
        "",
        "## MASE by horizon week",
        "",
        to_markdown(weekly.reset_index()),
        "",
    ]
    (out_dir / "leaderboard.md").write_text("\n".join(lines))
    return summary
