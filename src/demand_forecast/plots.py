"""One results figure: MASE by horizon, and one series' last forecast with its interval."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demand_forecast.backtest import EVALUATION

# Fixed model -> colour mapping, so a model keeps its colour across charts and runs.
COLOR_ORDER = ("gbm", "holt_winters", "moving_average", "seasonal_naive", "naive")
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _color(model: str, models: list[str]) -> str:
    order = [*COLOR_ORDER, *sorted(m for m in models if m not in COLOR_ORDER)]
    return SERIES_COLORS[order.index(model) % len(SERIES_COLORS)]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def example_series(series_table: pd.DataFrame, model: str) -> str:
    """The series whose MASE for ``model`` is the median one: a typical case, not the best."""
    rows = series_table[series_table["model"] == model].sort_values("mase")
    return str(rows.iloc[len(rows) // 2]["series_id"])


def results_chart(
    points: pd.DataFrame,
    horizon_table: pd.DataFrame,
    series_table: pd.DataFrame,
    best_model: str,
    observed: pd.Series,
    title: str,
    path: Path,
) -> Path:
    models = list(horizon_table.columns)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4), dpi=110)
    fig.patch.set_facecolor(SURFACE)

    labels = [str(v) for v in horizon_table.index]
    for model in models:
        left.plot(
            labels,
            horizon_table[model].to_numpy(),
            color=_color(model, models),
            linewidth=1.6,
            marker="o",
            markersize=5,
            label=model,
        )
    left.set_title("MASE by forecast horizon (lower is better)", fontsize=10, color=INK, loc="left")
    left.set_xlabel("days ahead", fontsize=9, color=INK_SECONDARY)
    left.legend(
        fontsize=8, frameon=False, labelcolor=INK, loc="center left", bbox_to_anchor=(1, 0.5)
    )
    _style(left)

    series = example_series(series_table, best_model)
    scored = points[(points["role"] == EVALUATION) & (points["model"] == best_model)]
    last_origin = scored["origin"].max()
    fc = scored[(scored["series_id"] == series) & (scored["origin"] == last_origin)]
    history = observed.loc[: fc["date"].max()].tail(len(fc) + 56)
    color = _color(best_model, models)
    right.fill_between(fc["date"], fc["lower"], fc["upper"], color=color, alpha=0.18, linewidth=0,
                       label=f"{best_model} conformal interval")  # fmt: skip
    right.plot(history.index, history.to_numpy(), color=INK_SECONDARY, linewidth=1.0,
               label="actual")  # fmt: skip
    right.plot(
        fc["date"], fc["forecast"], color=color, linewidth=1.6, label=f"{best_model} forecast"
    )
    right.axvline(last_origin, color=GRID, linewidth=1.2)
    right.set_title(f"Series {series}: last scored origin", fontsize=10, color=INK, loc="left")
    right.set_ylabel("units per day", fontsize=9, color=INK_SECONDARY)
    right.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left")
    right.tick_params(axis="x", rotation=30)
    _style(right)

    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
