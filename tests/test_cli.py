import json

import pandas as pd

from demand_forecast.cli import main


def test_generate_backtest_forecast(tmp_path, capsys) -> None:
    data = tmp_path / "synthetic.csv"
    main(["generate", "--out", str(data), "--stores", "1", "--items", "3", "--days", "260",
          "--future-days", "14", "--seed", "3"])  # fmt: skip
    assert pd.read_csv(data)["series_id"].nunique() == 3

    out = tmp_path / "report"
    main(["backtest", "--data", str(data), "--out", str(out), "--horizon", "7", "--step", "7",
          "--origins", "2", "--calibration-origins", "8", "--min-train", "120",
          "--models", "naive,seasonal_naive,moving_average", "--save-points"])  # fmt: skip
    summary = json.loads((out / "metrics.json").read_text())
    assert {row["model"] for row in summary["leaderboard"]} == {
        "naive", "seasonal_naive", "moving_average",
    }  # fmt: skip
    assert (out / "results.png").exists() and (out / "points.csv.gz").exists()
    assert "## Overall" in capsys.readouterr().out

    forecast = tmp_path / "forecast.csv"
    main(["forecast", "--data", str(data), "--model", "seasonal_naive", "--horizon", "7",
          "--calibration-origins", "8", "--out", str(forecast)])  # fmt: skip
    frame = pd.read_csv(forecast)
    assert len(frame) == 7 * 3
    assert frame["date"].min() > summary["last_observed_date"]
