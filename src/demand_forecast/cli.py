"""Command line: ``demand-forecast generate | download | backtest | forecast``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from demand_forecast import retail
from demand_forecast.backtest import BacktestConfig, forecast_future, run_backtest
from demand_forecast.models import MODEL_NAMES, build_models
from demand_forecast.panel import read_panel
from demand_forecast.report import write_report
from demand_forecast.synthetic import SyntheticConfig, generate


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _backtest_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", type=Path, required=True, help="long CSV (see panel.py)")
    p.add_argument("--horizon", type=int, default=28)
    p.add_argument("--step", type=int, default=28, help="days between origins")
    p.add_argument("--calibration-origins", type=int, default=4,
                   help="earlier origins used only to calibrate conformal intervals")  # fmt: skip
    p.add_argument("--season-length", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.2, help="0.2 gives 80 %% intervals")
    p.add_argument("--min-train", type=int, default=365)
    p.add_argument("--jobs", type=int, default=1, help="parallel Holt-Winters fits")
    p.add_argument("--gbm-max-iter", type=int, default=150)
    p.add_argument("--gbm-threads", type=int, default=None,
                   help="cap OpenMP threads for the gradient-boosting fits")  # fmt: skip


def _config(args: argparse.Namespace, n_origins: int) -> BacktestConfig:
    return BacktestConfig(
        horizon=args.horizon,
        step=args.step,
        n_origins=n_origins,
        calibration_origins=args.calibration_origins,
        season_length=args.season_length,
        alpha=args.alpha,
        min_train=args.min_train,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = SyntheticConfig(
        n_stores=args.stores,
        n_items=args.items,
        start=args.start,
        n_days=args.days,
        future_days=args.future_days,
        seed=args.seed,
    )
    frame = generate(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    _log(f"wrote {len(frame):,} rows ({cfg.n_stores * cfg.n_items} series) to {args.out}")


def cmd_download(args: argparse.Namespace) -> None:
    _log(f"downloading {retail.URL} (about 46 MB) into {args.raw_dir} ...")
    frame = retail.build(args.raw_dir, n_series=args.series, future_days=args.future_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    _log(f"wrote {len(frame):,} rows ({frame['series_id'].nunique()} products) to {args.out}")


def cmd_backtest(args: argparse.Namespace) -> None:
    started, cpu_started = time.perf_counter(), time.process_time()
    panel = read_panel(args.data)
    cfg = _config(args, args.origins)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    models = build_models(
        names,
        alpha=cfg.alpha,
        n_jobs=args.jobs,
        gbm_max_iter=args.gbm_max_iter,
        gbm_threads=args.gbm_threads,
    )
    _log(f"{panel.n_series} series x {panel.n_obs} days; models: {', '.join(models)}")
    result = run_backtest(panel, models, cfg, log=_log)
    summary = write_report(
        result,
        panel,
        args.name or args.data.stem,
        args.out,
        time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu_started,
        save_points=args.save_points,
    )
    print((args.out / "leaderboard.md").read_text())
    _log(f"wrote {args.out}/: {summary['wall_seconds']} s wall, {summary['cpu_seconds']} s CPU")


def cmd_forecast(args: argparse.Namespace) -> None:
    panel = read_panel(args.data)
    cfg = _config(args, 1)
    model = build_models(
        [args.model],
        alpha=cfg.alpha,
        n_jobs=args.jobs,
        gbm_max_iter=args.gbm_max_iter,
        gbm_threads=args.gbm_threads,
    )[args.model]
    frame = forecast_future(panel, model, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, float_format="%.3f")
    _log(f"wrote {len(frame):,} forecasts from {frame['date'].min()} to {args.out}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="demand-forecast", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write the seeded synthetic multi-store panel")
    p.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    p.add_argument("--stores", type=int, default=5)
    p.add_argument("--items", type=int, default=20)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--days", type=int, default=1096)
    p.add_argument("--future-days", type=int, default=28)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("download", help="download and aggregate UCI Online Retail II")
    p.add_argument("--out", type=Path, default=Path("data/online_retail.csv"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--series", type=int, default=100)
    p.add_argument("--future-days", type=int, default=28)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("backtest", help="rolling-origin backtest and leaderboards")
    _backtest_options(p)
    p.add_argument("--origins", type=int, default=8, help="scored origins")
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--out", type=Path, default=Path("reports/run"))
    p.add_argument("--name", help="dataset label in the report (default: file name)")
    p.add_argument("--save-points", action="store_true", help="also write every forecast")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("forecast", help="forecast past the end of the data with intervals")
    _backtest_options(p)
    p.add_argument("--model", default="gbm", choices=MODEL_NAMES)
    p.add_argument("--out", type=Path, default=Path("data/forecast.csv"))
    p.set_defaults(func=cmd_forecast)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
