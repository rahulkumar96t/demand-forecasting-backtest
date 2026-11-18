# Backtest: Synthetic (5 stores x 20 items)

100 series, 1096 observed days (2022-01-01 to 2024-12-31); horizon 28, step 28, 8 scored origins after 4 calibration origins; 80 % intervals. Run time 207 s wall, 200 s CPU.

## Overall

| model | mase | mase_median | smape | mae | rmse | bias_pct | coverage | msis | wins | cpu_s_per_origin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gbm | 0.744 | 0.739 | 75.151 | 5.748 | 10.628 | 1.010 | 0.774 | 3.379 | 86 | 4.855 |
| holt_winters | 0.804 | 0.801 | 78.271 | 6.354 | 11.787 | 0.906 | 0.775 | 3.669 | 5 | 9.367 |
| moving_average | 0.821 | 0.814 | 78.780 | 6.548 | 12.207 | -0.160 | 0.780 | 3.614 | 4 | 0.000 |
| naive | 1.032 | 0.997 | 84.069 | 8.482 | 15.870 | -7.608 | 0.771 | 4.760 | 4 | 0.000 |
| seasonal_naive | 1.039 | 1.038 | 82.426 | 8.462 | 15.792 | 0.334 | 0.783 | 4.682 | 1 | 0.000 |

## Prediction intervals

| model | method | coverage | scaled_width | msis |
|---|---|---:|---:|---:|
| gbm | conformal | 0.774 | 2.059 | 3.379 |
| gbm | quantile | 0.887 | 2.534 | 3.392 |
| moving_average | conformal | 0.780 | 2.239 | 3.614 |
| holt_winters | conformal | 0.775 | 2.230 | 3.669 |
| seasonal_naive | conformal | 0.783 | 2.618 | 4.682 |
| naive | conformal | 0.771 | 2.578 | 4.760 |

## MASE by horizon week

| days_ahead | naive | seasonal_naive | moving_average | holt_winters | gbm |
|---|---:|---:|---:|---:|---:|
| 1-7 | 1.006 | 1.026 | 0.805 | 0.783 | 0.733 |
| 8-14 | 1.055 | 1.049 | 0.840 | 0.818 | 0.753 |
| 15-21 | 1.030 | 1.034 | 0.808 | 0.795 | 0.735 |
| 22-28 | 1.038 | 1.048 | 0.830 | 0.820 | 0.754 |
