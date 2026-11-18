# Backtest: UCI Online Retail II (top 100 products)

100 series, 739 observed days (2009-12-01 to 2011-12-09); horizon 28, step 28, 8 scored origins after 4 calibration origins; 80 % intervals. Run time 198 s wall, 191 s CPU.

## Overall

| model | mase | mase_median | smape | mae | rmse | bias_pct | coverage | msis | wins | cpu_s_per_origin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gbm | 0.690 | 0.673 | 120.043 | 24.405 | 77.569 | -10.281 | 0.803 | 3.932 | 53 | 3.855 |
| holt_winters | 0.732 | 0.706 | 98.992 | 26.105 | 77.771 | -1.670 | 0.805 | 4.047 | 26 | 9.683 |
| moving_average | 0.783 | 0.772 | 113.542 | 28.188 | 79.601 | -5.109 | 0.786 | 4.131 | 4 | 0.000 |
| seasonal_naive | 0.889 | 0.829 | 95.440 | 32.984 | 117.790 | -1.041 | 0.820 | 5.510 | 4 | 0.000 |
| naive | 0.916 | 0.839 | 119.689 | 36.769 | 173.119 | 0.968 | 0.802 | 5.257 | 18 | 0.000 |

## Prediction intervals

| model | method | coverage | scaled_width | msis |
|---|---|---:|---:|---:|
| gbm | quantile | 0.889 | 1.522 | 3.676 |
| gbm | conformal | 0.803 | 1.540 | 3.932 |
| holt_winters | conformal | 0.805 | 1.552 | 4.047 |
| moving_average | conformal | 0.786 | 1.655 | 4.131 |
| naive | conformal | 0.802 | 1.718 | 5.257 |
| seasonal_naive | conformal | 0.820 | 1.664 | 5.510 |

## MASE by horizon week

| days_ahead | naive | seasonal_naive | moving_average | holt_winters | gbm |
|---|---:|---:|---:|---:|---:|
| 1-7 | 0.874 | 0.830 | 0.755 | 0.712 | 0.652 |
| 8-14 | 0.956 | 0.918 | 0.815 | 0.762 | 0.706 |
| 15-21 | 0.915 | 0.886 | 0.782 | 0.727 | 0.702 |
| 22-28 | 0.921 | 0.924 | 0.780 | 0.727 | 0.699 |
