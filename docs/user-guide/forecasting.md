# GHI forecasting

The `irradpy2.forecast` package provides **5 global horizontal irradiance
forecasting methods** spanning classical baselines and deep-learning models.

```python
from irradpy2.forecast import (
    forecast_persistence,
    forecast_arima,
    forecast_rnn,
    forecast_lstm,
    forecast_informer,
)
```

## Calling convention

All methods share the same core signature shape:

```python
ori_data, pre_data = forecast_persistence(datetime, ghi, horizon=6, test_ratio=0.15)
```

| Argument | Meaning | Unit / type |
|---|---|---|
| `datetime` | UTC timestamps, length `N` | datetime |
| `ghi` | observed global horizontal irradiance, length `N` | W m⁻² |
| `horizon` | forecast horizon (number of steps ahead) | steps |
| `test_ratio` | fraction of the series held out for evaluation | unitless |
| `seq_len` | input sequence length (deep learning only) | steps |
| `epochs`, `batch_size`, `lr` | training parameters (deep learning only) | — |

### Returns

Both return values are `pandas.DataFrame`s aligned on `Timestamp`:

| Value | Columns | Meaning |
|---|---|---|
| `ori_data` | `Timestamp`, `Global` | held-out observed GHI |
| `pre_data` | `Timestamp`, `pre_Global` | predicted GHI |

Column names are fixed: observed GHI is `Global`, predicted GHI is
`pre_Global` (input `ghi` values are likewise placed in a `Global` column
internally after sorting by `Timestamp` and clipping negatives to 0).

## Method catalogue

| Method | Signature | Type |
|---|---|---|
| `forecast_persistence` | `(datetime, ghi, horizon=6, test_ratio=0.15)` | baseline |
| `forecast_arima` | `(datetime, ghi, order=(2, 1, 2), test_ratio=0.15)` | classical |
| `forecast_rnn` | `(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=1e-3, hidden=128)` | deep learning |
| `forecast_lstm` | `(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=1e-3, hidden=128)` | deep learning |
| `forecast_informer` | `(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=1e-3, d_model=128, nhead=4, num_layers=2)` | deep learning |

Deep-learning models are implemented in PyTorch (`SeqDataset`, `RNNModel`,
`LSTMModel`, `InformerModel`) and train on CPU by default. The last
`test_ratio` portion of the series is held out; training windows are built by
sliding `seq_len`-length inputs to predict `horizon` steps ahead.

## Worked example

```python
from irradpy2.tool import read_csv, cal_metrics, plot_series
from irradpy2.forecast import forecast_lstm

data = read_csv("station_2025.csv")

ori_data, pre_data = forecast_lstm(
    data["UTC"].to_numpy(),
    data["ghi"].to_numpy(),
    seq_len=72,
    horizon=6,
    epochs=10,
    batch_size=64,
)

# ori_data: Timestamp, Global      pre_data: Timestamp, pre_Global
result = ori_data.assign(ghi_pred=pre_data["pre_Global"].to_numpy())
print(cal_metrics(result, obs_col="Global", pred_col="ghi_pred"))

plot_series(
    result,
    obs_col="Global",
    pred_col="ghi_pred",
    obs_label="Observed GHI",
    pred_label="Forecast GHI",
    title="GHI forecast (LSTM)",
    ylabel=r"GHI [W m$^{-2}$]",
)
```

## Practical guidance

- **Persistence** is the mandatory baseline: report skill scores relative to
  persistence, not just absolute error.
- **ARIMA** works well on hourly and sub-hourly series with modest seasonality
  left after diurnal normalisation.
- **Deep-learning methods** need substantially more data (weeks of continuous
  records) and benefit from normalised inputs; start with `epochs=10` for
  smoke tests before full training.
- Evaluate with `irradpy2.tool.cal_metrics` (RMSE, MAE, MBE, R²) and plot with
  `plot_series`.

## Dependencies

The forecasting module requires `torch` (deep learning) and `scikit-learn`
(metrics/normalisation utilities). These are core package dependencies; see
[installation](../installation.md).
