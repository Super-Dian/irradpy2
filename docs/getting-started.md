# Getting started

This walkthrough takes you from raw observations to evaluation metrics using
`irradpy2`. All examples assume:

```python
import numpy as np
import pandas as pd
from irradpy2.tool import (
    read_csv,
    cal_zen,
    prepare_clearsky,
    cal_metrics,
    plot_series,
)
```

## 1. Obtain data

### Ground observations

```python
from irradpy2.getirrad import download_bsrn

download_bsrn(
    site="CAB",
    start="2025-01-01",
    end="2025-01-31",
    username="your_bsrn_username",
    password="your_bsrn_password",
    save_path="cab_bsrn.csv",
)
```

Credential-free sources include MIDC, SAURAN, SOLRAD, SRML, and SURFRAD — see
[Data acquisition](user-guide/data-acquisition.md).

### Gridded meteorological data

```python
from irradpy2.getgridded import download_merra2

download_merra2(
    collection_names=["rad", "slv", "aer"],
    mode="point",
    lat=51.968,
    lon=4.928,
    start="2025-01-01",
    end="2025-01-31",
    save_path="cab_merra2.csv",
    username="your_earthdata_username",
    password="your_earthdata_password",
)
```

## 2. Read and align data

```python
ground = read_csv("cab_bsrn.csv")
merra2 = read_csv("cab_merra2.csv")
```

`read_csv` locates the time column automatically among common names
(`UTC`, `Timestamp`, `datetime`, `date`, `time`, …).

To resample or merge series on a uniform time grid:

```python
from irradpy2.tool import select_series, resample_series, merge_series

hourly = resample_series(
    ground,
    time_col="UTC",
    cols=["ghi"],
    freq="1h",
    agg="mean",
    lower_limit=0,
)

merged = merge_series([ground, merra2], time_col="UTC", dropna=True)
```

## 3. Solar geometry

```python
data = cal_zen(
    ground,
    latitude=51.968,
    longitude=4.928,
    time_col="UTC",
    zen_col="zen",   # degrees
)
```

> **Important.** `cal_zen` produces the zenith angle in **degrees**.
> Clear-sky models expect **radians** (`sza`). Convert with
> `sza = np.radians(data["zen"].to_numpy())`, or use `prepare_clearsky`
> (below), which produces `sza` in radians directly.

## 4. Prepare clear-sky model inputs

`prepare_clearsky` combines ground observations with MERRA-2 auxiliary
variables on a complete time index:

```python
prepared = prepare_clearsky(
    ground,
    merra2,
    latitude=51.968,
    longitude=4.928,
    begin="2025-01-01",
    end="2025-01-31",
)
```

It identifies time columns, builds a complete time index (default 1-minute),
aligns and interpolates meteorological variables, converts atmospheric
quantities, and computes `sza` (radians).

Typical output columns:

```text
UTC, ghi, dni, dhi, sza, press, albedo, wv,
aod700, ang_alpha, ang_beta, ozone_input
```

## 5. Run a clear-sky model

```python
from irradpy2.clear_sky import tj, rest2v5

# simple model: only needs zenith angle and time
ghi_clear, dni_clear, dhi_clear = tj(
    prepared["sza"].to_numpy(),
    prepared["UTC"].to_numpy(),
)

# fully parameterised model
ghi_clear, dni_clear, dhi_clear = rest2v5(
    prepared["sza"].to_numpy(),
    prepared["UTC"].to_numpy(),
    prepared["press"].to_numpy(),
    prepared["albedo"].to_numpy(),
    prepared["ang_alpha"].to_numpy(),
    prepared["ang_beta"].to_numpy(),
    prepared["ozone_input"].to_numpy(),
    prepared["NO2"].to_numpy() if "NO2" in prepared else np.zeros(len(prepared)),
    prepared["wv"].to_numpy(),
)
```

> **Return ordering.** Clear-sky models return
> `[ghi_clear, dni_clear, dhi_clear]` in that order.

62 models are available; see
[Clear-sky models](user-guide/clear-sky-models.md) for the input requirements
of each.

## 6. Detect clear-sky periods

```python
from irradpy2.csd import Ellis2018CSD

csd = Ellis2018CSD(
    prepared["ghi"].to_numpy(),   # observed GHI
    ghi_clear,                    # clear-sky GHI from step 5
)
```

Each CSD method returns a flag array where **1 = clouds suspected, 0 = clear**.

## 7. Separate GHI into diffuse and direct components

```python
from irradpy2.separation import engerer2

kd, dhi_sep, dni_sep = engerer2(
    ghi=prepared["ghi"].to_numpy(),
    datetime=prepared["UTC"].to_numpy(),
    lat=51.968,
    lon=4.928,
    averaging_period=1,
)
```

Returns `(kd, dhi, dni)` — the diffuse fraction and the two components.

## 8. Forecast GHI

```python
from irradpy2.forecast import forecast_persistence

ori_data, pre_data = forecast_persistence(
    prepared["UTC"].to_numpy(),
    prepared["ghi"].to_numpy(),
    horizon=6,
)
# ori_data: columns Timestamp, Global
# pre_data: columns Timestamp, pre_Global
```

Deep-learning variants (`forecast_rnn`, `forecast_lstm`, `forecast_informer`)
add `seq_len`, `epochs`, `batch_size`, and model-specific parameters.
See [Forecasting](user-guide/forecasting.md).

## 9. Evaluate and visualize

```python
from irradpy2.tool import cal_metrics, plot_series, filter_sep_data

result = ori_data.assign(ghi_pred=pre_data["pre_Global"].to_numpy())
metrics = cal_metrics(result, obs_col="Global", pred_col="ghi_pred")
print(metrics)   # {'MSE': ..., 'RMSE': ..., 'MAE': ..., 'MBE': ..., 'R2': ...}

plot_series(
    result,
    obs_col="Global",
    pred_col="ghi_pred",
    obs_label="Observed GHI",
    pred_label="Predicted GHI",
    title="GHI Forecast",
    ylabel=r"GHI [W m$^{-2}$]",
)
```

For separation evaluation, `filter_sep_data` removes low-sun and low-irradiance
records before computing metrics:

```python
filtered = filter_sep_data(
    data,
    latitude=51.968,
    longitude=4.928,
    max_zen=85,
    min_ghi=50,
)
```

## Next steps

- Browse the per-function [API reference pages](index.md#reference).
- Read the [data conventions](user-guide/data-conventions.md) before mixing
  outputs from different model families.
