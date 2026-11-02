# Data tools

The `irradpy2.tool` module provides reading, time-series, solar-geometry,
evaluation, and visualisation utilities used across the package.

```python
from irradpy2.tool import (
    read_csv,
    select_series,
    resample_series,
    merge_series,
    merge_columns,
    cal_zen,
    cal_exth,
    cal_doy_year,
    prepare_clearsky,
    filter_sep_data,
    cal_metrics,
    cal_mse, cal_rmse, cal_mae, cal_mbe, cal_r2,
    plot_columns, plot_clearsky, plot_csd, plot_series,
)
```

## Reading

```python
data = read_csv("irradiance.csv")
```

`read_csv` wraps `pandas.read_csv`. Time-column auto-detection lives in the
helpers that need it: `cal_zen` looks for `Timestamp`, `UTC`, `UTC_time`, or
`time`, and `prepare_clearsky` looks for `UTC`, `UTC_time`, or `time`.

## Time-series utilities

### select_series

```python
subset = select_series(data, time_col="UTC", cols=["ghi", "dni"], lower_limit=0)
```

Selects columns and optionally drops values below `lower_limit`.

### resample_series

```python
hourly = resample_series(
    data,
    time_col="UTC",
    cols=["ghi"],
    freq="1h",
    agg="mean",        # aggregation: "mean", "sum", "max", ...
    lower_limit=0,     # clip negatives before aggregating
    label_offset=None, # bin-label alignment offset
)
```

### merge_series / merge_columns

```python
merged = merge_series([ground, merra2, cams], time_col="UTC", dropna=True)

# merge arbitrary arrays into a DataFrame
merged = merge_columns(data, values=[arr1, arr2], columns=["ghi_clear", "dni_clear"])
```

## Solar geometry

| Function | Output | Unit |
|---|---|---|
| `cal_zen(data, latitude, longitude, time_col=None, zen_col="zen")` | solar zenith angle | **degrees** |
| `cal_exth(data, time_col="UTC", zen_col="zen", exth_col="exth")` | horizontal extraterrestrial irradiance | W m⁻² |
| `cal_doy_year(data, time_col="UTC", doy_col="doy", year_col="year")` | day-of-year and year | — |

Internally, `prepare_clearsky` computes `sza` in **radians** for the clear-sky
models (the private `_cal_sza` helper). When calling models yourself, convert
with `np.radians(zen)`.

## prepare_clearsky

```python
prepared = prepare_clearsky(
    ground_data,
    merra2_data,
    latitude=51.968,
    longitude=4.928,
    begin="2025-01-01",
    end="2025-01-31",
    freq="1min",
    source_prefix="merra2",
)
```

Pipeline: identify time columns → build complete time index at `freq` → align
ground and MERRA-2 records → interpolate meteorological variables → convert
atmospheric quantities → compute `sza` (radians). Typical output columns:

```text
UTC, ghi, dni, dhi, sza, press, albedo, wv,
aod700, ang_alpha, ang_beta, ozone_input
```

## filter_sep_data

Filters a table to physically meaningful records for separation evaluation:

```python
filtered = filter_sep_data(
    data,
    latitude=51.968,
    longitude=4.928,
    time_col="Timestamp",
    zen_col="zen",
    ghi_col="Global",
    dni_col="Direct",
    dhi_col="Diffuse",
    pred_cols=None,
    max_zen=85,
    min_ghi=50,
    require_positive=True,
)
```

Computes `zen` when absent, then keeps rows with `zen < max_zen`,
`ghi > min_ghi`, and (optionally) positive observed components.

## Evaluation metrics

```python
metrics = cal_metrics(data, obs_col="ghi", pred_col="ghi_pred")
# {'MSE': ..., 'RMSE': ..., 'MAE': ..., 'MBE': ..., 'R2': ...}
```

Individual metrics: `cal_mse`, `cal_rmse`, `cal_mae`, `cal_mbe`, `cal_r2`.

> **MBE sign convention.** `irradpy2` defines
> `MBE = mean(predicted - observed)`. Positive MBE indicates **overestimation**,
> negative MBE indicates **underestimation**. Some literature uses the opposite
> convention — check before comparing against published values.

## Visualisation

| Function | Purpose |
|---|---|
| `plot_columns(data, y_cols, labels=None, x_col=None, title=..., ylabel=...)` | compare arbitrary columns |
| `plot_clearsky(data, model_cols, x_col=None, components=("ghi", "dni", "dhi"))` | clear-sky model outputs |
| `plot_csd(data, ghi_col="ghi", clear_col="ghi_clear", csd_col="csd", max_points=3000)` | detection flags over irradiance |
| `plot_series(data, obs_col, pred_col, obs_label=..., pred_label=..., title=..., ylabel=..., threshold=0, max_value=None)` | observed vs predicted |

Example:

```python
plot_series(
    data,
    obs_col="ghi",
    pred_col="ghi_pred",
    obs_label="Observed GHI",
    pred_label="Predicted GHI",
    title="GHI Forecast",
    ylabel=r"GHI [W m$^{-2}$]",
)
```

All plotting functions use `matplotlib`; each one creates a figure, calls
`plt.tight_layout()`, then `plt.show()` followed by `plt.close()`. Figures are
displayed immediately and not returned, so capture them with `plt.savefig`
via a matplotlib `Agg` backend or a monkey-patched `plt.show` if you need
files instead of windows.
