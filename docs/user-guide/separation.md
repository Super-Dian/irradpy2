# Irradiance separation

The `irradpy2.separation` package provides **7 irradiance separation models**
that estimate the diffuse and direct components of global horizontal
irradiance (GHI).

```python
from irradpy2.separation import boland, disc, erbs, engerer2, orgill_hollands, yang4, louche
```

## Calling convention

```python
kd, dhi, dni = engerer2(ghi, datetime, lat, lon, averaging_period=1)
```

### Returns

| Value | Meaning | Unit |
|---|---|---|
| `kd` | diffuse fraction (`dhi / ghi`) | unitless |
| `dhi` | estimated diffuse horizontal irradiance | W m⁻² |
| `dni` | estimated direct normal irradiance | W m⁻² |

### Inputs

| Parameter | Meaning | Unit / type |
|---|---|---|
| `ghi` | global horizontal irradiance, shape `(N,)` or `(N, 1)` | W m⁻² |
| `datetime` | UTC timestamps, length `N` | datetime |
| `lat`, `lon` | site coordinates | degrees |
| `averaging_period` | temporal averaging period | minutes |
| `pressure` | surface pressure (DISC only) | hPa or mb |

> Shaped inputs are normalised internally: scalars are broadcast to length `N`,
> 1-D arrays are treated as column vectors, and mismatched lengths raise
> `ValueError`.

## Model catalogue

| Model | Signature | Notes |
|---|---|---|
| `boland` | `(ghi, datetime, lat, lon, averaging_period=15)` | logistic kd model |
| `disc` | `(ghi, datetime, lat, lon, pressure)` | requires pressure; estimates DNI from clearness index |
| `engerer2` | `(ghi, datetime, lat, lon, averaging_period)` | averaging period required |
| `erbs` | `(ghi, datetime, lat, lon)` | classic Erbs decomposition |
| `louche` | `(ghi, datetime, lat, lon)` | |
| `orgill_hollands` | `(ghi, datetime, lat, lon)` | |
| `yang4` | `(ghi, datetime, lat, lon, averaging_period)` | |

Supported `averaging_period` values are model-dependent (typically 1, 5, 10,
15, 30 minutes); invalid values raise an error at call time.

## Worked example

```python
import numpy as np
from irradpy2.tool import read_csv, filter_sep_data, cal_metrics, plot_series
from irradpy2.separation import engerer2

data = read_csv("station_2025.csv")

kd, dhi_pred, dni_pred = engerer2(
    ghi=data["ghi"].to_numpy(),
    datetime=data["UTC"].to_numpy(),
    lat=51.968,
    lon=4.928,
    averaging_period=1,
)

data = data.assign(kd=kd, dhi_pred=dhi_pred, dni_pred=dni_pred)
```

## Evaluating separation results

Compare against observed components with the metrics in `irradpy2.tool`.
`filter_sep_data` removes physically ambiguous records first
(low sun, low GHI, negative values):

```python
filtered = filter_sep_data(
    data,
    latitude=51.968,
    longitude=4.928,
    time_col="UTC",
    zen_col="zen",
    ghi_col="ghi",
    dni_col="dni",
    dhi_col="dhi",
    pred_cols=["dhi_pred", "dni_pred"],
    max_zen=85,
    min_ghi=50,
    require_positive=True,
)

print(cal_metrics(filtered, obs_col="dhi", pred_col="dhi_pred"))
```

Visual comparison:

```python
from irradpy2.tool import plot_series

plot_series(
    filtered,
    obs_col="dhi",
    pred_col="dhi_pred",
    obs_label="Observed DHI",
    pred_label="Estimated DHI",
    title="Engerer2 separation",
    ylabel=r"DHI [W m$^{-2}$]",
)
```

## Practical guidance

- Separation models are empirical; error characteristics differ by climate.
  When observed DHI/DNI are available, evaluate several models on your site
  before choosing one.
- `kd` values outside `[0, 1]` can occur at the edges of the day; use
  `filter_sep_data` before computing summary statistics.
- If your data resolution differs from the model's expected averaging period,
  resample first with `irradpy2.tool.resample_series`.
