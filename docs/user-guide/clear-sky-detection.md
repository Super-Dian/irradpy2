# Clear-sky detection (CSD)

The `irradpy2.csd` package provides **23 clear-sky detection methods** that
label each timestamp as clear or cloudy, typically by comparing observed
irradiance against a clear-sky reference.

```python
from irradpy2.csd import Ellis2018CSD, Reno2016CSD, Gueymard2013CSD
```

> **Parameters differ per method.** The conventions on this page (column-vector
> inputs, the 0/1 flag, shared calling shape) apply to all 23 methods, but each
> method's arguments follow its own source literature and are **not**
> interchangeable. Per-function parameter semantics are documented one entry
> at a time in the [CSD API reference](../reference/csd.md).

## Calling convention

Each method is a function that takes numpy column-vector inputs and returns a
detection flag array:

```python
csd = Ellis2018CSD(ghi, ghics)
```

### Return value

| Value | Meaning |
|---|---|
| `1` | clouds suspected |
| `0` | clear |

### Notes

- Inputs are column vectors (shape `(N, 1)`; 1-D arrays are accepted and
  reshaped). Most implementations are direct structural translations of the
  original MATLAB code and preserve its validation order.
- Most methods assume **1-minute resolution** data.
- An optional `plot_figure` argument, when provided, renders a diagnostic
  figure for that method.
- The package wraps every CSD function to suppress `RuntimeWarning` noise from
  numerical edge cases (e.g. zenith angles near 90°) without altering results.

## Method catalogue and required inputs

| Method | Signature | Needs clear-sky reference |
|---|---|---|
| `AliaMartinez2016CSD` | `(ghi, ghics, zen, datetime)` | GHI |
| `Batlles1999CSD` | `(ghi, dhi, exth, zen)` | — |
| `BrightSun2020CSDc` | `(zen, ghi, ghics, dhi, difcs, datetime)` | GHI, DHI |
| `BrightSun2020CSDs` | `(zen, ghi, ghics, dhi, difcs, datetime)` | GHI, DHI |
| `Ellis2018CSD` | `(ghi, ghics)` | GHI |
| `Garcia2014CSD` | `(ghi, dhi, zen, aod550, datetime)` | — |
| `Gueymard2013CSD` | `(dni, dnics, zen)` | DNI |
| `Ineichen2006CSD` | `(exth, dni, zen)` | — |
| `Ineichen2009CSD` | `(ghi, exth, zen)` | — |
| `Ineichen2016CSD` | `(ghi, dni, dhi, zen, exth, aod550)` | — |
| `Inman2015CSD` | `(ghi, ghics, dni, dnics)` | GHI, DNI |
| `Larraneta2017CSD` | `(dni, dnics, zen, datetime)` | DNI |
| `Lefevre2013CSD` | `(ghi, dhi, exth, zen)` | — |
| `LongAckerman2000CSD` | `(ghi, dhi, zen, datetime)` | — |
| `Perez1990CSD` | `(dhi, dni, zen)` | — |
| `Polo2009CSD` | `(ghi, ghics, zen, datetime)` | GHI |
| `Quesadaruiz2015CSD` | `(ghi, ghics)` | GHI |
| `Reno2016CSD` | `(ghi, ghics)` | GHI |
| `RuizArias2018CSD` | `(ghi, dhi, zen, datetime)` | — |
| `Shen2018CSD` | `(dni, dnics, aod550)` | DNI |
| `Xie2013CSD` | `(ghi, ghics, dni, dnics, zen, exth)` | GHI, DNI |
| `Zhang2018CSD` | `(ghi, ghics)` | GHI |
| `Zhao2018CSD` | `(dni, dnics)` | DNI |

Input abbreviations:

| Name | Meaning | Unit |
|---|---|---|
| `ghi` / `ghics` | observed / clear-sky global horizontal irradiance | W m⁻² |
| `dni` / `dnics` | observed / clear-sky direct normal irradiance | W m⁻² |
| `dhi` / `difcs` | observed / clear-sky diffuse horizontal irradiance | W m⁻² |
| `exth` | extraterrestrial horizontal irradiance | W m⁻² |
| `zen` | solar zenith angle | degrees |
| `aod550` | aerosol optical depth at 550 nm | unitless |
| `datetime` | UTC timestamps | datetime |

`exth` can be computed with `irradpy2.tool.cal_exth` (after `cal_zen`).

## Worked example

```python
import numpy as np
from irradpy2.tool import read_csv, cal_zen, cal_exth, prepare_clearsky
from irradpy2.clear_sky import ineichen_2008
from irradpy2.csd import Reno2016CSD
from irradpy2.tool import plot_csd

data = read_csv("station_2025.csv")
data = cal_zen(data, latitude=51.968, longitude=4.928, time_col="UTC")
data = cal_exth(data, time_col="UTC", zen_col="zen")

sza = np.radians(data["zen"].to_numpy())
ghi_clear, dni_clear, dhi_clear = ineichen_2008(
    sza,
    data["UTC"].to_numpy(),
    data["press"].to_numpy(),
    data["aod700"].to_numpy(),
    data["wv"].to_numpy(),
)

csd = Reno2016CSD(data["ghi"].to_numpy(), ghi_clear)

data = data.assign(ghi_clear=ghi_clear, csd=csd)
plot_csd(data, ghi_col="ghi", clear_col="ghi_clear", csd_col="csd")
```

## Visualisation

```python
from irradpy2.tool import plot_csd

plot_csd(data, ghi_col="ghi", clear_col="ghi_clear", csd_col="csd", max_points=3000)
```

`plot_csd` overlays the observed series, the clear-sky reference, and the
detection flag.

## References

Method references follow the original publications (Ellis & Reno 2018;
Reno & Hansen 2016; Gueymard 2013; Ineichen 2006/2009/2016; Perez 1990;
Bright & Sun 2020; …). Each function's docstring carries its citation details.
