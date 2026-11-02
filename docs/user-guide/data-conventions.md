# Data conventions

This page is the canonical reference for variable names, units, and return
ordering across `irradpy2`. Follow it when mixing outputs from different
model families.

## Variable names and units

| Variable | Meaning | Typical unit |
|---|---|---|
| `UTC` | UTC timestamp | datetime |
| `ghi` | Global horizontal irradiance | W m⁻² |
| `dni` | Direct normal irradiance | W m⁻² |
| `dhi` | Diffuse horizontal irradiance | W m⁻² |
| `zen` | Solar zenith angle (`cal_zen`) | **degree** |
| `sza` | Solar zenith angle (clear-sky models) | **radian** |
| `press` / `pressure` | Atmospheric pressure | hPa or mb |
| `wv` | Precipitable water | model-dependent (cm or kg m⁻²) |
| `aod700` | Aerosol optical depth at 700 nm | dimensionless |
| `aod550` | Aerosol optical depth at 550 nm | dimensionless |
| `broadbandaod` | Broadband aerosol optical depth | dimensionless |
| `albedo` | Surface albedo | dimensionless |
| `ang_alpha` / `ang_beta` | Ångström turbidity coefficients | dimensionless |
| `ozone` / `ozone_input` | Total column ozone | cm-atm or Dobson |
| `NO2` | Nitrogen dioxide | dimensionless |
| `exth` | Extraterrestrial horizontal irradiance | W m⁻² |
| `kd` | Diffuse fraction (`dhi / ghi`) | dimensionless |
| `doy` / `year` | Day of year / calendar year | integer |

## Return ordering

### Clear-sky models (`irradpy2.clear_sky`)

All 62 models return a 3-element list in **GHI, DNI, DHI** order:

```python
ghi_clear, dni_clear, dhi_clear = model(sza, datetime, ...)
# result[0] = GHI  [W m-2]
# result[1] = DNI  [W m-2]
# result[2] = DHI  [W m-2]
```

### Separation models (`irradpy2.separation`)

```python
kd, dhi, dni = model(ghi, datetime, lat, lon, ...)
```

### CSD methods (`irradpy2.csd`)

```python
csd = method(ghi, ghics, ...)
# 1 = clouds suspected, 0 = clear
```

### Forecasting (`irradpy2.forecast`)

```python
ori_data, pre_data = forecast_xxx(datetime, ghi, ...)
# ori_data columns: Timestamp, Global
# pre_data columns: Timestamp, pre_Global
```

## Zenith angle: degrees vs radians

This is the most common source of silent error:

| Function / family | Produces / expects | Unit |
|---|---|---|
| `tool.cal_zen` | produces `zen` | degrees |
| `tool.cal_exth` | expects `zen` | degrees |
| `tool.filter_sep_data` | `max_zen` threshold | degrees |
| **`clear_sky.*`** | **expects `sza`** | **radians** |
| `csd.*` (`zen` argument) | expects `zen` | degrees |
| `tool.prepare_clearsky` | produces `sza` | radians |

Convert with `np.radians(zen)` or `np.degrees(sza)` as appropriate.

## MBE sign convention

```text
MBE = mean(predicted - observed)
```

- positive MBE → **overestimation**
- negative MBE → **underestimation**

Some literature defines MBE with the opposite sign. State the convention when
reporting results.

## Time-column names

Source datasets use inconsistent time-column names. `irradpy2.tool` functions
that auto-detect a time column accept: `UTC`, `UTC_time`, `UTC_Time`,
`Timestamp`, `datetime`, `date`, `time`. When auto-detection is ambiguous,
pass `time_col` explicitly.

## Ground-data variable mapping

Downloaders map source-specific names to standard names where possible, e.g.
MERRA-2 point mode:

| MERRA-2 native | Standard name | Unit after mapping |
|---|---|---|
| `SWGDN` | `ghi` | W m⁻² |
| `SWGDNCLR` | `ghi_clear` | W m⁻² |
| `TQV` | `water_vapour` | kg m⁻² |
| `TO3` | `ozone` | Dobson |
| `PS` | `pressure` | Pa |
| `TOTANGSTR` | `angstrom_exponent` | 1 |
| `ALBEDO` | `albedo` | 1 |
| `CLDTOT` | `cloud_fraction` | 1 |
| `TAUTOT` | `cloud_optical_thickness` | 1 |
| `TOTSCATAU` / `TOTEXTTAU` | `aod_scattering` / `aod550` | 1 |

ERA5 and CAMS downloaders perform analogous mapping (`map_variables=True`).
Disable it with `map_variables=False` to keep native names.

## Missing values and quality control

- Prefer `filter_sep_data` before separation evaluation (drops `zen > 85°`,
  `ghi < 50 W m⁻²`, and negatives when `require_positive=True`).
- Metric helpers drop `NaN` pairs automatically (`dropna` on
  `[obs_col, pred_col]`).
- Negative irradiance values are clipped to 0 by the forecasting pipeline and
  by `resample_series(lower_limit=0)`.
