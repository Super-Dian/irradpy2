# Clear-sky irradiance models

The `irradpy2.clear_sky` package provides **62 clear-sky irradiance models**.
Each model computes clear-sky global horizontal (GHI), direct normal (DNI), and
diffuse horizontal (DHI) irradiance from solar geometry and atmospheric inputs.

```python
from irradpy2.clear_sky import tj, rest2v5, ineichen_2008, bird
```

> **Parameters differ per model.** The conventions on this page (return order,
> units, angle units) are shared by all 62 models, but each model's argument
> list and atmospheric inputs follow its own source literature and are **not**
> interchangeable. Per-function parameter semantics are documented one entry
> at a time in the [clear-sky API reference](../reference/clear-sky.md).

## Calling convention

Every model is a plain function with positional array-like inputs
(`numpy.ndarray`). All inputs must share the same length `N`.

### Return ordering (important)

All models return a 3-element list in **GHI, DNI, DHI** order:

```python
ghi_clear, dni_clear, dhi_clear = model(sza, datetime, ...)
# equivalently: result[0] = GHI, result[1] = DNI, result[2] = DHI
```

Units: W m⁻².

### Solar zenith angle

The `sza` argument is the solar zenith angle in **radians**. The
`prepare_clearsky` helper in `irradpy2.tool` produces `sza` in radians; if you
use `cal_zen` instead, convert with `np.radians(zen)`.

### Time input

The `datetime` argument is array-like of datetimes (e.g. a pandas Series or
`numpy` datetime array). Internally, day-of-year and year are derived from it.
A few models take `Dayth` / `Year` directly instead of `datetime` (`hlj`).

## Input families

Models differ only in which atmospheric variables they require. Group them by
required inputs to pick one that matches your data:

### Minimal — zenith angle only

```text
abcgs, adnot, biga, dpp, hs, paltridge, schulze
```

```python
ghi_clear, dni_clear, dhi_clear = abcgs(sza)
```

### Zenith angle + time

```text
ashrae_1985, ashrae_2005, elmghouchi, sharma, tj, yangwalsh
```

```python
ghi_clear, dni_clear, dhi_clear = tj(sza, datetime)
```

### Time + pressure (and often water vapour / albedo)

```text
atwaterball, atwaterball2, badescu, bashahu, bclsm, campbell, capderou, cem,
cls, cpcr2, csr, dai, dogniaux, esra, furich, heliosat, heliosat2, hlj, hoyt,
ideriah, ineichen_2008, ineichen_2018, ineichenperez, iqbal_a, iqbal_b, iqbal_c,
janjai, josefsson, kasm, kasten, king, kumar, mac1, mac2, metstat, mmac,
modified_iqbalc, mrmv5, mrmv6, paulescu, perrin, pr, psim, rest2v5, rsc, watt,
yang
```

Common optional arguments across this family:

| Parameter | Meaning | Unit |
|---|---|---|
| `press` / `pressure` | surface pressure | hPa or mb |
| `wv` | precipitable water | cm or kg m⁻² (model-dependent) |
| `albedo` | surface albedo | unitless |
| `ang_alpha` / `ang_beta` | Ångström turbidity coefficients | unitless |
| `ozone` | total column ozone | cm-atm or Dobson |
| `NO2` | nitrogen dioxide | unitless (`rest2v5`, `calinoiu`) |
| `aod700` / `aod550` / `broadbandaod` | aerosol optical depth | unitless |
| `altitude` | site altitude | m |
| `TL2_series` | Linke turbidity series | unitless |

## Full model catalogue (signatures)

| Model | Signature |
|---|---|
| `abcgs` | `(sza)` |
| `adnot` | `(zsa)` |
| `ashrae_1985` | `(sza, datetime)` |
| `ashrae_2005` | `(sza, datetime)` |
| `atwaterball` | `(sza, datetime, press, wv)` |
| `atwaterball2` | `(sza, datetime, press, wv)` |
| `badescu` | `(sza, datetime, press, wv, ozone)` |
| `bashahu` | `(sza, datetime, press, ang_alpha, ang_beta, wv)` |
| `bclsm` | `(sza, datetime, wv)` |
| `biga` | `(sza)` |
| `bird` | `(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, wv)` |
| `calinoiu` | `(sza, datetime, ang_beta, ozone, NO2, wv)` |
| `campbell` | `(sza, datetime, press)` |
| `capderou` | `(sza, datetime, latitude, altitude, press)` |
| `cem` | `(sza, datetime, press, albedo, broadbandaod, wv)` |
| `cls` | `(sza, datetime, press, wv, albedo)` |
| `cpcr2` | `(sza, datetime, pressure, wv, ozone, ang_beta, ang_alpha, albedo)` |
| `csr` | `(sza, datetime, pressure, broadbandaod, wv, ozone, albedo)` |
| `dai` | `(sza, datetime, pressure, wv, ang_beta, ang_alpha)` |
| `dogniaux` | `(sza, datetime, pressure, decli, TL2_series)` |
| `dpp` | `(sza)` |
| `elmghouchi` | `(sza, datetime)` |
| `esra` | `(sza, datetime, altitude, TL2_series)` |
| `furich` | `(sza, datetime, altitude)` |
| `heliosat` | `(sza, datetime, press, TL2_series)` |
| `heliosat2` | `(sza, datetime, altitude, TL2_series)` |
| `hlj` | `(sza, Dayth, Year, altitude)` |
| `hoyt` | `(sza, datetime, press, albedo, ang_beta, ozone, wv)` |
| `hs` | `(sza)` |
| `ideriah` | `(sza, datetime, pressure, wv, ang_alpha, ang_beta)` |
| `ineichen_2008` | `(sza, datetime, press, aod700, wv)` |
| `ineichen_2018` | `(sza, datetime, pressure, wv, aod550)` |
| `ineichenperez` | `(zsa, datetime, altitude, TL2_series)` |
| `iqbal_a` | `(sza, datetime, pressure, wv, ozone, ang_alpha, ang_beta, albedo)` |
| `iqbal_b` | `(sza, datetime, pressure, wv, ozone, ang_beta, albedo)` |
| `iqbal_c` | `(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, wv)` |
| `janjai` | `(sza, datetime, altitude, ang_alpha, ang_beta, ozone, wv)` |
| `josefsson` | `(sza, datetime, press, wv, albedo)` |
| `kasm` | `(sza, datetime, press, wv)` |
| `kasten` | `(sza, datetime, altitude, TL2_series)` |
| `king` | `(sza, datetime, press, albedo, wv, ang_beta)` |
| `kumar` | `(sza, datetime, press)` |
| `mac1` | `(sza, datetime, pressure, wv, albedo)` |
| `mac2` | `(sza, datetime, press, albedo, ang_alpha, ang_beta, wv)` |
| `metstat` | `(sza, datetime, press, albedo, broadbandaod, ozone, wv)` |
| `mmac` | `(sza, datetime, press, albedo, broadbandaod, wv)` |
| `modified_iqbalc` | `(sza, datetime, altitude, albedo, ang_alpha, ang_beta, ozone, wv)` |
| `mrmv5` | `(sza, datetime, press, albedo, ang_beta, ozone, wv)` |
| `mrmv6` | `(sza, datetime, press, albedo, aod550, ozone, wv)` |
| `paltridge` | `(sza)` |
| `paulescu` | `(sza, datetime, altitude, press, ang_beta, ozone, wv)` |
| `perrin` | `(sza, datetime, press, ang_beta, wv, ozone)` |
| `pr` | `(sza, datetime, press, albedo, ang_beta, ozone, wv)` |
| `psim` | `(sza, datetime, press, albedo, ang_beta, ozone, wv)` |
| `rest2v5` | `(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, NO2, wv)` |
| `rsc` | `(sza, datetime, press, albedo, ang_beta, ozone, wv)` |
| `schulze` | `(sza)` |
| `sharma` | `(sza, datetime)` |
| `tj` | `(sza, datetime)` |
| `watt` | `(sza, datetime, pressure, wv, ozone, ang_beta, ang_alpha, albedo)` |
| `yang` | `(sza, datetime, press, ang_beta, ozone, wv)` |
| `yangwalsh` | `(zsa, datetime)` |

> **Note.** `mrmv6` additionally reads `AEROSOL-TABLE.txt` from the current
> working directory — see [installation notes](../installation.md#known-packaging-notes).

## Worked examples

### TJ (1957) — minimal inputs

```python
from irradpy2.clear_sky import tj

ghi_clear, dni_clear, dhi_clear = tj(sza, datetime)
```

### Simplified Solis (Ineichen 2008)

```python
from irradpy2.clear_sky import ineichen_2008

ghi_clear, dni_clear, dhi_clear = ineichen_2008(sza, datetime, press, aod700, wv)
```

### REST2 v5 — fully parameterised

```python
from irradpy2.clear_sky import rest2v5

ghi_clear, dni_clear, dhi_clear = rest2v5(
    sza, datetime, press, albedo,
    ang_alpha, ang_beta, ozone, NO2, wv,
)
```

### Preparing inputs from ground + MERRA-2

```python
from irradpy2.tool import read_csv, prepare_clearsky

ground = read_csv("ground_data.csv")
merra2 = read_csv("merra2_data.csv")

data = prepare_clearsky(
    ground, merra2,
    latitude=51.968, longitude=4.928,
    begin="2025-01-01", end="2025-01-31",
)
# data["sza"] is in radians — ready for any clear_sky model
```

## References

Each model function's docstring carries its primary literature reference
(e.g. Threlkeld & Jordan 1957 for `tj`; Gueymard 2008 for `rest2v5`;
Ineichen & Perez 2002 for `ineichenperez`). Consult the function docstring for
the canonical citation:

```python
help(ineichen_2008)
```
