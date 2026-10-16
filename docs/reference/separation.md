# `irradpy2.separation` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (7): `boland`, `disc`, `engerer2`, `erbs`, `louche`, `orgill_hollands`, `yang4`

---

### `boland(ghi, datetime, lat, lon, averaging_period=15)`

Boland irradiance separation model.

**Parameters**

- `ghi` (*numpy.ndarray*) — Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].
- `averaging_period` (*int, default 15*) — Data averaging period in minutes. Supported values are 15 and 60.

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

**References**

Boland, J., Scott, L., and Luther, M. (2001).
Modelling the diffuse fraction of global solar radiation on a horizontal
surface. Environmetrics, 12(2), 103-116.

### `disc(ghi, datetime, lat, lon, pressure)`

DISC irradiance separation model.

**Parameters**

- `ghi` (*numpy.ndarray*) — Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].
- `pressure` (*numpy.ndarray*) — Atmospheric pressure [hPa], with shape (N,) or (N, 1).

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

**References**

Maxwell, E. L. (1987). A Quasi-Physical Model for Converting Hourly
Global Horizontal to Direct Normal Insolation. SERI/TR-215-3087.

### `engerer2(ghi, datetime, lat, lon, averaging_period)`

Engerer2 irradiance separation model.

**Parameters**

- `ghi` (*numpy.ndarray*) — Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].
- `averaging_period` (*int*) — Averaging period in minutes. Supported values are 1, 5, 10, 15, 30, 60, and 1440.

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

### `erbs(ghi, datetime, lat, lon)`

Erbs irradiance separation model.  The clearness index kt = GHI / extraterrestrial horizontal irradiance is clipped to [0, 1] and mapped to the diffuse fraction kd with the piecewise Erbs correlation (linear below kt = 0.22, quartic up to kt = 0.8, constant 0.165 above). DHI = GHI * kd; DNI is recovered as (GHI - DHI) / cos(zenith) only where the zenith angle is below 87 degrees, otherwise it is set to 0 and the missing fraction is assigned to DHI.

**Parameters**

- `ghi` (*numpy.ndarray*) — Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

### `louche(ghi, datetime, lat, lon)`

Louche irradiance separation model.

**Parameters**

- `ghi` (*numpy.ndarray*) — Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

**References**

Louche, A., Notton, G., Poggi, P., and Simonnot, G. (1991).
Correlations for direct normal and global horizontal irradiation on a
French Mediterranean site. Solar Energy, 46(4), 261-266.

### `orgill_hollands(ghi, datetime, lat, lon)`

Orgill-Hollands irradiance separation model.

**Parameters**

- `ghi` (*numpy.ndarray*) — Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

**References**

Orgill, J. F., and Hollands, K. G. T. (1977).
Correlation equation for hourly diffuse radiation on a horizontal
surface. Solar Energy, 19(4), 357-359.

### `yang4(ghi, datetime, lat, lon, averaging_period)`

Yang4 irradiance separation model.

**Parameters**

- `ghi` (*numpy.ndarray*) — One-minute global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
- `datetime` (*numpy.ndarray*) — UTC datetime values with length N.
- `lat` (*float*) — Site latitude [degrees].
- `lon` (*float*) — Site longitude [degrees].
- `averaging_period` (*int*) — Input averaging period in minutes. Yang4 only supports 1.

**Returns**

- `kd` (*numpy.ndarray*) — Diffuse fraction with shape (N, 1).
- `dhi` (*numpy.ndarray*) — Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
- `dni` (*numpy.ndarray*) — Direct normal irradiance [W/m^2] with shape (N, 1).

**References**

Yang, D. (2021). Temporal-resolution cascade model for separation of
1-min beam and diffuse irradiance. Journal of Renewable and Sustainable
Energy, 13(5), 056101.

