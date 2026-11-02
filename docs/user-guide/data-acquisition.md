# Data acquisition

`irradpy2` provides two downloader families:

- **`irradpy2.getirrad`** — ground-based observation networks (6 sources).
- **`irradpy2.getgridded`** — gridded / reanalysis meteorological data (3 sources).

All downloaders write CSV (or NetCDF for MERRA-2 area mode) to a path you
specify, and accept `start` / `end` as date strings (`"YYYY-MM-DD"`).

## Ground-based observations (`getirrad`)

```python
from irradpy2.getirrad import (
    download_bsrn,
    download_midc,
    download_sauran,
    download_solrad,
    download_srml,
    download_surfrad,
)
```

| Function | Source | Credentials | Notes |
|---|---|---|---|
| `download_bsrn` | [BSRN](https://bsrn.awi.de/) | `username`, `password` | Station code required (e.g. `"CAB"`) |
| `download_midc` | NREL MIDC | none | Station code (e.g. `"BMS"`) |
| `download_sauran` | SAURAN | none | Southern African network |
| `download_solrad` | NOAA SOLRAD | none | |
| `download_srml` | UO SRML | none | Oregon network |
| `download_surfrad` | NOAA SURFRAD | none | |

### Signatures

```python
download_bsrn(site, start, end, username, password, save_path)
download_midc(site, start, end, save_path)
download_sauran(site, start, end, save_path)
download_solrad(site, start, end, save_path)
download_srml(site, start, end, save_path)
download_surfrad(site, start, end, save_path)
```

### Example

```python
from irradpy2.getirrad import download_bsrn

download_bsrn(
    site="CAB",
    start="2025-01-01",
    end="2025-01-31",
    username="your_bsrn_username",
    password="your_bsrn_password",
    save_path="cab_2025.csv",
)
```

## Gridded / reanalysis data (`getgridded`)

```python
from irradpy2.getgridded import download_merra2, download_era5, download_cams
```

Each of `download_era5` and `download_cams` dispatches on `mode`:

- `mode="point"` — one location (`lat`, `lon`), one output file.
- `mode="area"` — bounding box (`lat_1, lon_1, lat_2, lon_2`), files per variable
  into `output_dir`.

### MERRA-2

```python
download_merra2(
    collection_names=["rad", "slv", "aer"],
    mode="point",              # "area" (default) or "point"
    lat=51.968,
    lon=4.928,
    start="2025-01-01",
    end="2025-01-31",
    save_path="merra2_point.csv",   # point mode
    username="your_earthdata_username",
    password="your_earthdata_password",
)
```

Notes:

- Requires a [NASA Earthdata](https://urs.earthdata.nasa.gov/) account. You may
  pass `auth={"username": ..., "password": ...}` instead of the keyword pair.
- Collections: `rad` (radiation), `slv` (surface/atmosphere), `aer` (aerosols).
- Point mode maps gridded variables to standard names
  (`SWGDN → ghi`, `SWGDNCLR → ghi_clear`, `TQV → water_vapour`,
  `TO3 → ozone`, `PS → pressure`, `TOTANGSTR → angstrom_exponent`, …) and
  reports the matched grid cell in metadata.
- Area mode supports `merge_timelapse` (`"none"`, `"yearly"`, `"monthly"`),
  `thread_num`, and `variables` selection.

### ERA5

```python
result = download_era5(
    mode="point",
    start="2025-01-01",
    end="2025-01-31",
    variables=["t2m", "sp", "tp"],   # optional subset
    lat=51.968,
    lon=4.928,
    save_path="era5_point.csv",
    api_url="https://cds.climate.copernicus.eu/api/v2",
    api_key="your_cds_api_key",
)
```

- Requires [Copernicus CDS](https://cds.climate.copernicus.eu/) credentials.
- Returns a `PointResult` / `AreaResult` dataclass carrying the output path and
  metadata. `map_variables=True` applies standard-name mapping;
  `convert_units=True` converts to W m⁻² and common conventions.

### CAMS

```python
result = download_cams(
    mode="point",
    lat=51.968,
    lon=4.928,
    start="2025-01-01",
    end="2025-01-31",
    sky_type="observed_cloud",
    time_step="1min",
    save_path="cams_point.csv",
    api_url="https://ads.atmosphere.copernicus.eu/api/v2",
    api_key="your_ads_api_key",
)
```

- Requires [Copernicus ADS](https://atmosphere.copernicus.eu/) credentials.
- `sky_type` selects the CAMS product (`"observed_cloud"`, `"clear_sky"`).
- Returns `CamsPointResult` / `CamsAreaResult`.

## Time coverage and resolution

- Gridded sources are typically hourly; ground networks vary (1-minute common).
- Use `irradpy2.tool.resample_series` to place mixed sources on one grid before
  comparison, and `merge_series` to align them on the time index.

## See also

- [Data tools](data-tools.md) — reading, resampling, and merging downloaded CSVs.
- [Getting started](../getting-started.md) — full pipeline example.
