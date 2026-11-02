# `irradpy2.getgridded` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (8): `download_cams`, `download_cams_area`, `download_cams_point`, `download_era5`, `download_era5_area`, `download_era5_point`, `download_merra2`, `list_merra2_collections`

---

### `download_cams(mode='point', quiet=True, api_url=None, api_key=None, **kwargs)`

Unified CAMS entry point; dispatches to the point or area downloader by `mode`.

**Parameters**

- `mode` (*str, default "point"*) — `"point"` or `"area"` (case-insensitive; `ValueError` otherwise).
- `quiet` (*bool, default True*) — Suppress `cdsapi` client progress output.
- `api_url` (*str, optional*) — ADS API endpoint override.
- `api_key` (*str, optional*) — ADS API key override.
- `**kwargs` (*keyword arguments*) — `mode="point"` requires `lat`, `lon`, `start`, `end` (optional `save_path`, `file_format`, `sky_type`, `time_step`, `time_ref`, `altitude`, `parse_csv`, `map_variables`); `mode="area"` requires `start`, `end` (optional `lat_1`, `lon_1`, `lat_2`, `lon_2`, `variables`, `sky_type`, `output_dir`, `file_prefix`). Missing required keys raise `ValueError`.

**Returns**

CamsPointResult or CamsAreaResult — whichever dataclass the selected downloader returns.

**Notes**

Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.

### `download_cams_area(lat_1=None, lon_1=None, lat_2=None, lon_2=None, start=None, end=None, variables=None, sky_type='observed_cloud', output_dir='CAMS_data', quiet=True, file_prefix=None, api_url=None, api_key=None)`

Download CAMS gridded satellite radiation for a period as NetCDF files (one per variable and month).

**Parameters**

- `lat_1` (*float, optional*) — Corner latitude [degrees]. When any of the four corners is `None` the whole-domain request is used (no area restriction); when all four are given they are sorted internally to [north, west, south, east].
- `lon_1` (*float, optional*) — Corner longitude [degrees]; see `lat_1`.
- `lat_2` (*float, optional*) — Corner latitude [degrees]; see `lat_1`.
- `lon_2` (*float, optional*) — Corner longitude [degrees]; see `lat_1`.
- `start` (*Any*) — Start of the period; required (`ValueError` when `None`).
- `end` (*Any*) — End of the period; required; must not precede `start` (`ValueError`).
- `variables` (*sequence of str, optional*) — Local aliases (`ghi`, `dni`, `dhi`, `bhi`) resolved via `AREA_VARIABLE_MAP`; defaults to `["ghi", "dni", "dhi"]`.
- `sky_type` (*str, default "observed_cloud"*) — `"clear"` or `"observed_cloud"` (`ValueError` otherwise).
- `output_dir` (*str or pathlib.Path, default "CAMS_data"*) — Directory for NetCDF parts and metadata (created if missing).
- `file_prefix` (*str, optional*) — Output file prefix; defaults to `cams_area_<corners>` when an area is given, else `cams_area`.
- `quiet` (*bool, default True*) — Suppress `cdsapi` client progress output.
- `api_url` (*str, optional*) — ADS API endpoint override.
- `api_key` (*str, optional*) — ADS API key override.

**Returns**

CamsAreaResult dataclass — `netcdf_paths` (list of pathlib.Path), `metadata` (dict), `metadata_path`. A `<prefix>_metadata.json` sidecar is written.

**Notes**

One retrieve is issued per variable and calendar month. Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.

### `download_cams_point(lat, lon, start, end, save_path=None, file_format='csv', sky_type='observed_cloud', time_step='1min', time_ref='UT', altitude=None, quiet=True, parse_csv=True, map_variables=True, api_url=None, api_key=None)`

Download CAMS satellite-derived radiation for one point from the ADS API.

**Parameters**

- `lat` (*float*) — Latitude [degrees].
- `lon` (*float*) — Longitude [degrees].
- `start` (*Any*) — Start of the period; anything `pandas.Timestamp` accepts, converted to UTC (tz-aware values are converted, naive values are taken as-is).
- `end` (*Any*) — End of the period (same parsing as `start`); must not precede `start` (`ValueError`).
- `save_path` (*str or pathlib.Path, optional*) — Output path; defaults to `cams_<lat>_<lon>_<start>_<end>.csv`. The suffix follows `file_format` (`.csv` or `.nc`).
- `file_format` (*str, default "csv"*) — `"csv"` or `"netcdf"` (`ValueError` otherwise).
- `sky_type` (*str, default "observed_cloud"*) — `"clear"` or `"observed_cloud"` (`ValueError` otherwise).
- `time_step` (*str, default "1min"*) — One of `"1min"`, `"15min"`, `"1h"`, `"1d"`, `"1M"` (`ValueError` otherwise).
- `time_ref` (*str, default "UT"*) — `"UT"` (universal time) or `"TST"` (true solar time) (`ValueError` otherwise).
- `altitude` (*float, optional*) — Site altitude [m]; when `None` the service default (-999) is sent.
- `quiet` (*bool, default True*) — Suppress `cdsapi` client progress output.
- `parse_csv` (*bool, default True*) — CSV only: parse the download, standardize it and rewrite it with a leading `UTC_time` column. Ignored for NetCDF (`data` stays `None`).
- `map_variables` (*bool, default True*) — Rename parsed columns to standard names (`ghi`, `dni`, `dhi`, `ghi_clear`, ...) via `VARIABLE_MAP`.
- `api_url` (*str, optional*) — ADS API endpoint override.
- `api_key` (*str, optional*) — ADS API key override.

**Returns**

CamsPointResult dataclass — `data` (pandas.DataFrame when parsed, else `None`), `metadata` (dict), `data_path`, `metadata_path` (pathlib.Path). A `<stem>_metadata.json` sidecar is written.

**Notes**

A parsed CSV is rewritten in place with columns ordered `UTC_time`, then GHI/DNI/DHI (observed and clear-sky), then the rest. Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.

### `download_era5(mode='point', start=None, end=None, variables=None, api_url=None, api_key=None, quiet=True, **kwargs)`

Unified ERA5 entry point; dispatches to the point or area downloader by `mode`.

**Parameters**

- `mode` (*str, default "point"*) — `"point"` or `"area"` (case-insensitive; `ValueError` otherwise).
- `start` (*Any*) — Start of the period; required (`ValueError` when `None`).
- `end` (*Any*) — End of the period; required.
- `variables` (*sequence of str, optional*) — Forwarded to the chosen downloader.
- `api_url` (*str, optional*) — CDS API endpoint override.
- `api_key` (*str, optional*) — CDS API key override.
- `quiet` (*bool, default True*) — Suppress `cdsapi` client progress output.
- `**kwargs` (*keyword arguments*) — `mode="point"` requires `lat`, `lon` (optional `save_path`, `map_variables`, `convert_units`); `mode="area"` requires `lat_1`, `lon_1`, `lat_2`, `lon_2` (optional `output_dir`, `file_prefix`). Missing required keys raise `ValueError`.

**Returns**

PointResult or AreaResult — whichever dataclass the selected downloader returns.

**Notes**

Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.

### `download_era5_area(lat_1, lon_1, lat_2, lon_2, start, end, variables=None, output_dir='ERA5_data', file_prefix=None, api_url=None, api_key=None, quiet=True)`

Download ERA5 reanalysis for a bounding box from the CDS API as monthly GRIB files.

**Parameters**

- `lat_1` (*float*) — First corner latitude [degrees].
- `lon_1` (*float*) — First corner longitude [degrees].
- `lat_2` (*float*) — Opposite corner latitude [degrees]; must differ from `lat_1` (`ValueError`), order versus `lat_1` is irrelevant (sorted internally).
- `lon_2` (*float*) — Opposite corner longitude [degrees]; must differ from `lon_1` (`ValueError`).
- `start` (*Any*) — Start of the period; anything `pandas.Timestamp` accepts, converted to UTC (tz-aware values are converted, naive values are taken as-is).
- `end` (*Any*) — End of the period (same parsing as `start`); must not precede `start` (`ValueError`).
- `variables` (*sequence of str, optional*) — CDS variable names or local aliases; defaults to `["ssrd", "strd", "t2m", "sp", "tcc"]`.
- `output_dir` (*str or pathlib.Path, default "ERA5_data"*) — Directory for GRIB parts and metadata (created if missing).
- `file_prefix` (*str, optional*) — Output file prefix; defaults to `era5_area_<lat_1>_<lon_1>_<lat_2>_<lon_2>`.
- `api_url` (*str, optional*) — CDS API endpoint override.
- `api_key` (*str, optional*) — CDS API key override.
- `quiet` (*bool, default True*) — Suppress `cdsapi` client progress output.

**Returns**

AreaResult dataclass — `grib_paths` (list of pathlib.Path, one GRIB part per month), `metadata` (dict), `metadata_path`. A `<prefix>_metadata.json` sidecar is written.

**Notes**

The range is split into month chunks, each retrieved as its own GRIB file. Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.

### `download_era5_point(lat, lon, start, end, variables=None, save_path=None, api_url=None, api_key=None, quiet=True, map_variables=True, convert_units=True)`

Download ERA5 reanalysis for one point from the CDS API and write it to CSV.

**Parameters**

- `lat` (*float*) — Latitude [degrees].
- `lon` (*float*) — Longitude [degrees].
- `start` (*Any*) — Start of the period; anything `pandas.Timestamp` accepts, converted to UTC (tz-aware values are converted, naive values are taken as-is).
- `end` (*Any*) — End of the period (same parsing as `start`); must not precede `start` (`ValueError`).
- `variables` (*sequence of str, optional*) — CDS variable names or local aliases (`ghi`, `temp_air`, ... resolved via `INPUT_VARIABLE_MAP`); defaults to `["ssrd", "strd", "t2m", "sp", "tcc", "tp"]`.
- `save_path` (*str or pathlib.Path, optional*) — Output CSV path; defaults to `era5_<lat>_<lon>_<start>_<end>.csv`. The `.csv` suffix is forced.
- `api_url` (*str, optional*) — CDS API endpoint override.
- `api_key` (*str, optional*) — CDS API key override.
- `quiet` (*bool, default True*) — Suppress `cdsapi` client progress output.
- `map_variables` (*bool, default True*) — Rename columns to standard names (`ghi`, `longwave_down`, `temp_air`, `pressure`, `cloud_fraction`, `precipitation`, ...).
- `convert_units` (*bool, default True*) — Apply unit conversions (e.g. K -> degC, m -> mm).

**Returns**

PointResult dataclass — `data` (pandas.DataFrame with a `UTC_time` column plus the requested variables), `metadata` (dict: request, columns, coverage), `csv_path`, `metadata_path` (pathlib.Path). The CSV and a `<stem>_metadata.json` sidecar are written to disk.

**Notes**

Issues one CDS retrieve covering the whole range, then parses and rewrites the CSV. Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.

### `download_merra2(collection_names=None, initial_year=None, final_year=None, initial_month=1, final_month=12, initial_day=1, final_day=None, lat_1=-90, lon_1=-180, lat_2=90, lon_2=180, lat=None, lon=None, mode='area', start=None, end=None, variables=None, output_dir='MERRA2_data', save_path=None, auth=None, username=None, password=None, merge_timelapse='none', thread_num=1, map_variables=True, timeout=(30, 300))`

Unified MERRA2 downloader.  mode="area"     Download NetCDF subset files by four area parameters: lat_1, lon_1, lat_2, lon_2.  mode="point"     Download single-point CSV time series by lat and lon.

**Parameters**

- `collection_names` (*str or list of str, optional*) — MERRA-2 collection(s) from the built-in registry (`rad`, `slv`, `aer`, `asm`, `dewpt`, `evspsbl`, `hur`, `huss`, `phis`, `pr`, `prc`, `prbc`, `prmax`, `prsn`, `rls`, `sftgif`, `sftkf`, `sftlf`, `sftof`, `shg`, `sic`, `tas`, `tasmax`, `tasmin`, `uas`, `vas`). Point mode defaults to `["rad", "slv", "aer"]`; area mode requires it explicitly (`ValueError`) to avoid huge default downloads.
- `initial_year` (*int, optional*) — First year. Required in area mode (`ValueError`); in point mode used only when `start`/`end` are not given.
- `final_year` (*int, optional*) — Last year; same rules as `initial_year`.
- `initial_month` (*int, default 1*) — First month of the year/month/day range.
- `final_month` (*int, default 12*) — Last month of the year/month/day range.
- `initial_day` (*int, default 1*) — First day of the year/month/day range.
- `final_day` (*int, optional*) — Last day. Required in area mode and whenever the year/month/day range form is used (`ValueError`).
- `lat_1` (*float, default -90*) — Area corner latitude [degrees] (area mode); must satisfy `lat_1 <= lat_2` (`RuntimeError` otherwise).
- `lon_1` (*float, default -180*) — Area corner longitude [degrees] (area mode); must satisfy `lon_1 <= lon_2` (`RuntimeError` otherwise).
- `lat_2` (*float, default 90*) — Opposite area corner latitude [degrees] (area mode).
- `lon_2` (*float, default 180*) — Opposite area corner longitude [degrees] (area mode).
- `lat` (*float, optional*) — Point latitude [degrees]. Required in point mode (`ValueError`).
- `lon` (*float, optional*) — Point longitude [degrees]. Required in point mode (`ValueError`).
- `mode` (*str, default "area"*) — `"area"` (NetCDF subset by four corners) or `"point"` (single-point CSV time series); `ValueError` otherwise.
- `start` (*Any, optional*) — Point-mode period start (e.g. `"YYYY-MM-DD"`); `start` and `end` must be given together. Date-only values start at 00:00:00 UTC.
- `end` (*Any, optional*) — Point-mode period end; date-only values extend to 23:59:59.
- `variables` (*sequence or dict, optional*) — Variable subset. A string or list is allowed for a single collection; with multiple collections pass a dict `{collection: ["VAR", ...]}` (`ValueError` otherwise). Names must exist in the collection's registry (`ValueError`).
- `output_dir` (*str or pathlib.Path, default "MERRA2_data"*) — Area-mode output directory (created if missing); downloads land in a dated subdirectory.
- `save_path` (*str or pathlib.Path, optional*) — Point mode: also write the resulting table to this CSV. Ignored in area mode.
- `auth` (*dict, optional*) — Earthdata credentials `{"uid": ..., "password": ...}`; alternative to `username`/`password`. Required in both modes (`ValueError` when absent).
- `username` (*str, optional*) — Earthdata Login username (combined with `password` into `auth`).
- `password` (*str, optional*) — Earthdata Login password.
- `merge_timelapse` (*str, default "none"*) — Area mode: `"none"`, `"daily"`, `"monthly"` or `"yearly"` — merge downloaded files afterwards (`RuntimeError` otherwise).
- `thread_num` (*int, default 1*) — Area mode: parallel download threads.
- `map_variables` (*bool, default True*) — Rename columns to standard names (`VARIABLE_MAP`).
- `timeout` (*tuple of float, default (30, 300)*) — Point mode: `(connect, read)` timeouts in seconds for OPeNDAP requests.

**Returns**

Depends on `mode`: area mode returns `str` — the directory holding the downloaded (and merged) NetCDF files; point mode returns `(pandas.DataFrame, dict)` — the time-series table (with `time` as the first column) and metadata including per-variable units.

**Notes**

Requires NASA Earthdata Login credentials. Area mode validates `collection_names`, `initial_year`, `final_year` and `final_day` up front (`ValueError`) and raises `RuntimeError` when the bounding box or `merge_timelapse` is invalid, or after three failed download retries.

### `list_merra2_collections()`

Return a copy of the built-in MERRA2 collection registry.

**Returns**

dict — a copy of the built-in registry `{collection_name: {...}}`; mutating it does not affect the module-level table.

