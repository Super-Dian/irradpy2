"""
CAMS downloader for irradpy2.

Main capabilities
-----------------
- Download CAMS point time-series data
- Download CAMS gridded monthly netCDF data
- Export parsed CSV without timezone suffix in the UTC_time column
- Export metadata as JSON

Requirements
------------
- cdsapi >= 0.7.7
- pandas

Notes
-----
- Point dataset:
    "cams-solar-radiation-timeseries"
- Area dataset:
    "cams-gridded-solar-radiation"
- The gridded product is monthly netCDF.
- CAMS gridded coverage is limited and may not be suitable for all regions.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cdsapi
import pandas as pd


ADS_API_URL = "https://ads.atmosphere.copernicus.eu/api"

POINT_DATASET = "cams-solar-radiation-timeseries"
AREA_DATASET = "cams-gridded-solar-radiation"
AREA_VERSION = "4.6"

DEFAULT_AREA_VARIABLES = ["ghi", "dni", "dhi"]

TIMESTEP_MAP: Dict[str, str] = {"1min": "1minute", "15min": "15minute", "1h": "1hour", "1d": "1day", "1M": "1month"}

TIME_REF_MAP: Dict[str, str] = {"UT": "universal_time", "TST": "true_solar_time"}

AREA_VARIABLE_MAP: Dict[str, str] = {
    "ghi": "global_horizontal_irradiation",
    "bhi": "direct_horizontal_irradiation",
    "dni": "direct_normal_irradiation",
    "dhi": "diffuse_horizontal_irradiation",
}

VARIABLE_MAP: Dict[str, str] = {
    "time": "UTC_time",
    "Observation period": "UTC_time",
    "GHI": "ghi",
    "BHI": "bhi",
    "BNI": "dni",
    "DNI": "dni",
    "DHI": "dhi",
    "GHIc": "ghi_clear",
    "BHIc": "bhi_clear",
    "BNIc": "dni_clear",
    "DNIc": "dni_clear",
    "DHIc": "dhi_clear",
}


@dataclass
class CamsPointResult:
    data: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    data_path: Path
    metadata_path: Path


@dataclass
class CamsAreaResult:
    netcdf_paths: List[Path]
    metadata: Dict[str, Any]
    metadata_path: Path


def _to_utc_naive_timestamp(dt: Any) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _build_client(api_url: Optional[str] = None, api_key: Optional[str] = None, quiet: bool = True) -> cdsapi.Client:
    kwargs: Dict[str, Any] = {"quiet": quiet}
    if api_key is not None and api_url is None:
        kwargs["url"] = ADS_API_URL
        kwargs["key"] = api_key
    else:
        if api_url is not None:
            kwargs["url"] = api_url
        if api_key is not None:
            kwargs["key"] = api_key
    return cdsapi.Client(**kwargs)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _extract_or_move_single_file(raw_path: Path, final_path: Path, allowed_suffixes: Iterable[str]) -> Path:
    allowed_suffixes = {suffix.lower() for suffix in allowed_suffixes}
    _ensure_parent(final_path)

    if zipfile.is_zipfile(raw_path):
        with zipfile.ZipFile(raw_path, "r") as zf:
            file_members = [name for name in zf.namelist() if not name.endswith("/")]
            matched = [name for name in file_members if Path(name).suffix.lower() in allowed_suffixes]

            if not matched:
                if len(file_members) == 1:
                    matched = file_members
                else:
                    raise ValueError(f"No file with suffix {sorted(allowed_suffixes)} found in archive: {raw_path}")

            member = matched[0]
            with zf.open(member) as src, open(final_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        raw_path.unlink(missing_ok=True)
    else:
        if final_path.exists():
            final_path.unlink()
        raw_path.replace(final_path)

    return final_path


def _normalize_area_variables(variables: Optional[Sequence[str]]) -> List[str]:
    if variables is None:
        variables = DEFAULT_AREA_VARIABLES

    normalized: List[str] = []
    seen = set()

    for var in variables:
        official = AREA_VARIABLE_MAP.get(var, var)
        if official not in seen:
            normalized.append(official)
            seen.add(official)

    return normalized


def _month_keys(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[int, int]]:
    months: List[Tuple[int, int]] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1)

    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = pd.Timestamp(year=current.year + 1, month=1, day=1)
        else:
            current = pd.Timestamp(year=current.year, month=current.month + 1, day=1)

    return months


def _default_point_csv_name(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"cams_{lat:.4f}_{lon:.4f}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"


def _default_area_prefix(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> str:
    return f"cams_area_{lat_1:.2f}_{lon_1:.2f}_{lat_2:.2f}_{lon_2:.2f}"


def read_cams_point_csv(csv_path: str | Path, map_variables: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse a CAMS point CSV file into a time-indexed DataFrame.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the downloaded CAMS CSV (semicolon-separated with a
        ``# Observation period`` header line).
    map_variables : bool, default True
        Rename columns to standard names (``ghi``, ``dni``, ``dhi``,
        ``ghi_clear``, ...) via ``VARIABLE_MAP``.

    Returns
    -------
    data : pandas.DataFrame
        Table indexed by UTC timestamps (observation-period column dropped).
    metadata : dict
        ``source_file``, ``row_count`` and, when non-empty, ``start_utc``/
        ``end_utc``.

    Raises
    ------
    ValueError
        When the CAMS header line or the observation-period column is missing.
    """
    csv_path = Path(csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata: Dict[str, Any] = {}
    header_line = None
    data_start_idx = None

    for idx, line in enumerate(lines):
        line = line.rstrip("\n")
        if line.startswith("# Observation period"):
            header_line = line.lstrip("# ").split(";")
            data_start_idx = idx + 1
            break
        if line.startswith("#") and ": " in line:
            key, value = line.lstrip("# ").split(": ", 1)
            metadata[key] = value

    if header_line is None or data_start_idx is None:
        raise ValueError("Could not find CAMS CSV header line '# Observation period...'")

    data_text = "".join(lines[data_start_idx:])
    df = pd.read_csv(StringIO(data_text), sep=";", header=None, names=header_line)

    if "Observation period" not in df.columns:
        raise ValueError("Missing 'Observation period' column in CAMS CSV.")

    obs_period = df["Observation period"].astype(str).str.split("/")
    time_index = pd.DatetimeIndex(pd.to_datetime(obs_period.str[0], utc=True)).tz_convert(None)
    df.index = time_index
    df.index.name = "time"

    df = df.drop(columns=["Observation period"])

    if map_variables:
        df = df.rename(columns=VARIABLE_MAP)

    metadata["source_file"] = str(csv_path)
    metadata["row_count"] = int(len(df))

    if not df.empty:
        metadata["start_utc"] = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
        metadata["end_utc"] = df.index.max().strftime("%Y-%m-%d %H:%M:%S")

    return df, metadata


def _build_point_request(lat: float, lon: float, start: Any, end: Any, file_format: str, sky_type: str, time_step: str, time_ref: str, altitude: Optional[float]) -> Dict[str, Any]:
    start_ts = _to_utc_naive_timestamp(start)
    end_ts = _to_utc_naive_timestamp(end)

    if end_ts < start_ts:
        raise ValueError("end must be later than or equal to start.")
    if file_format not in {"csv", "netcdf"}:
        raise ValueError("file_format must be 'csv' or 'netcdf'.")
    if sky_type not in {"clear", "observed_cloud"}:
        raise ValueError("sky_type must be 'clear' or 'observed_cloud'.")
    if time_step not in TIMESTEP_MAP:
        raise ValueError(f"time_step must be one of {list(TIMESTEP_MAP.keys())}")
    if time_ref not in TIME_REF_MAP:
        raise ValueError(f"time_ref must be one of {list(TIME_REF_MAP.keys())}")

    altitude_value = "-999." if altitude is None else str(float(altitude))

    request = {"sky_type": sky_type, "location": {"latitude": float(lat), "longitude": float(lon)}, "altitude": altitude_value, "date": f"{start_ts.strftime('%Y-%m-%d')}/{end_ts.strftime('%Y-%m-%d')}", "time_step": TIMESTEP_MAP[time_step], "time_reference": TIME_REF_MAP[time_ref], "format": file_format}
    return request


def _build_area_request(year: int, month: int, variable: str, sky_type: str, area: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    if sky_type not in {"clear", "observed_cloud"}:
        raise ValueError("sky_type must be 'clear' or 'observed_cloud'.")
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12.")

    request: Dict[str, Any] = {"variable": [variable], "sky_type": [sky_type], "version": [AREA_VERSION], "year": [f"{year:04d}"], "month": [f"{month:02d}"]}

    if area is not None:
        if len(area) != 4:
            raise ValueError("area must be [north, west, south, east].")
        request["area"] = [float(v) for v in area]

    return request


def download_cams_point(lat: float, lon: float, start: Any, end: Any, save_path: Optional[str | Path] = None, file_format: str = "csv", sky_type: str = "observed_cloud", time_step: str = "1min", time_ref: str = "UT", altitude: Optional[float] = None, quiet: bool = True, parse_csv: bool = True, map_variables: bool = True, api_url: Optional[str] = None, api_key: Optional[str] = None) -> CamsPointResult:
    """
    Download CAMS satellite-derived radiation for one point from the ADS API.

    Parameters
    ----------
    lat : float
        Latitude [degrees].
    lon : float
        Longitude [degrees].
    start : Any
        Start of the period; anything `pandas.Timestamp` accepts, converted to UTC (tz-aware values are converted, naive values are taken as-is).
    end : Any
        End of the period (same parsing as `start`); must not precede `start` (`ValueError`).
    save_path : str or pathlib.Path, optional
        Output path; defaults to `cams_<lat>_<lon>_<start>_<end>.csv`. The suffix follows `file_format` (`.csv` or `.nc`).
    file_format : str, default "csv"
        `"csv"` or `"netcdf"` (`ValueError` otherwise).
    sky_type : str, default "observed_cloud"
        `"clear"` or `"observed_cloud"` (`ValueError` otherwise).
    time_step : str, default "1min"
        One of `"1min"`, `"15min"`, `"1h"`, `"1d"`, `"1M"` (`ValueError` otherwise).
    time_ref : str, default "UT"
        `"UT"` (universal time) or `"TST"` (true solar time) (`ValueError` otherwise).
    altitude : float, optional
        Site altitude [m]; when `None` the service default (-999) is sent.
    quiet : bool, default True
        Suppress `cdsapi` client progress output.
    parse_csv : bool, default True
        CSV only: parse the download, standardize it and rewrite it with a leading `UTC_time` column. Ignored for NetCDF (`data` stays `None`).
    map_variables : bool, default True
        Rename parsed columns to standard names (`ghi`, `dni`, `dhi`, `ghi_clear`, ...) via `VARIABLE_MAP`.
    api_url : str, optional
        ADS API endpoint override.
    api_key : str, optional
        ADS API key override.

    Returns
    -------
    CamsPointResult dataclass — `data` (pandas.DataFrame when parsed, else `None`), `metadata` (dict), `data_path`, `metadata_path` (pathlib.Path). A `<stem>_metadata.json` sidecar is written.

    Notes
    -----
    A parsed CSV is rewritten in place with columns ordered `UTC_time`, then GHI/DNI/DHI (observed and clear-sky), then the rest. Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.
    """
    start_ts = _to_utc_naive_timestamp(start)
    end_ts = _to_utc_naive_timestamp(end)

    if save_path is None:
        save_path = _default_point_csv_name(lat, lon, start_ts, end_ts)

    save_path = Path(save_path)

    if file_format == "csv":
        save_path = save_path.with_suffix(".csv")
        allowed_suffixes = {".csv"}
    else:
        save_path = save_path.with_suffix(".nc")
        allowed_suffixes = {".nc", ".netcdf"}

    raw_download = save_path.with_name(save_path.stem + "__raw.download")
    metadata_path = save_path.with_name(save_path.stem + "_metadata.json")

    client = _build_client(api_url=api_url, api_key=api_key, quiet=quiet)
    request = _build_point_request(lat=lat, lon=lon, start=start, end=end, file_format=file_format, sky_type=sky_type, time_step=time_step, time_ref=time_ref, altitude=altitude)

    _ensure_parent(raw_download)
    client.retrieve(POINT_DATASET, request, str(raw_download))

    final_path = _extract_or_move_single_file(raw_path=raw_download, final_path=save_path, allowed_suffixes=allowed_suffixes)

    data: Optional[pd.DataFrame] = None
    parsed_meta: Dict[str, Any] = {}

    if file_format == "csv" and parse_csv:
        data, parsed_meta = read_cams_point_csv(final_path, map_variables=map_variables)

        export_df = data.copy()
        export_df.insert(0, "UTC_time", data.index.strftime("%Y-%m-%d %H:%M:%S"))
        export_df = export_df.reset_index(drop=True)

        ordered_cols = ["ghi", "dni", "dhi", "ghi_clear", "dni_clear", "dhi_clear", "bhi", "bhi_clear"]
        existing_cols = [col for col in ordered_cols if col in export_df.columns]
        other_cols = [col for col in export_df.columns if col not in existing_cols and col != "UTC_time"]
        export_df = export_df[["UTC_time"] + existing_cols + other_cols]

        export_df.to_csv(final_path, index=False, encoding="utf-8-sig")
        data = export_df

    metadata: Dict[str, Any] = {"mode": "point", "dataset": POINT_DATASET, "request": request, "output_file": str(final_path), "file_format": file_format, "sky_type": sky_type, "time_step": time_step, "time_ref": time_ref, "parse_csv": parse_csv, "map_variables": map_variables}
    metadata.update(parsed_meta)

    _write_json(metadata, metadata_path)

    return CamsPointResult(data=data, metadata=metadata, data_path=final_path, metadata_path=metadata_path)


def download_cams_area(lat_1: Optional[float] = None, lon_1: Optional[float] = None, lat_2: Optional[float] = None, lon_2: Optional[float] = None, start: Any = None, end: Any = None, variables: Optional[Sequence[str]] = None, sky_type: str = "observed_cloud", output_dir: str | Path = "CAMS_data", quiet: bool = True, file_prefix: Optional[str] = None, api_url: Optional[str] = None, api_key: Optional[str] = None) -> CamsAreaResult:
    """
    Download CAMS gridded satellite radiation for a period as NetCDF files (one per variable and month).

    Parameters
    ----------
    lat_1 : float, optional
        Corner latitude [degrees]. When any of the four corners is `None` the whole-domain request is used (no area restriction); when all four are given they are sorted internally to [north, west, south, east].
    lon_1 : float, optional
        Corner longitude [degrees]; see `lat_1`.
    lat_2 : float, optional
        Corner latitude [degrees]; see `lat_1`.
    lon_2 : float, optional
        Corner longitude [degrees]; see `lat_1`.
    start : Any
        Start of the period; required (`ValueError` when `None`).
    end : Any
        End of the period; required; must not precede `start` (`ValueError`).
    variables : sequence of str, optional
        Local aliases (`ghi`, `dni`, `dhi`, `bhi`) resolved via `AREA_VARIABLE_MAP`; defaults to `["ghi", "dni", "dhi"]`.
    sky_type : str, default "observed_cloud"
        `"clear"` or `"observed_cloud"` (`ValueError` otherwise).
    output_dir : str or pathlib.Path, default "CAMS_data"
        Directory for NetCDF parts and metadata (created if missing).
    file_prefix : str, optional
        Output file prefix; defaults to `cams_area_<corners>` when an area is given, else `cams_area`.
    quiet : bool, default True
        Suppress `cdsapi` client progress output.
    api_url : str, optional
        ADS API endpoint override.
    api_key : str, optional
        ADS API key override.

    Returns
    -------
    CamsAreaResult dataclass — `netcdf_paths` (list of pathlib.Path), `metadata` (dict), `metadata_path`. A `<prefix>_metadata.json` sidecar is written.

    Notes
    -----
    One retrieve is issued per variable and calendar month. Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.
    """
    if start is None or end is None:
        raise ValueError("start and end must be provided in area mode.")

    start_ts = _to_utc_naive_timestamp(start)
    end_ts = _to_utc_naive_timestamp(end)

    if end_ts < start_ts:
        raise ValueError("end must be later than or equal to start.")

    area = None
    if None not in [lat_1, lon_1, lat_2, lon_2]:
        north = max(float(lat_1), float(lat_2))
        south = min(float(lat_1), float(lat_2))
        west = min(float(lon_1), float(lon_2))
        east = max(float(lon_1), float(lon_2))
        area = [north, west, south, east]

    normalized_variables = _normalize_area_variables(variables)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_prefix is None:
        if area is not None:
            file_prefix = _default_area_prefix(lat_1=float(lat_1), lon_1=float(lon_1), lat_2=float(lat_2), lon_2=float(lon_2))
        else:
            file_prefix = "cams_area"

    client = _build_client(api_url=api_url, api_key=api_key, quiet=quiet)

    netcdf_paths: List[Path] = []
    chunk_requests: List[Dict[str, Any]] = []

    for variable in normalized_variables:
        for year, month in _month_keys(start_ts, end_ts):
            request = _build_area_request(year=year, month=month, variable=variable, sky_type=sky_type, area=area)

            stem = f"{file_prefix}_{variable}_{year:04d}{month:02d}"
            raw_download = output_dir / f"{stem}__raw.download"
            final_path = output_dir / f"{stem}.nc"

            client.retrieve(AREA_DATASET, request, str(raw_download))
            _extract_or_move_single_file(raw_path=raw_download, final_path=final_path, allowed_suffixes={".nc", ".netcdf"})

            netcdf_paths.append(final_path)
            chunk_requests.append({"variable": variable, "year": year, "month": month, "request": request, "output_file": str(final_path)})

    metadata_path = output_dir / f"{file_prefix}_metadata.json"
    mapped_variables = [key for key, value in AREA_VARIABLE_MAP.items() if value in normalized_variables]

    metadata: Dict[str, Any] = {"mode": "area", "dataset": AREA_DATASET, "start_utc": start_ts.strftime("%Y-%m-%d %H:%M:%S"), "end_utc": end_ts.strftime("%Y-%m-%d %H:%M:%S"), "variables": normalized_variables, "mapped_variables": mapped_variables, "sky_type": sky_type, "area": area, "gridded_version": AREA_VERSION, "netcdf_count": len(netcdf_paths), "netcdf_paths": [str(path) for path in netcdf_paths], "requests": chunk_requests}

    _write_json(metadata, metadata_path)

    return CamsAreaResult(netcdf_paths=netcdf_paths, metadata=metadata, metadata_path=metadata_path)


def download_cams(mode: str = "point", quiet: bool = True, api_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any) -> CamsPointResult | CamsAreaResult:
    """
    Unified CAMS entry point; dispatches to the point or area downloader by `mode`.

    Parameters
    ----------
    mode : str, default "point"
        `"point"` or `"area"` (case-insensitive; `ValueError` otherwise).
    quiet : bool, default True
        Suppress `cdsapi` client progress output.
    api_url : str, optional
        ADS API endpoint override.
    api_key : str, optional
        ADS API key override.
    **kwargs : keyword arguments
        `mode="point"` requires `lat`, `lon`, `start`, `end` (optional `save_path`, `file_format`, `sky_type`, `time_step`, `time_ref`, `altitude`, `parse_csv`, `map_variables`); `mode="area"` requires `start`, `end` (optional `lat_1`, `lon_1`, `lat_2`, `lon_2`, `variables`, `sky_type`, `output_dir`, `file_prefix`). Missing required keys raise `ValueError`.

    Returns
    -------
    CamsPointResult or CamsAreaResult — whichever dataclass the selected downloader returns.

    Notes
    -----
    Uses the CDS `cdsapi` client: passing only `api_key` targets the public endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from `~/.cdsapirc`, with `api_url`/`api_key` overriding individual values.
    """
    mode = mode.lower().strip()

    if mode == "point":
        for key in ["lat", "lon", "start", "end"]:
            if key not in kwargs:
                raise ValueError(f"mode='point' requires keyword argument: {key}")

        return download_cams_point(lat=kwargs["lat"], lon=kwargs["lon"], start=kwargs["start"], end=kwargs["end"], save_path=kwargs.get("save_path"), file_format=kwargs.get("file_format", "csv"), sky_type=kwargs.get("sky_type", "observed_cloud"), time_step=kwargs.get("time_step", "1min"), time_ref=kwargs.get("time_ref", "UT"), altitude=kwargs.get("altitude"), quiet=quiet, parse_csv=kwargs.get("parse_csv", True), map_variables=kwargs.get("map_variables", True), api_url=api_url, api_key=api_key)

    if mode == "area":
        for key in ["start", "end"]:
            if key not in kwargs:
                raise ValueError(f"mode='area' requires keyword argument: {key}")

        return download_cams_area(lat_1=kwargs.get("lat_1"), lon_1=kwargs.get("lon_1"), lat_2=kwargs.get("lat_2"), lon_2=kwargs.get("lon_2"), start=kwargs["start"], end=kwargs["end"], variables=kwargs.get("variables"), sky_type=kwargs.get("sky_type", "observed_cloud"), output_dir=kwargs.get("output_dir", "CAMS_data"), quiet=quiet, file_prefix=kwargs.get("file_prefix"), api_url=api_url, api_key=api_key)

    raise ValueError("mode must be either 'point' or 'area'.")


__all__ = [
    "ADS_API_URL",
    "POINT_DATASET",
    "AREA_DATASET",
    "AREA_VERSION",
    "DEFAULT_AREA_VARIABLES",
    "TIMESTEP_MAP",
    "TIME_REF_MAP",
    "AREA_VARIABLE_MAP",
    "VARIABLE_MAP",
    "CamsPointResult",
    "CamsAreaResult",
    "read_cams_point_csv",
    "download_cams_point",
    "download_cams_area",
    "download_cams",
]
