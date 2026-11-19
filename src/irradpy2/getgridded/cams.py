"""
CAMS downloader for irradpy2.

Main capabilities
-----------------
- Download CAMS point time-series data
- Download CAMS gridded monthly netCDF data
- Export interval-mean irradiance in W/m2 with period-end timestamps
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
import xarray as xr

from .era5 import _transform_netcdf_download, _netcdf_unit_key


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
    "Clear sky GHI": "ghi_clear",
    "Clear sky BHI": "bhi_clear",
    "Clear sky BNI": "dni_clear",
    "Clear sky DNI": "dni_clear",
    "Clear sky DHI": "dhi_clear",
    "TOA": "toa",
}


RADIATION_COLUMNS = {
    "ghi", "bhi", "dni", "dhi", "ghi_clear", "bhi_clear",
    "dni_clear", "dhi_clear", "toa",
}
UNCORRECTED_RADIATION_COLUMNS = {"GHI no corr", "BHI no corr", "DHI no corr", "BNI no corr"}


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


def _build_client(api_url: Optional[str] = None, api_key: Optional[str] = None, quiet: bool = False) -> cdsapi.Client:
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


def read_cams_point_csv(
    csv_path: str | Path, map_variables: bool = True, *, convert_units: bool = True,
    time_ref: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read native CAMS CSV, using the declared time reference when available."""
    if time_ref is not None and time_ref not in TIME_REF_MAP:
        raise ValueError(f"time_ref must be one of {list(TIME_REF_MAP)}.")
    csv_path = Path(csv_path)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    metadata: Dict[str, Any] = {}
    header_line = None
    data_start_idx = None

    for idx, line in enumerate(lines):
        line = line.rstrip("\n")
        if line.startswith("# Observation period"):
            header_line = [field.strip() for field in line.lstrip("# ").split(";")]
            data_start_idx = idx + 1
            break
        if line.startswith("#") and ": " in line:
            key, value = line.lstrip("# ").split(": ", 1)
            metadata[key] = value

    if header_line is None or data_start_idx is None:
        raise ValueError("Could not find CAMS CSV header line '# Observation period...'")

    header_ref = next((str(value).lower() for key, value in metadata.items()
                       if key.lower().startswith("time reference")), "")
    if "solar" in header_ref or header_ref == "tst":
        declared_ref = "TST"
    elif "universal" in header_ref or header_ref in {"ut", "utc"}:
        declared_ref = "UT"
    else:
        declared_ref = None
    if declared_ref is not None and time_ref is not None and time_ref != declared_ref:
        raise ValueError(f"CAMS response uses {declared_ref}, but {time_ref} was requested.")
    time_ref = declared_ref or time_ref or "UT"

    data_text = "".join(lines[data_start_idx:])
    df = pd.read_csv(StringIO(data_text), sep=";", header=None, names=header_line)

    if "Observation period" not in df.columns:
        raise ValueError("Missing 'Observation period' column in CAMS CSV.")

    periods = df.pop("Observation period").astype("string").str.split("/", n=1, expand=True)
    if periods.shape[1] != 2:
        raise ValueError("Each CAMS observation period must contain a start and an end.")
    if time_ref == "TST":
        periods = periods.apply(lambda col: col.str.replace(r"(?:Z|[+-]\d{2}:?\d{2})$", "", regex=True))
    start = pd.to_datetime(periods[0], errors="coerce", utc=time_ref == "UT")
    end = pd.to_datetime(periods[1], errors="coerce", utc=time_ref == "UT")
    hours = (end - start).dt.total_seconds() / 3600.0
    if hours.isna().any() or (hours <= 0).any():
        raise ValueError("CAMS observation periods must have valid, increasing endpoints.")

    radiation = [
        column for column in df.columns
        if VARIABLE_MAP.get(column) in RADIATION_COLUMNS or column in UNCORRECTED_RADIATION_COLUMNS
    ]
    if not radiation:
        raise ValueError("No recognized irradiance columns were found in the CAMS CSV.")
    if convert_units:
        for column in radiation:
            irradiation = pd.to_numeric(df[column], errors="coerce")
            df[column] = irradiation.where(irradiation >= 0) / hours

    df.index = pd.DatetimeIndex(end)
    if time_ref == "UT":
        df.index = df.index.tz_convert(None)
    df.index.name = "time"

    if map_variables:
        df = df.rename(columns=VARIABLE_MAP)
    if not df.columns.is_unique:
        raise ValueError("CAMS CSV contains duplicate output variable names.")

    unit = "W/m2" if convert_units else "Wh/m2"
    metadata.update({
        "source_file": str(csv_path),
        "row_count": len(df),
        "time_reference": time_ref,
        "timestamp_label": "interval_end",
        "units": {(VARIABLE_MAP.get(col, col) if map_variables else col): unit for col in radiation},
    })

    if not df.empty:
        if time_ref == "UT":
            metadata["start_utc"] = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
            metadata["end_utc"] = df.index.max().strftime("%Y-%m-%d %H:%M:%S")
        metadata["integration_seconds_min"] = float(hours.min() * 3600)
        metadata["integration_seconds_max"] = float(hours.max() * 3600)

    return df, metadata


_ENERGY_UNITS = {"whm-2": 1.0, "wh/m2": 1.0, "jm-2": 1 / 3600, "j/m2": 1 / 3600}
_POWER_UNITS = {"wm-2", "w/m2"}


def _netcdf_integration_hours(ds, time_step, start):
    time = next((ds[name] for name in ("time", "valid_time", "Time") if name in ds.coords), None)
    if time is not None:
        bounds_name = time.attrs.get("bounds") or time.encoding.get("bounds")
        if bounds_name is None:
            bounds_name = next((name for name in ("time_bnds", "time_bounds") if name in ds), None)
        if bounds_name is not None:
            bounds = ds[bounds_name]
            bound_dim, = (dim for dim in bounds.dims if dim not in time.dims)
            if bounds.sizes[bound_dim] != 2:
                raise ValueError("CAMS time bounds must have two endpoints per period.")
            hours = (bounds.isel({bound_dim: 1}) - bounds.isel({bound_dim: 0})) / pd.Timedelta(hours=1)
            if not (hours > 0).all().compute().item():
                raise ValueError("CAMS time bounds must define positive integration periods.")
            return hours

    if time_step == "1M":
        if time is None or time.ndim != 1 or not time.size:
            raise ValueError("Monthly CAMS NetCDF requires time coordinates or time bounds.")
        dates = pd.DatetimeIndex(time.values)
        if dates[0] == dates[0].normalize() and dates[0].day == 1 and dates[0].to_period("M") == pd.Timestamp(start).to_period("M") + 1:
            dates = dates - pd.Timedelta(1, "ns")
        periods = dates.to_period("M")
        if periods[0] != pd.Timestamp(start).to_period("M") or not periods.is_unique:
            raise ValueError("Cannot identify the monthly CAMS integration periods from its time coordinate.")
        return xr.DataArray(periods.days_in_month * 24, coords=time.coords, dims=time.dims)
    return pd.Timedelta(time_step) / pd.Timedelta(hours=1)


def _standardize_netcdf(datasets, time_step, *, start=None, map_variables=True, time_ref="UT"):
    ds = xr.merge(datasets, compat="no_conflicts", combine_attrs="drop_conflicts")
    names = {name.lower(): alias for name, alias in VARIABLE_MAP.items()}
    names.update({name: alias for alias, name in AREA_VARIABLE_MAP.items()})
    names.update({name: name for name in RADIATION_COLUMNS})
    hours = None
    rename = {}
    units = {}
    for name, data in ds.data_vars.items():
        canonical = names.get(name.lower(), name)
        source_unit = _netcdf_unit_key(data.attrs.get("units"))
        if source_unit in _ENERGY_UNITS:
            if hours is None:
                hours = _netcdf_integration_hours(ds, time_step, start)
            ds[name] = (data.where(data >= 0) * _ENERGY_UNITS[source_unit] / hours).assign_attrs(data.attrs)
            for attribute in ("standard_name", "long_name"):
                ds[name].attrs.pop(attribute, None)
            ds[name].attrs["cell_methods"] = "time: mean"
        elif source_unit not in _POWER_UNITS:
            if canonical in RADIATION_COLUMNS:
                raise ValueError(f"Unexpected CAMS unit for {name}: {data.attrs.get('units')!r}")
            continue
        ds[name].attrs["units"] = "W/m2"
        output_name = canonical if map_variables else name
        units[output_name] = "W/m2"
        if output_name != name:
            rename[name] = output_name

    if not units:
        raise ValueError("No CAMS radiation variables were found in the NetCDF response.")
    return ds.rename(rename).assign_attrs(time_reference=TIME_REF_MAP[time_ref]), units


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


def download_cams_point(lat: float, lon: float, start: Any, end: Any, save_path: Optional[str | Path] = None, file_format: str = "csv", sky_type: str = "observed_cloud", time_step: str = "1min", time_ref: str = "UT", altitude: Optional[float] = None, quiet: bool = False, parse_csv: bool = True, map_variables: bool = True, api_url: Optional[str] = None, api_key: Optional[str] = None, *, convert_units: bool = True) -> CamsPointResult:
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

    data: Optional[pd.DataFrame] = None
    parsed_meta: Dict[str, Any] = {}
    if file_format == "netcdf" and convert_units:
        final_path = save_path
        parsed_meta["units"] = _transform_netcdf_download(
            raw_download, final_path, lambda files: _standardize_netcdf(
                files, time_step, start=start, map_variables=map_variables, time_ref=time_ref
            )
        )
    else:
        final_path = _extract_or_move_single_file(
            raw_path=raw_download, final_path=save_path, allowed_suffixes=allowed_suffixes
        )

    if file_format == "csv" and parse_csv:
        data, parsed_meta = read_cams_point_csv(
            final_path, map_variables=map_variables, convert_units=convert_units, time_ref=time_ref
        )

        export_df = data.copy()
        time_column = "UTC_time" if time_ref == "UT" else "solar_time"
        export_df.insert(0, time_column, data.index.strftime("%Y-%m-%d %H:%M:%S"))
        export_df = export_df.reset_index(drop=True)

        ordered_cols = ["ghi", "dni", "dhi", "ghi_clear", "dni_clear", "dhi_clear", "bhi", "bhi_clear"]
        existing_cols = [col for col in ordered_cols if col in export_df.columns]
        other_cols = [col for col in export_df.columns if col not in existing_cols and col != time_column]
        export_df = export_df[[time_column] + existing_cols + other_cols]

        export_df.to_csv(final_path, index=False, encoding="utf-8-sig")
        data = export_df

    metadata: Dict[str, Any] = {"mode": "point", "dataset": POINT_DATASET, "request": request, "output_file": str(final_path), "file_format": file_format, "sky_type": sky_type, "time_step": time_step, "time_ref": time_ref, "parse_csv": parse_csv, "map_variables": map_variables}
    metadata.update(parsed_meta)
    metadata["convert_units"] = convert_units
    metadata["output_representation"] = "parsed_csv" if data is not None else "standardized_netcdf" if parsed_meta else "provider_native"

    _write_json(metadata, metadata_path)

    return CamsPointResult(data=data, metadata=metadata, data_path=final_path, metadata_path=metadata_path)


def download_cams_area(lat_1: Optional[float] = None, lon_1: Optional[float] = None, lat_2: Optional[float] = None, lon_2: Optional[float] = None, start: Any = None, end: Any = None, variables: Optional[Sequence[str]] = None, sky_type: str = "observed_cloud", output_dir: str | Path = "CAMS_data", quiet: bool = True, file_prefix: Optional[str] = None, api_url: Optional[str] = None, api_key: Optional[str] = None, *, convert_units: bool = True, map_variables: bool = True) -> CamsAreaResult:
    """Download gridded CAMS irradiation as W/m2 in NetCDF by default."""
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
    units: Dict[str, str] = {}

    for variable in normalized_variables:
        for year, month in _month_keys(start_ts, end_ts):
            request = _build_area_request(year=year, month=month, variable=variable, sky_type=sky_type, area=area)

            stem = f"{file_prefix}_{variable}_{year:04d}{month:02d}"
            raw_download = output_dir / f"{stem}__raw.download"
            final_path = output_dir / f"{stem}.nc"

            client.retrieve(AREA_DATASET, request, str(raw_download))
            if convert_units:
                units.update(_transform_netcdf_download(
                    raw_download, final_path, lambda files: _standardize_netcdf(files, "15min", map_variables=map_variables)
                ))
            else:
                _extract_or_move_single_file(raw_path=raw_download, final_path=final_path, allowed_suffixes={".nc", ".netcdf"})

            netcdf_paths.append(final_path)
            chunk_requests.append({"variable": variable, "year": year, "month": month, "request": request, "output_file": str(final_path)})

    metadata_path = output_dir / f"{file_prefix}_metadata.json"
    mapped_variables = [key for key, value in AREA_VARIABLE_MAP.items() if value in normalized_variables]

    metadata: Dict[str, Any] = {"mode": "area", "dataset": AREA_DATASET, "start_utc": start_ts.strftime("%Y-%m-%d %H:%M:%S"), "end_utc": end_ts.strftime("%Y-%m-%d %H:%M:%S"), "variables": normalized_variables, "mapped_variables": mapped_variables, "sky_type": sky_type, "area": area, "gridded_version": AREA_VERSION, "netcdf_count": len(netcdf_paths), "netcdf_paths": [str(path) for path in netcdf_paths], "requests": chunk_requests}

    metadata["output_representation"] = "standardized_netcdf" if convert_units else "provider_native"
    metadata["units"] = units
    _write_json(metadata, metadata_path)

    return CamsAreaResult(netcdf_paths=netcdf_paths, metadata=metadata, metadata_path=metadata_path)


def download_cams(mode: str = "point", quiet: bool = True, api_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any) -> CamsPointResult | CamsAreaResult:
    mode = mode.lower().strip()

    if mode == "point":
        for key in ["lat", "lon", "start", "end"]:
            if key not in kwargs:
                raise ValueError(f"mode='point' requires keyword argument: {key}")

        return download_cams_point(lat=kwargs["lat"], lon=kwargs["lon"], start=kwargs["start"], end=kwargs["end"], save_path=kwargs.get("save_path"), file_format=kwargs.get("file_format", "csv"), sky_type=kwargs.get("sky_type", "observed_cloud"), time_step=kwargs.get("time_step", "1min"), time_ref=kwargs.get("time_ref", "UT"), altitude=kwargs.get("altitude"), quiet=quiet, parse_csv=kwargs.get("parse_csv", True), map_variables=kwargs.get("map_variables", True), api_url=api_url, api_key=api_key, convert_units=kwargs.get("convert_units", True))

    if mode == "area":
        for key in ["start", "end"]:
            if key not in kwargs:
                raise ValueError(f"mode='area' requires keyword argument: {key}")

        return download_cams_area(lat_1=kwargs.get("lat_1"), lon_1=kwargs.get("lon_1"), lat_2=kwargs.get("lat_2"), lon_2=kwargs.get("lon_2"), start=kwargs["start"], end=kwargs["end"], variables=kwargs.get("variables"), sky_type=kwargs.get("sky_type", "observed_cloud"), output_dir=kwargs.get("output_dir", "CAMS_data"), quiet=quiet, file_prefix=kwargs.get("file_prefix"), api_url=api_url, api_key=api_key, convert_units=kwargs.get("convert_units", True), map_variables=kwargs.get("map_variables", True))

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
