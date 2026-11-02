"""
ERA5 downloader for irradpy2.

Main capabilities
-----------------
- Download ERA5 point time-series data and export CSV
- Download ERA5 area gridded data as monthly GRIB files
- Keep UTC output style simple and consistent with irradpy2
- Use unified output names consistent with MERRA-2 when possible

Requirements
------------
- cdsapi >= 0.7.7
- pandas
"""

from __future__ import annotations

import calendar
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cdsapi
import pandas as pd


CDS_API_URL = "https://cds.climate.copernicus.eu/api"

POINT_DATASET = "reanalysis-era5-single-levels-timeseries"
AREA_DATASET = "reanalysis-era5-single-levels"

ERA5_HOURS = [f"{hour:02d}:00" for hour in range(24)]

DEFAULT_POINT_VARIABLES = ["ssrd", "strd", "t2m", "sp", "tcc", "tp"]
DEFAULT_AREA_VARIABLES = ["ssrd", "strd", "t2m", "sp", "tcc"]


# -----------------------------------------------------------------------------
# Input aliases and output names
# -----------------------------------------------------------------------------

INPUT_VARIABLE_MAP: Dict[str, str] = {
    "ghi": "surface_solar_radiation_downwards",
    "longwave_down": "surface_thermal_radiation_downwards",
    "temp_air": "2m_temperature",
    "temp_dew": "2m_dewpoint_temperature",
    "pressure": "surface_pressure",
    "precipitation": "total_precipitation",
    "cloud_fraction": "total_cloud_cover",
    "wind_u_10m": "10m_u_component_of_wind",
    "wind_v_10m": "10m_v_component_of_wind",
    "wind_u_100m": "100m_u_component_of_wind",
    "wind_v_100m": "100m_v_component_of_wind",
    "boundary_layer_height": "boundary_layer_height",
    "cloud_base_height": "cloud_base_height",
    "mean_sea_level_pressure": "mean_sea_level_pressure",
    "sea_surface_temperature": "sea_surface_temperature",
    "skin_temperature": "skin_temperature",
    "wind_gust_10m": "10m_wind_gust_since_previous_post_processing",

    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "tcc": "total_cloud_cover",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "blh": "boundary_layer_height",
    "cbh": "cloud_base_height",
    "msl": "mean_sea_level_pressure",
    "sst": "sea_surface_temperature",
    "skt": "skin_temperature",
    "fg10": "10m_wind_gust_since_previous_post_processing",
}

OUTPUT_VARIABLE_MAP: Dict[str, str] = {
    "valid_time": "UTC_time",
    "time": "UTC_time",

    "surface_solar_radiation_downwards": "ghi",
    "surface_thermal_radiation_downwards": "longwave_down",
    "2m_temperature": "temp_air",
    "2m_dewpoint_temperature": "temp_dew",
    "surface_pressure": "pressure",
    "total_precipitation": "precipitation",
    "total_cloud_cover": "cloud_fraction",
    "10m_u_component_of_wind": "wind_u_10m",
    "10m_v_component_of_wind": "wind_v_10m",
    "100m_u_component_of_wind": "wind_u_100m",
    "100m_v_component_of_wind": "wind_v_100m",
    "boundary_layer_height": "boundary_layer_height",
    "cloud_base_height": "cloud_base_height",
    "mean_sea_level_pressure": "mean_sea_level_pressure",
    "sea_surface_temperature": "sea_surface_temperature",
    "skin_temperature": "skin_temperature",
    "10m_wind_gust_since_previous_post_processing": "wind_gust_10m",

    "ssrd": "ghi",
    "strd": "longwave_down",
    "t2m": "temp_air",
    "d2m": "temp_dew",
    "sp": "pressure",
    "tp": "precipitation",
    "tcc": "cloud_fraction",
    "u10": "wind_u_10m",
    "v10": "wind_v_10m",
    "u100": "wind_u_100m",
    "v100": "wind_v_100m",
    "blh": "boundary_layer_height",
    "cbh": "cloud_base_height",
    "msl": "mean_sea_level_pressure",
    "sst": "sea_surface_temperature",
    "skt": "skin_temperature",
    "fg10": "wind_gust_10m",
}


def _k_to_c(value: pd.Series) -> pd.Series:
    return value - 273.15


def _j_to_w(value: pd.Series) -> pd.Series:
    return value / 3600.0


def _m_to_mm(value: pd.Series) -> pd.Series:
    return value * 1000.0


UNIT_CONVERTERS: Dict[str, Any] = {
    "ssrd": _j_to_w,
    "strd": _j_to_w,
    "t2m": _k_to_c,
    "d2m": _k_to_c,
    "tp": _m_to_mm,
    "sst": _k_to_c,
    "skt": _k_to_c,

    "surface_solar_radiation_downwards": _j_to_w,
    "surface_thermal_radiation_downwards": _j_to_w,
    "2m_temperature": _k_to_c,
    "2m_dewpoint_temperature": _k_to_c,
    "total_precipitation": _m_to_mm,
    "sea_surface_temperature": _k_to_c,
    "skin_temperature": _k_to_c,
}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class PointResult:
    data: pd.DataFrame
    metadata: Dict[str, Any]
    csv_path: Path
    metadata_path: Path


@dataclass
class AreaResult:
    grib_paths: List[Path]
    metadata: Dict[str, Any]
    metadata_path: Path


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _to_utc_naive_timestamp(dt: Any) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _normalize_variables(variables: Optional[Sequence[str]], default_variables: Sequence[str]) -> List[str]:
    if variables is None:
        variables = default_variables

    normalized: List[str] = []
    seen = set()

    for var in variables:
        official = INPUT_VARIABLE_MAP.get(var, var)
        if official not in seen:
            normalized.append(official)
            seen.add(official)

    return normalized


def _build_client(api_url: Optional[str] = None, api_key: Optional[str] = None, quiet: bool = True) -> cdsapi.Client:
    kwargs: Dict[str, Any] = {"quiet": quiet}

    if api_key is not None and api_url is None:
        kwargs["url"] = CDS_API_URL
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


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1)

    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_start = current
        month_end = pd.Timestamp(year=current.year, month=current.month, day=last_day)
        chunk_start = max(start, month_start)
        chunk_end = min(end, month_end)

        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))

        if current.month == 12:
            current = pd.Timestamp(year=current.year + 1, month=1, day=1)
        else:
            current = pd.Timestamp(year=current.year, month=current.month + 1, day=1)

    return chunks


def _default_point_csv_name(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"era5_{lat:.4f}_{lon:.4f}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"


def _default_area_prefix(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> str:
    return f"era5_area_{lat_1:.2f}_{lon_1:.2f}_{lat_2:.2f}_{lon_2:.2f}"


# -----------------------------------------------------------------------------
# Parsing and standardization
# -----------------------------------------------------------------------------

def read_era5_point_csv(csv_path: str | Path, map_variables: bool = True, convert_units: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    metadata: Dict[str, Any] = {"source_file": str(csv_path), "row_count": int(len(df))}

    if "latitude" in df.columns and not df.empty:
        metadata["latitude"] = float(df["latitude"].iloc[0])

    if "longitude" in df.columns and not df.empty:
        metadata["longitude"] = float(df["longitude"].iloc[0])

    time_column = None
    for candidate in ["valid_time", "time"]:
        if candidate in df.columns:
            time_column = candidate
            break

    if time_column is None:
        raise ValueError("Could not find 'valid_time' or 'time' column in ERA5 point CSV.")

    time_index = pd.DatetimeIndex(pd.to_datetime(df[time_column], utc=True)).tz_localize(None)
    df.index = time_index
    df.index.name = "time"

    drop_cols = [col for col in [time_column, "latitude", "longitude"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    if convert_units:
        for column in df.columns:
            converter = UNIT_CONVERTERS.get(column)
            if converter is not None:
                df[column] = converter(df[column])

    if map_variables:
        df = df.rename(columns=OUTPUT_VARIABLE_MAP)

    if not df.empty:
        metadata["start_utc"] = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
        metadata["end_utc"] = df.index.max().strftime("%Y-%m-%d %H:%M:%S")

    return df, metadata


# -----------------------------------------------------------------------------
# Downloaders
# -----------------------------------------------------------------------------

def download_era5_point(lat: float, lon: float, start: Any, end: Any, variables: Optional[Sequence[str]] = None, save_path: Optional[str | Path] = None, api_url: Optional[str] = None, api_key: Optional[str] = None, quiet: bool = True, map_variables: bool = True, convert_units: bool = True) -> PointResult:
    start_ts = _to_utc_naive_timestamp(start)
    end_ts = _to_utc_naive_timestamp(end)

    if end_ts < start_ts:
        raise ValueError("end must be later than or equal to start.")

    normalized_variables = _normalize_variables(variables=variables, default_variables=DEFAULT_POINT_VARIABLES)

    if save_path is None:
        save_path = _default_point_csv_name(lat, lon, start_ts, end_ts)

    save_path = Path(save_path)
    if save_path.suffix.lower() != ".csv":
        save_path = save_path.with_suffix(".csv")

    raw_download = save_path.with_name(save_path.stem + "__raw.download")
    raw_csv = save_path.with_name(save_path.stem + "__raw.csv")
    metadata_path = save_path.with_name(save_path.stem + "_metadata.json")

    client = _build_client(api_url=api_url, api_key=api_key, quiet=quiet)

    request = {"variable": normalized_variables, "location": {"latitude": float(lat), "longitude": float(lon)}, "date": [f"{start_ts.strftime('%Y-%m-%d')}/{end_ts.strftime('%Y-%m-%d')}"], "data_format": "csv"}

    _ensure_parent(raw_download)
    client.retrieve(POINT_DATASET, request, str(raw_download))

    _extract_or_move_single_file(raw_path=raw_download, final_path=raw_csv, allowed_suffixes={".csv"})

    data, parsed_meta = read_era5_point_csv(raw_csv, map_variables=map_variables, convert_units=convert_units)

    export_df = data.copy()
    export_df.insert(0, "UTC_time", export_df.index.strftime("%Y-%m-%d %H:%M:%S"))
    export_df = export_df.reset_index(drop=True)

    ordered_cols = ["ghi", "longwave_down", "temp_air", "pressure", "cloud_fraction", "precipitation"]
    existing_cols = [col for col in ordered_cols if col in export_df.columns]
    other_cols = [col for col in export_df.columns if col not in existing_cols and col != "UTC_time"]
    export_df = export_df[["UTC_time"] + existing_cols + other_cols]

    export_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    raw_csv.unlink(missing_ok=True)

    metadata: Dict[str, Any] = {"mode": "point", "dataset": POINT_DATASET, "request": request, "output_csv": str(save_path), "column_names": list(export_df.columns), "map_variables": map_variables, "convert_units": convert_units}
    metadata.update(parsed_meta)

    _write_json(metadata, metadata_path)

    return PointResult(data=export_df, metadata=metadata, csv_path=save_path, metadata_path=metadata_path)


def download_era5_area(lat_1: float, lon_1: float, lat_2: float, lon_2: float, start: Any, end: Any, variables: Optional[Sequence[str]] = None, output_dir: str | Path = "ERA5_data", file_prefix: Optional[str] = None, api_url: Optional[str] = None, api_key: Optional[str] = None, quiet: bool = True) -> AreaResult:
    start_ts = _to_utc_naive_timestamp(start)
    end_ts = _to_utc_naive_timestamp(end)

    if end_ts < start_ts:
        raise ValueError("end must be later than or equal to start.")
    if lat_1 == lat_2:
        raise ValueError("lat_1 and lat_2 cannot be equal.")
    if lon_1 == lon_2:
        raise ValueError("lon_1 and lon_2 cannot be equal.")

    north = max(float(lat_1), float(lat_2))
    south = min(float(lat_1), float(lat_2))
    west = min(float(lon_1), float(lon_2))
    east = max(float(lon_1), float(lon_2))

    normalized_variables = _normalize_variables(variables=variables, default_variables=DEFAULT_AREA_VARIABLES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_prefix is None:
        file_prefix = _default_area_prefix(lat_1=lat_1, lon_1=lon_1, lat_2=lat_2, lon_2=lon_2)

    metadata_path = output_dir / f"{file_prefix}_metadata.json"
    client = _build_client(api_url=api_url, api_key=api_key, quiet=quiet)

    chunk_infos: List[Dict[str, Any]] = []
    grib_paths: List[Path] = []

    for chunk_start, chunk_end in _month_chunks(start_ts, end_ts):
        year_str = f"{chunk_start.year:04d}"
        month_str = f"{chunk_start.month:02d}"
        days = [f"{day:02d}" for day in range(chunk_start.day, chunk_end.day + 1)]

        request = {"product_type": "reanalysis", "variable": normalized_variables, "year": [year_str], "month": [month_str], "day": days, "time": ERA5_HOURS, "area": [north, west, south, east], "data_format": "grib"}

        part_name = f"{file_prefix}_{year_str}{month_str}.grib"
        part_path = output_dir / part_name
        raw_download = output_dir / f"{file_prefix}_{year_str}{month_str}__raw.download"

        client.retrieve(AREA_DATASET, request, str(raw_download))
        _extract_or_move_single_file(raw_path=raw_download, final_path=part_path, allowed_suffixes={".grib", ".grb"})

        grib_paths.append(part_path)
        chunk_infos.append({"chunk_start": chunk_start.strftime("%Y-%m-%d"), "chunk_end": chunk_end.strftime("%Y-%m-%d"), "grib_path": str(part_path), "request": request})

    mapped_variables = [OUTPUT_VARIABLE_MAP.get(var, var) for var in normalized_variables]

    metadata: Dict[str, Any] = {"mode": "area", "dataset": AREA_DATASET, "lat_1": lat_1, "lon_1": lon_1, "lat_2": lat_2, "lon_2": lon_2, "north": north, "south": south, "west": west, "east": east, "start_utc": start_ts.strftime("%Y-%m-%d %H:%M:%S"), "end_utc": end_ts.strftime("%Y-%m-%d %H:%M:%S"), "variables": normalized_variables, "mapped_variables": mapped_variables, "file_prefix": file_prefix, "grib_count": len(grib_paths), "grib_paths": [str(path) for path in grib_paths], "chunks": chunk_infos}

    _write_json(metadata, metadata_path)

    return AreaResult(grib_paths=grib_paths, metadata=metadata, metadata_path=metadata_path)


# -----------------------------------------------------------------------------
# Unified public entry
# -----------------------------------------------------------------------------

def download_era5(mode: str = "point", start: Any = None, end: Any = None, variables: Optional[Sequence[str]] = None, api_url: Optional[str] = None, api_key: Optional[str] = None, quiet: bool = True, **kwargs: Any) -> PointResult | AreaResult:
    mode = mode.lower().strip()

    if start is None or end is None:
        raise ValueError("start and end must be provided.")

    if mode == "point":
        for key in ["lat", "lon"]:
            if key not in kwargs:
                raise ValueError(f"mode='point' requires keyword argument: {key}")

        return download_era5_point(lat=kwargs["lat"], lon=kwargs["lon"], start=start, end=end, variables=variables, save_path=kwargs.get("save_path"), api_url=api_url, api_key=api_key, quiet=quiet, map_variables=kwargs.get("map_variables", True), convert_units=kwargs.get("convert_units", True))

    if mode == "area":
        for key in ["lat_1", "lon_1", "lat_2", "lon_2"]:
            if key not in kwargs:
                raise ValueError(f"mode='area' requires keyword argument: {key}")

        return download_era5_area(lat_1=kwargs["lat_1"], lon_1=kwargs["lon_1"], lat_2=kwargs["lat_2"], lon_2=kwargs["lon_2"], start=start, end=end, variables=variables, output_dir=kwargs.get("output_dir", "ERA5_data"), file_prefix=kwargs.get("file_prefix"), api_url=api_url, api_key=api_key, quiet=quiet)

    raise ValueError("mode must be either 'point' or 'area'.")


__all__ = [
    "POINT_DATASET",
    "AREA_DATASET",
    "DEFAULT_POINT_VARIABLES",
    "DEFAULT_AREA_VARIABLES",
    "INPUT_VARIABLE_MAP",
    "OUTPUT_VARIABLE_MAP",
    "PointResult",
    "AreaResult",
    "read_era5_point_csv",
    "download_era5_point",
    "download_era5_area",
    "download_era5",
]