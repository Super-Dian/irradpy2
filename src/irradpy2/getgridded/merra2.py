"""
MERRA2 downloader module for irradpy2.

Main capabilities
-----------------
- Download MERRA2 data by area or single point
- mode="area": download NetCDF subset files by bounding box
- mode="point": download single-point CSV time series
- Keep the original MERRA2 variable registry and area-download workflow

Notes
-----
This module only handles downloading. Local NetCDF extraction and clear-sky
post-processing should be placed in separate modules.
"""

import datetime
import _thread
import errno
from contextlib import ExitStack
import json
import logging
import multiprocessing
import os
import re
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http import cookiejar
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple, Union

import netCDF4
import numpy as np
import pandas as pd
import requests
import xarray as xr

from .era5 import (
    _area_request_tag, _concat_netcdf_time, _netcdf_unit_key, _open_netcdf_parts,
    _write_json, _write_netcdf,
)

try:
    import earthaccess
    from pydap.client import get_cmr_urls
    from pydap.net import create_session
except ImportError:
    earthaccess = None
    get_cmr_urls = None
    create_session = None

# =============================================================================
# Variable registry
# =============================================================================

var_list = {
    "rad": {
        "esdt_dir": "M2T1NXRAD.5.12.4",
        "collection": "tavg1_2d_rad_Nx",
        "var_name": ["ALBEDO", "CLDTOT", "SWGDN", "SWGDNCLR", "TAUTOT"],
        "standard_name": "radiation",
    },
    "slv": {
        "esdt_dir": "M2T1NXSLV.5.12.4",
        "collection": "tavg1_2d_slv_Nx",
        "var_name": ["TQV", "TO3", "PS"],
        "standard_name": "surface",
    },
    "aer": {
        "esdt_dir": "M2T1NXAER.5.12.4",
        "collection": "tavg1_2d_aer_Nx",
        "var_name": ["TOTSCATAU", "TOTEXTTAU", "TOTANGSTR"],
        "standard_name": "aerosols",
    },
    "asm": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["PHIS"],
        "standard_name": "parameters",
    },
    "dewpt": {
        "esdt_dir": "M2T1NXSLV.5.12.4",
        "collection": "tavg1_2d_slv_Nx",
        "var_name": ["T2MDEW"],
        "standard_name": "dew_point_temperature",
    },
    "evspsbl": {
        "esdt_dir": "M2T1NXFLX.5.12.4",
        "collection": "tavg1_2d_flx_Nx",
        "var_name": ["EVAP"],
        "standard_name": "water_evaporation_flux",
    },
    "hur": {
        "esdt_dir": "M2T3NPCLD.5.12.4",
        "collection": "tavg3_3d_cld_Np",
        "var_name": ["RH"],
        "standard_name": "relative_humidity",
    },
    "huss": {
        "esdt_dir": "M2T1NXSLV.5.12.4",
        "collection": "tavg1_2d_slv_Nx",
        "var_name": ["QV2M"],
        "standard_name": "specific_humidity",
    },
    "phis": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["PHIS"],
        "standard_name": "surface_geopotential",
    },
    "pr": {
        "esdt_dir": "M2T1NXFLX.5.12.4",
        "collection": "tavg1_2d_flx_Nx",
        "var_name": ["PRECTOT"],
        "standard_name": "precipitation_flux",
    },
    "prc": {
        "esdt_dir": "M2T1NXFLX.5.12.4",
        "collection": "tavg1_2d_flx_Nx",
        "var_name": ["PRECCON"],
        "standard_name": "convective_precipitation_flux",
    },
    "prbc": {
        "esdt_dir": "M2T1NXFLX.5.12.4",
        "collection": "tavg1_2d_flx_Nx",
        "var_name": ["PRECTOTCORR"],
        "standard_name": "precipitation_flux_bias_corr",
    },
    "prmax": {
        "esdt_dir": "M2SDNXSLV.5.12.4",
        "collection": "statD_2d_slv_Nx",
        "var_name": ["TPRECMAX"],
        "standard_name": "precipitation_flux",
    },
    "prsn": {
        "esdt_dir": "M2T1NXFLX.5.12.4",
        "collection": "tavg1_2d_flx_Nx",
        "var_name": ["PRECSNO"],
        "standard_name": "snowfall_flux",
    },
    "rls": {
        "esdt_dir": "M2T1NXRAD.5.12.4",
        "collection": "tavg1_2d_rad_Nx",
        "var_name": ["LWGNT"],
        "standard_name": "surface_net_downward_longwave_flux",
    },
    "sftgif": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["FRLANDICE"],
        "standard_name": "land_ice_area_fraction",
    },
    "sftkf": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["FRLAKE"],
        "standard_name": "lake_area_fraction",
    },
    "sftlf": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["FRLAND"],
        "standard_name": "land_area_fraction",
    },
    "sftof": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["FROCEAN"],
        "standard_name": "sea_area_fraction",
    },
    "shg": {
        "esdt_dir": "M2C0NXASM.5.12.4",
        "collection": "const_2d_asm_Nx",
        "var_name": ["SHG"],
        "standard_name": "isotropic_stdv_of_gravity_wave_drag_topography",
    },
    "sic": {
        "esdt_dir": "M2T1NXOCN.5.12.4",
        "collection": "tavg1_2d_ocn_Nx",
        "var_name": ["FRSEAICE"],
        "standard_name": "sea_ice_area_fraction",
    },
    "tas": {
        "esdt_dir": "M2T1NXSLV.5.12.4",
        "collection": "tavg1_2d_slv_Nx",
        "var_name": ["T2M"],
        "standard_name": "air_temperature",
    },
    "tasmax": {
        "esdt_dir": "M2SDNXSLV.5.12.4",
        "collection": "statD_2d_slv_Nx",
        "var_name": ["T2MMAX"],
        "standard_name": "air_temperature",
    },
    "tasmin": {
        "esdt_dir": "M2SDNXSLV.5.12.4",
        "collection": "statD_2d_slv_Nx",
        "var_name": ["T2MMIN"],
        "standard_name": "air_temperature",
    },
    "uas": {
        "esdt_dir": "M2I1NXASM.5.12.4",
        "collection": "inst1_2d_asm_Nx",
        "var_name": ["U10M"],
        "standard_name": "eastward_wind",
    },
    "vas": {
        "esdt_dir": "M2I1NXASM.5.12.4",
        "collection": "inst1_2d_asm_Nx",
        "var_name": ["V10M"],
        "standard_name": "northward_wind",
    },
}


VARIABLE_MAP = {
    "time": "UTC_time",
    "SWGDN": "ghi",
    "SWGDNCLR": "ghi_clear",
    "ALBEDO": "albedo",
    "LWGNT": "longwave_net",
    "LWGEM": "longwave_emitted",
    "LWGAB": "longwave_absorbed",
    "CLDTOT": "cloud_fraction",
    "TAUTOT": "cloud_optical_thickness",
    "T2M": "temp_air",
    "T2MDEW": "temp_dew",
    "T2MMAX": "temp_air_max",
    "T2MMIN": "temp_air_min",
    "TQV": "water_vapour",
    "TO3": "ozone",
    "PS": "pressure",
    "RH": "relative_humidity",
    "QV2M": "specific_humidity",
    "TOTSCATAU": "aod_scattering",
    "TOTEXTTAU": "aod550",
    "TOTANGSTR": "angstrom_exponent",
    "EVAP": "evaporation_rate",
    "PRECTOT": "precipitation_rate",
    "PRECCON": "convective_precipitation_rate",
    "PRECTOTCORR": "precipitation_corrected_rate",
    "TPRECMAX": "precipitation_max_rate",
    "PRECSNO": "snowfall_rate",
    "U10M": "wind_u_10m",
    "V10M": "wind_v_10m",
    "FRSEAICE": "sea_ice_fraction",
    "PHIS": "surface_geopotential",
}


POINT_TEMPERATURE_COLUMNS = ("temp_air", "temp_dew", "temp_air_max", "temp_air_min")
_POINT_WATER_FLUX_COLUMNS = (
    "evaporation_rate", "precipitation_rate", "convective_precipitation_rate",
    "precipitation_corrected_rate", "precipitation_max_rate", "snowfall_rate",
)
_UNIT_CONVERSIONS = {
    **dict.fromkeys(POINT_TEMPERATURE_COLUMNS, ({"k", "kelvin"}, {"degc", "°c", "celsius"}, "degC", 1.0, -273.15)),
    "pressure": ({"pa", "pascal", "pascals"}, {"hpa", "mb", "mbar", "millibar", "millibars"}, "hPa", 0.01, 0.0),
    "relative_humidity": ({"1", "fraction", "dimensionless", "unitless", "0-1"}, {"%", "percent"}, "%", 100.0, 0.0),
    **dict.fromkeys(_POINT_WATER_FLUX_COLUMNS, ({"kgm-2s-1", "kg/m2/s"}, {"mm/h", "mmh-1"}, "mm/h", 3600.0, 0.0)),
}

# The radiation product supplies an hourly mean labeled at the interval center.
POINT_RADIATION_TIMING = {
    "timestamp_label": "interval_center",
    "integration_seconds": 3600,
}

POINT_UNSUPPORTED_COLLECTIONS = {"asm", "phis", "sftgif", "sftkf", "sftlf", "sftof", "shg", "hur"}


# =============================================================================
# Common helpers
# =============================================================================

def list_merra2_collections():
    """Return a copy of the built-in MERRA2 collection registry."""
    return {k: dict(v) for k, v in var_list.items()}


def _normalize_collection_names(collection_names):
    if isinstance(collection_names, str):
        return [collection_names]
    if isinstance(collection_names, (list, tuple)):
        return list(collection_names)
    raise TypeError("collection_names must be a string or a list of strings.")


def _get_auth(auth=None, username=None, password=None):
    if auth is not None:
        return auth
    if username is None or password is None:
        raise ValueError("username and password must be provided, or pass auth={'uid': ..., 'password': ...}.")
    return {"uid": username, "password": password}


def _is_date_only(value):
    if isinstance(value, str):
        value = value.strip()
        return bool(re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value)) or bool(re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", value))
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return True
    return False


def _to_utc_naive_time(time_value, end_of_day=False):
    time_value_raw = time_value
    time_value = pd.to_datetime(time_value)

    if time_value.tzinfo is not None:
        time_value = time_value.tz_convert("UTC").tz_localize(None)

    if _is_date_only(time_value_raw):
        if end_of_day:
            time_value = time_value.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            time_value = time_value.replace(hour=0, minute=0, second=0, microsecond=0)

    return time_value


def _resolve_date_range(start=None, end=None, initial_year=None, final_year=None, initial_month=1, final_month=12, initial_day=1, final_day=None):
    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("start and end must be provided together.")

        start_time = _to_utc_naive_time(start, end_of_day=False)
        end_time = _to_utc_naive_time(end, end_of_day=True)

        if start_time > end_time:
            raise ValueError("start must be earlier than or equal to end.")

        return start_time, end_time

    if initial_year is None or final_year is None:
        raise ValueError("Either start/end or initial_year/final_year must be provided.")
    if final_day is None:
        raise ValueError("final_day must be provided.")

    start_time = pd.Timestamp(year=int(initial_year), month=int(initial_month), day=int(initial_day), hour=0, minute=0, second=0)
    end_time = pd.Timestamp(year=int(final_year), month=int(final_month), day=int(final_day), hour=23, minute=59, second=59)

    if start_time > end_time:
        raise ValueError("start date must be earlier than or equal to end date.")

    return start_time, end_time


def _split_year_ranges(start_time, end_time):
    current_start = start_time
    while current_start.year < end_time.year:
        current_end = pd.Timestamp(year=current_start.year, month=12, day=31, hour=23, minute=59, second=59)
        yield current_start, current_end
        current_start = pd.Timestamp(year=current_start.year + 1, month=1, day=1, hour=0, minute=0, second=0)
    yield current_start, end_time


def _resolve_variables_for_collection(collection_name, variables=None):
    if collection_name not in var_list:
        raise ValueError(f"Unsupported collection name: {collection_name}. Available collections are: {list(var_list.keys())}")

    default_variables = var_list[collection_name]["var_name"]

    if variables is None:
        selected = default_variables
    elif isinstance(variables, str):
        selected = [variables]
    elif isinstance(variables, list):
        selected = variables
    else:
        raise TypeError("variables must be None, a variable name string, a list, or a dict.")

    invalid = [v for v in selected if v not in default_variables]
    if invalid:
        raise ValueError(f"Invalid variables for '{collection_name}': {invalid}. Available variables are: {default_variables}")

    return selected


def _collection_variables(collection_names, variables=None, collection_name=None):
    if isinstance(variables, dict):
        return variables.get(collection_name)
    if isinstance(variables, (list, str)):
        if len(collection_names) > 1:
            raise ValueError("When requesting multiple collections, variables must be a dict, for example {'rad': ['SWGDN'], 'tas': ['T2M']}.")
        return variables
    return None


def _build_merra2_var_dicts(collection_names, variables=None):
    merra2_var_dicts = {}
    for collection_name in collection_names:
        selected_variables = _resolve_variables_for_collection(collection_name, _collection_variables(collection_names, variables, collection_name))
        item = dict(var_list[collection_name])
        item["var_name"] = selected_variables
        merra2_var_dicts[collection_name] = item
    return merra2_var_dicts


def _safe_decode_response(response):
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.text


def _normalize_units(name, units):
    native, converted, target, factor, offset = _UNIT_CONVERSIONS[name]
    token = _netcdf_unit_key(units)
    if token in converted:
        return target, 1.0, 0.0
    if not token or token in native:
        return target, factor, offset
    raise ValueError(f"Unexpected MERRA-2 unit for {name}: {units!r}.")


def _standardize_area(datasets, map_variables):
    ds = datasets[0]
    names = {}
    for original, data in ds.data_vars.items():
        name = VARIABLE_MAP.get(original, original)
        missing = data.attrs.get("fmissing_value")
        if missing is not None:
            for sentinel in np.asarray(missing).reshape(-1):
                data = data.where(data != sentinel)
            data.attrs.pop("fmissing_value", None)
            ds[original] = data
        if name in _UNIT_CONVERSIONS:
            target, factor, offset = _normalize_units(name, data.attrs.get("units"))
            if factor != 1.0 or offset:
                ds[original] = (data * factor + offset).assign_attrs(data.attrs)
                ds[original].attrs.pop("valid_min", None)
                ds[original].attrs.pop("valid_max", None)
                ds[original].attrs.pop("valid_range", None)
            ds[original].attrs["units"] = target
        if name in {"ghi", "ghi_clear", "longwave_net", "longwave_emitted", "longwave_absorbed"}:
            if _netcdf_unit_key(data.attrs.get("units")) in {"wm-2", "w/m2"}:
                ds[original].attrs["units"] = "W/m2"
            ds[original].attrs.update(POINT_RADIATION_TIMING)
        elif name == "relative_humidity":
            ds[original].attrs.update(timestamp_label="interval_center", integration_seconds=10800)
        if map_variables and name != original:
            names[original] = name
    if map_variables and "precipitation_rate" in ds and "precipitation" not in ds:
        ds["precipitation"] = ds["precipitation_rate"].assign_attrs(units="mm", long_name="Hourly precipitation amount")
    elif map_variables and "PRECTOT" in ds and "precipitation" not in ds:
        ds["precipitation"] = ds["PRECTOT"].assign_attrs(units="mm", long_name="Hourly precipitation amount")
    ds = ds.rename(names)
    ds.attrs["irradpy2_units"] = "standardized"
    return ds, {name: value.attrs["units"] for name, value in ds.data_vars.items() if "units" in value.attrs}


def _rename_dataset_with_variable_map(ds):
    rename_dict = {name: VARIABLE_MAP[name] for name in ds.variables if name in VARIABLE_MAP}
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds


# =============================================================================
# Area downloader: OPeNDAP / NetCDF
# =============================================================================

class DownloadManager(object):
    """
    Low-level authenticated downloader for Earthdata-protected MERRA2 links.
    This class is kept close to the original implementation and is mainly used by SocketManager.
    """

    __AUTHENTICATION_URL = "https://urs.earthdata.nasa.gov/oauth/authorize"
    __username = ""
    __password = ""
    __download_url = ""
    __download_path = ""
    _authenticated_session = None

    def __init__(self, username="", password="", link=None, download_path="download"):
        self.set_username_and_password(username, password)
        self.download_url = link
        self.download_path = download_path

    @property
    def download_url(self):
        return self.__download_url

    @download_url.setter
    def download_url(self, link):
        if link is None:
            self.__download_url = ""
        else:
            try:
                self.get_filename(link)
            except Exception:
                raise ValueError(f"The URL seems to not have the right structure: {link}")
            self.__download_url = link

    @property
    def download_path(self):
        return self.__download_path

    @download_path.setter
    def download_path(self, file_path):
        self.__download_path = file_path

    def set_username_and_password(self, username, password):
        self.__username = username
        self.__password = password

    def _mp_download_wrapper(self, url_item, filename=None):
        file_path = os.path.join(self.download_path, filename or self.get_filename(url_item))
        self.__download_and_save_file(url_item, file_path)

    def start_download(self, filename=None):
        if self._authenticated_session is None:
            self._authenticated_session = self.__create_authenticated_sesseion()
        os.makedirs(self.download_path, exist_ok=True)
        self._mp_download_wrapper(self.download_url, filename=filename)

    @staticmethod
    def get_filename(url):
        reg_exp = r"(\w+)(\.\w+)+(?!.*(\w+)(\.\w+)+)"
        match = re.search(reg_exp, url)
        if match is None:
            raise ValueError(f"Cannot extract filename from URL: {url}")
        return match.group(0)

    def __download_and_save_file(self, url, file_path):
        r = self._authenticated_session.get(url, stream=True, allow_redirects=True, timeout=(30, 300))
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise RuntimeError("Downloaded content is HTML instead of NetCDF file. Authentication may have failed or the URL redirected to a login page.")

        first_chunk = None
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_chunk is None:
                        first_chunk = chunk
                    f.write(chunk)

        if first_chunk is None:
            raise RuntimeError("Downloaded file is empty.")

        first_chunk_lower = first_chunk[:200].lower()
        if b"<html" in first_chunk_lower or b"<!doctype html" in first_chunk_lower or b"<?xml" in first_chunk_lower:
            raise RuntimeError("Downloaded content looks like HTML/XML instead of a NetCDF file.")

        return r.status_code

    def __create_authenticated_sesseion(self):
        s = requests.Session()
        s.headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/40.0.2214.85 Safari/537.36")}
        s.auth = (self.__username, self.__password)
        s.cookies = self.__authorize_cookies_with_urllib()
        return s

    def __authorize_cookies_with_urllib(self):
        top_level_url = "https://urs.earthdata.nasa.gov"
        p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        p.add_password(None, top_level_url, self.__username, self.__password)

        auth_handler = urllib.request.HTTPBasicAuthHandler(p)
        auth_cookie_jar = cookiejar.CookieJar()
        cookie_jar = urllib.request.HTTPCookieProcessor(auth_cookie_jar)
        opener = urllib.request.build_opener(auth_handler, cookie_jar)

        with opener.open(self.download_url, timeout=30):
            pass

        return auth_cookie_jar


class SocketManager:
    """
    High-level MERRA2 area download manager.
    Workflow: coordinate translation -> Cloud CMR discovery -> download -> optional merge.
    """

    deff4 = netCDF4.default_fillvals["f4"]

    global_retry = True
    lat_coords = np.arange(0, 361, dtype=int)
    lon_coords = np.arange(0, 576, dtype=int)

    def __init__(self):
        self._cloud_manager = None

    def generate_url_params(self, parameter, time_para, lat_para, lon_para, level_para=""):
        slices = time_para + level_para + lat_para + lon_para
        coordinates = ["time" + time_para, "lat" + lat_para, "lon" + lon_para]
        if level_para:
            coordinates.append("lev" + level_para)
        return ",".join([*(name + slices for name in parameter), *coordinates])

    def translate_lat_to_geos5_native(self, latitude):
        return ((latitude + 90) / 0.5)

    def translate_lon_to_geos5_native(self, longitude):
        return ((longitude + 180) / 0.625)

    def find_closest_coordinate(self, calc_coord, coord_array):
        index = np.abs(coord_array - calc_coord).argmin()
        return coord_array[index]

    def build_remote_url(self, merra2_collection, date):
        """Legacy URL builder; live downloads use _build_cloud_area_url instead."""
        if merra2_collection["collection"].startswith("const"):
            return ("https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/{db_name}/1980/").format(db_name=merra2_collection["esdt_dir"])
        return ("https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{db_name}/{date:%Y}/{date:%m}/").format(db_name=merra2_collection["esdt_dir"], date=date)

    def reconstruct_filename(self, name, params):
        return "{name}?{params}".format(name=name, params=params)

    def build_remote_filename(self, merra2_collection, date, params):
        merra_stream = "400"
        file_extension = "nc"
        date_string = "{date:%Y%m%d}".format(date=date)

        if date.year < 1992:
            merra_stream = "100"
        elif date.year < 2001:
            merra_stream = "200"
        elif date.year < 2011:
            merra_stream = "300"

        if merra2_collection["collection"].startswith("const"):
            merra_stream = "101"
            date_string = "00000000"
            file_extension = "nc4"

        if params is not None:
            return "MERRA2_{stream}.{abbrv}.{date}.nc4.{ext}?{params}".format(stream=merra_stream, abbrv=merra2_collection["collection"], date=date_string, params=params, ext=file_extension)
        return "MERRA2_{stream}.{abbrv}.{date}.nc4.{ext}".format(stream=merra_stream, abbrv=merra2_collection["collection"], date=date_string, ext=file_extension)

    def _build_cloud_area_url(self, merra2_collection, date, params, auth):
        """Discover the daily Cloud granule and request a NetCDF4 DAP4 subset."""
        if self._cloud_manager is None:
            self._cloud_manager = MERRA2CSVManager(
                username=auth["uid"], password=auth["password"]
            )

        dataset = merra2_collection["esdt_dir"]
        collection = merra2_collection["collection"]
        if collection.startswith("const"):
            short_name, version = self._cloud_manager._dataset_identity(dataset)
            urls = get_cmr_urls(
                short_name=short_name, version=version,
                session=self._cloud_manager.session, limit=10,
            )
            date_string = "00000000"
        else:
            day = pd.Timestamp(date)
            urls = self._cloud_manager._get_opendap_urls(
                dataset, day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            )
            date_string = date.strftime("%Y%m%d")

        # CMR temporal searches can also return an adjacent daily granule.
        marker = f".{collection}.{date_string}.nc4"
        urls = sorted({url for url in urls if marker in url})
        if len(urls) != 1:
            raise RuntimeError(
                f"Expected one Cloud OPeNDAP granule for {dataset} on {date}, "
                f"found {len(urls)}."
            )

        # Cloud subsetting requires DAP4; old DAP2 queries do not subset correctly.
        url = urls[0].split("?", 1)[0] + ".dap.nc4"
        if params:
            selections = ["/" + item for item in params.split(",") if item]
            url += "?dap4.ce=" + ";".join(selections)
        return url

    def download_merra2_nc(self, date, merra2_collection, output_directory, params, auth):
        # Preserve the original local filenames used by the merge/resume logic.
        filename = self.build_remote_filename(merra2_collection, date, None)
        file_path = os.path.join(output_directory, filename)

        if os.path.exists(file_path):
            logging.info("- Detected duplicated file from date " + str(date) + ", deleting existing file before downloading...")
            os.remove(file_path)

        start_time = time.time()
        logging.info("* File from date " + str(date) + " begin to download")
        retry = True
        limit = 0
        passwd = False

        while retry and limit < 1:
            limit += 1
            log = "{dataset}/{date} # {limit} Time Retry #".format(
                dataset=merra2_collection["esdt_dir"], date=date, limit=limit
            )
            try:
                url = self._build_cloud_area_url(merra2_collection, date, params, auth)
                download_manager = DownloadManager(
                    username=auth["uid"], password=auth["password"],
                    link=url, download_path=output_directory,
                )
                download_manager._authenticated_session = self._cloud_manager.session
                download_manager.start_download(filename=filename)
                retry = False
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logging.error(log + ": requested URL is not available.")
                elif e.code == 401:
                    logging.error(log + ": username or password is not correct.")
                    passwd = True
                elif e.code == 302:
                    logging.error(log + ": HTTP Error 302: redirect loop, authentication/session may be unstable.")
                else:
                    logging.error(log + ": " + str(e))
            except IOError as e:
                logging.error(log + ": " + str(e))
            except IndexError:
                logging.error(log + ": unknown download URL.")
            except TypeError:
                logging.error(log + ": file chunk type mismatch.")
            except Exception as e:
                logging.error(log + ": " + str(e))

        if retry:
            logging.critical("* File from date " + str(date) + " failed download")
            if not passwd:
                self.global_retry = True
            if os.path.exists(file_path):
                os.remove(file_path)
            return False

        logging.info("* File from date " + str(date) + " finished download with %3.2f seconds" % (time.time() - start_time))
        return True

    def iter_days(self, first: datetime.date, last: datetime.date):
        current = first
        while current <= last:
            yield current
            current += datetime.timedelta(1)

    def iter_months(self, first: datetime.date, last: datetime.date):
        ym_start = 12 * first.year + first.month - 1
        ym_end = 12 * last.year + last.month - 1
        for ym in range(ym_start, ym_end + 1):
            y, m = divmod(ym, 12)
            yield "{:0>4d}-{:0>2d}".format(y, m + 1)

    def iter_years(self, first: datetime.date, last: datetime.date):
        for y in range(first.year, last.year + 1):
            yield "{:0>4d}".format(y)

    def subdaily_universal_download(self, merra2_collection: dict, initial_year: int, final_year: int, initial_month: int = 1, final_month: int = 12, initial_day: int = 1, final_day: Optional[int] = None, params: Optional[str] = None, auth: dict = None, output_directory: Union[str, Path] = None, thread_num: int = 5, merge: str = "monthly"):
        if not isinstance(output_directory, Path):
            log_file = Path(output_directory)
        else:
            log_file = output_directory

        log_file = os.path.join(log_file, "index.npy")
        if os.path.exists(log_file):
            log = np.load(log_file).tolist()
        else:
            log = []

        for i, n in enumerate(log):
            if n.endswith("False"):
                log[i] = n.replace("False", merge)
            if n.endswith("True"):
                log[i] = n.replace("True", merge)

        logging.info("---- Begin Analysing Directory ----")
        for name in os.listdir(output_directory):
            if name.endswith(".nc4.nc") and merra2_collection["collection"] in name and self.reconstruct_filename(name, params) + str(merge) not in log:
                intact = False
                try:
                    with xr.open_dataset(os.path.join(output_directory, name), engine="netcdf4") as ds:
                        intact = bool(ds.lat.size and ds.lon.size and set(merra2_collection["var_name"]) <= set(ds.data_vars))
                except Exception:
                    intact = False

                if intact:
                    log.append(self.reconstruct_filename(name, params) + str(merge))
                    logging.info("- Found previous intact file " + self.reconstruct_filename(name, params))
                    np.save(log_file, np.array(log))
                else:
                    logging.error("* Found previous corrupted file " + self.reconstruct_filename(name, params) + ", scheduled for redownload")
        logging.info("----  End Analysing Directory  ----")

        dates = []
        temp_log = []
        for date in self.iter_days(datetime.date(initial_year, initial_month, initial_day), datetime.date(final_year, final_month, final_day)):
            file_key = self.build_remote_filename(merra2_collection, date, params) + str(merge)
            if file_key in temp_log or file_key in log:
                logging.info("Skipping existing file " + self.build_remote_filename(merra2_collection, date, params) + " from " + merra2_collection["esdt_dir"])
                continue
            temp_log.append(file_key)
            logging.info("Preparing new file " + self.build_remote_filename(merra2_collection, date, params) + " from " + merra2_collection["esdt_dir"])
            dates.append(date)

        if len(dates) > 0:
            logging.info("----  Begin  Download  ----")
            time_start = None

            for start in range(0, len(dates), thread_num):
                end = min(start + thread_num, len(dates))
                if time_start is not None:
                    logging.info("~ %d Files Downloaded In %3.2f Seconds, %d Files Left, Expected %3.2f Minutes Remaining..." % (thread_num, time.time() - time_start, len(dates) - start, (len(dates) - start) / thread_num * (time.time() - time_start) / 60))
                time_start = time.time()

                if thread_num == 1:
                    rel = []
                    for current_date in dates[start:end]:
                        result = self.download_merra2_nc(current_date, merra2_collection=merra2_collection, output_directory=output_directory, params=params, auth=auth)
                        rel.append(result)
                else:
                    pool = multiprocessing.Pool(thread_num)
                    rel = pool.map_async(partial(self.download_merra2_nc, merra2_collection=merra2_collection, output_directory=output_directory, params=params, auth=auth), dates[start:end]).get(0.5 * _thread.TIMEOUT_MAX)
                    pool.close()
                    pool.join()

                for i, success in enumerate(rel):
                    if success:
                        name = self.build_remote_filename(merra2_collection, dates[start:end][i], None)
                        intact = False
                        try:
                            with xr.open_dataset(os.path.join(output_directory, name), engine="netcdf4") as ds:
                                intact = bool(ds.lat.size and ds.lon.size and set(merra2_collection["var_name"]) <= set(ds.data_vars))
                        except Exception:
                            intact = False

                        if intact:
                            logging.info("% New file from date " + str(dates[start:end][i]) + " confirmed")
                            log.append(self.build_remote_filename(merra2_collection, dates[start:end][i], params) + str(merge))
                        else:
                            logging.critical("% New file from date " + str(dates[start:end][i]) + " integrity check failed")
                            self.global_retry = True
                    else:
                        logging.critical("% New file from date " + str(dates[start:end][i]) + " failed")
                        self.global_retry = True

                np.save(log_file, np.array(log))
            logging.info("---- Download Finished ----")

    def merge_variables_perday(self, path_data: Union[str, Path], collection_names: List[str], final_year: int, final_month: int, final_day: int, date: datetime.date, map_variables: bool = True):
        if not isinstance(path_data, Path):
            path_data = Path(path_data)

        delete_set = []
        time_start = time.time()
        search_str = "*{0}.nc4.nc".format(str(date).replace("-", ""))
        nc_files = [str(f) for f in path_data.rglob(search_str)]
        nc_files.sort()

        if len(nc_files) == 0:
            logging.info("* Skipping data in {0}".format(date))
            return delete_set

        logging.info("* Processing data in {0}...".format(date))
        final_ds = xr.Dataset()
        collections = []

        for name in nc_files:
            logging.info("% Merging data for {0}".format(os.path.split(name)[1]))
            var = os.path.split(name)[1].split("_")[3]
            if var not in collections:
                collections.append(var)
            with xr.open_dataset(name, engine="netcdf4") as source_ds:
                remote_ds = source_ds.load()
            try:
                final_ds = xr.merge([final_ds, remote_ds], join="outer")
            except ValueError:
                remote_ds.close()
                raise RuntimeError("Corrupted file detected: {0}".format(os.path.split(name)[1]))
            remote_ds.close()

        if len(collections) != len(collection_names):
            raise RuntimeError("Partially downloaded collection on date {0}.".format(date))

        collections.sort()
        filename = os.path.join(path_data, "{0}_merra2_reanalysis_{1}.nc".format("-".join(collections), date))

        if map_variables:
            final_ds = _rename_dataset_with_variable_map(final_ds)

        encoding = {v: {"zlib": True, "complevel": 4} for v in final_ds.data_vars}
        logging.info("- Saving data for {0}".format(date))
        final_ds.to_netcdf(filename, encoding=encoding)
        final_ds.close()

        for name in nc_files:
            delete_set.append(name)

        logging.info("* File from date {} finished daily merge in {:.2f} seconds".format(date, time.time() - time_start))
        return delete_set

    def merge_variables_permonth(self, path_data: Union[str, Path], collection_names: List[str], final_year: int, final_month: int, final_day: int, date: str, map_variables: bool = True):
        if not isinstance(path_data, Path):
            path_data = Path(path_data)

        time_start = time.time()
        delete_set = []
        search_str = "*{0}-*.nc".format(date)
        nc_files = [str(f) for f in path_data.rglob(search_str)]
        nc_files.sort()

        if len(nc_files) == 0:
            logging.info("* Skipping data in {0}".format(date))
            return delete_set

        logging.info("* Processing data in {0}".format(date))
        with xr.open_dataset(nc_files[0], engine="netcdf4") as source_ds:
            final_ds = source_ds.load()
        time_dimension = "UTC_time" if "UTC_time" in final_ds.dims else "time"
        collections = os.path.split(nc_files[0])[1].split("_")[0]

        for name in nc_files[1:]:
            logging.info("% Merging data for {0}".format(os.path.split(name)[1]))
            var = os.path.split(name)[1].split("_")[0]
            if collections is None or len(collections) > len(var):
                collections = var
            with xr.open_dataset(name, engine="netcdf4") as source_ds:
                remote_ds = source_ds.load()
            final_ds = xr.concat([final_ds, remote_ds], dim=time_dimension)

        filename = os.path.join(path_data, "{0}_merra2_reanalysis_{1}.nc".format(collections, date))

        if map_variables:
            final_ds = _rename_dataset_with_variable_map(final_ds)

        encoding = {v: {"zlib": True, "complevel": 4} for v in final_ds.data_vars}
        logging.info("- Saving data for {0}".format(date))
        final_ds.to_netcdf(filename, encoding=encoding)
        final_ds.close()

        for name in nc_files:
            delete_set.append(name)

        logging.info("* File from date {} finished monthly merge in {:.2f} seconds".format(date, time.time() - time_start))
        return delete_set

    def merge_variables_peryear(self, path_data: Union[str, Path], collection_names: List[str], final_year: int, final_month: int, final_day: int, date: str, map_variables: bool = True):
        if not isinstance(path_data, Path):
            path_data = Path(path_data)

        time_start = time.time()
        delete_set = []
        search_str = "*{0}-*.nc".format(date)
        nc_files = [str(f) for f in path_data.rglob(search_str)]
        nc_files.sort()

        if len(nc_files) == 0:
            logging.info("* Skipping data in {0}".format(date))
            return delete_set

        logging.info("* Processing data in {0}".format(date))
        with xr.open_dataset(nc_files[0], engine="netcdf4") as source_ds:
            final_ds = source_ds.load()
        time_dimension = "UTC_time" if "UTC_time" in final_ds.dims else "time"
        collections = os.path.split(nc_files[0])[1].split("_")[0]

        for name in nc_files[1:]:
            logging.info("% Merging data for {0}".format(os.path.split(name)[1]))
            var = os.path.split(name)[1].split("_")[0]
            if collections is None or len(collections) > len(var):
                collections = var
            with xr.open_dataset(name, engine="netcdf4") as source_ds:
                remote_ds = source_ds.load()
            final_ds = xr.concat([final_ds, remote_ds], dim=time_dimension)

        filename = os.path.join(path_data, "{0}_merra2_reanalysis_{1}.nc".format(collections, date))

        if map_variables:
            final_ds = _rename_dataset_with_variable_map(final_ds)

        encoding = {v: {"zlib": True, "complevel": 4} for v in final_ds.data_vars}
        logging.info("- Saving data for {0}".format(date))
        final_ds.to_netcdf(filename, encoding=encoding)
        final_ds.close()

        for name in nc_files:
            delete_set.append(name)

        logging.info("* File from date {} finished yearly merge in {:.2f} seconds".format(date, time.time() - time_start))
        return delete_set

    def daily_download_and_convert(self, collection_names: List[str], initial_year: int, final_year: Optional[int] = datetime.datetime.now().year, initial_month: int = 1, final_month: Optional[int] = datetime.datetime.now().month, initial_day: int = 1, final_day: Optional[int] = datetime.datetime.now().day, lat_1: Optional[float] = -90, lon_1: Optional[float] = -180, lat_2: Optional[float] = 90, lon_2: Optional[float] = 180, merra2_var_dicts: Optional[Dict[str, dict]] = None, output_dir: Union[str, Path] = None, auth: dict = None, merge_timelapse: str = "monthly", thread_num: Optional[int] = 5, map_variables: bool = True):
        if lat_1 > lat_2 or lon_1 > lon_2:
            raise RuntimeError("Illegal data area selected.")
        if merge_timelapse not in ["none", "daily", "monthly", "yearly"]:
            raise RuntimeError("Illegal merge timelapse given, select from ['none', 'daily', 'monthly', 'yearly'].")

        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

        max_retry = 3
        retry_count = 0
        self.global_retry = True

        if output_dir is None:
            output_dir = Path.cwd()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        temp_dir_name = "{0}-{1}-{2}~{3}-{4}-{5} {6} [{7},{8}]~[{9},{10}]".format(initial_year, initial_month, initial_day, final_year, final_month, final_day, "-".join(collection_names), lat_1, lon_1, lat_2, lon_2)
        temp_dir_download = os.path.join(output_dir, temp_dir_name)

        while self.global_retry and retry_count < max_retry:
            retry_count += 1
            self.global_retry = False

            logging.info("Downloading data from {0}-{1}-{2} to {3}-{4}-{5}...".format(initial_year, initial_month, initial_day, final_year, final_month, final_day))

            try:
                os.mkdir(temp_dir_download)
            except OSError as err:
                if err.errno != errno.EEXIST:
                    raise err
                logging.info("Request already exists in download directory, adding more files...")

            for collection_name in collection_names:
                if merra2_var_dicts is None:
                    merra2_var_dict = var_list[collection_name]
                else:
                    merra2_var_dict = merra2_var_dicts[collection_name]

                lat_coord_1 = self.translate_lat_to_geos5_native(lat_1)
                lon_coord_1 = self.translate_lon_to_geos5_native(lon_1)
                lat_coord_2 = self.translate_lat_to_geos5_native(lat_2)
                lon_coord_2 = self.translate_lon_to_geos5_native(lon_2)

                lat_co_1_closest = self.find_closest_coordinate(lat_coord_1, self.lat_coords)
                lon_co_1_closest = self.find_closest_coordinate(lon_coord_1, self.lon_coords)
                lat_co_2_closest = self.find_closest_coordinate(lat_coord_2, self.lat_coords)
                lon_co_2_closest = self.find_closest_coordinate(lon_coord_2, self.lon_coords)

                requested_lat = "[{lat_1}:{lat_2}]".format(lat_1=lat_co_1_closest, lat_2=lat_co_2_closest)
                requested_lon = "[{lon_1}:{lon_2}]".format(lon_1=lon_co_1_closest, lon_2=lon_co_2_closest)

                if isinstance(merra2_var_dict["var_name"], list):
                    requested_params = merra2_var_dict["var_name"]
                else:
                    requested_params = [merra2_var_dict["var_name"]]

                product = merra2_var_dict["collection"]
                if product.startswith(("const", "statD")):
                    requested_time = "[0:0]"
                elif product == "tavg3_3d_cld_Np":
                    requested_time = "[0:7]"
                else:
                    requested_time = "[0:23]"

                requested_levels = "[0:41]" if product == "tavg3_3d_cld_Np" else ""
                parameter = self.generate_url_params(
                    requested_params, requested_time, requested_lat, requested_lon, requested_levels
                )

                self.subdaily_universal_download(merra2_var_dict, initial_year, final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day, output_directory=temp_dir_download, auth=auth, params=parameter, thread_num=thread_num, merge=merge_timelapse)

            if self.global_retry:
                logging.error("Download integrity check failed, skip merging and retry later.")
                continue

            merge_collection_names = []
            for collection_name in collection_names:
                if merra2_var_dicts is None:
                    merra2_var_dict = var_list[collection_name]
                else:
                    merra2_var_dict = merra2_var_dicts[collection_name]
                if not merra2_var_dict["collection"].startswith("const"):
                    merge_collection_names.append(collection_name)

            try:
                if merge_timelapse in ["daily", "monthly", "yearly"]:
                    logging.info("---- Begin Merging In Daily Variables ----")
                    for date in self.iter_days(datetime.date(initial_year, initial_month, initial_day), datetime.date(final_year, final_month, final_day)):
                        delete_set = self.merge_variables_perday(temp_dir_download, merge_collection_names, final_year, final_month, final_day, date, map_variables=map_variables)
                        for name in delete_set:
                            while True:
                                try:
                                    os.remove(name)
                                    break
                                except OSError as err:
                                    logging.error("OSError: {0} {1}, retrying...".format(os.path.split(name)[1], err))
                                    time.sleep(20)
                    logging.info("---- Finish Merging In Daily Variables ----")

                if merge_timelapse in ["monthly", "yearly"]:
                    logging.info("---- Begin Merging In Monthly Variables ----")
                    for date in self.iter_months(datetime.date(initial_year, initial_month, initial_day), datetime.date(final_year, final_month, final_day)):
                        delete_set = self.merge_variables_permonth(temp_dir_download, merge_collection_names, final_year, final_month, final_day, date, map_variables=map_variables)
                        for name in delete_set:
                            while True:
                                try:
                                    os.remove(name)
                                    break
                                except OSError as err:
                                    logging.error("OSError: {0} {1}, retrying...".format(os.path.split(name)[1], err))
                                    time.sleep(20)
                    logging.info("---- Finish Merging In Monthly Variables ----")

                if merge_timelapse == "yearly":
                    logging.info("---- Begin Merging In Yearly Variables ----")
                    for date in self.iter_years(datetime.date(initial_year, initial_month, initial_day), datetime.date(final_year, final_month, final_day)):
                        delete_set = self.merge_variables_peryear(temp_dir_download, merge_collection_names, final_year, final_month, final_day, date, map_variables=map_variables)
                        for name in delete_set:
                            while True:
                                try:
                                    os.remove(name)
                                    break
                                except OSError as err:
                                    logging.error("OSError: {0} {1}, retrying...".format(os.path.split(name)[1], err))
                                    time.sleep(20)
                    logging.info("---- Finish Merging In Yearly Variables ----")

            except RuntimeError:
                self.global_retry = True

            if self.global_retry:
                logging.error("Requested data partially downloaded, retry downloading...")

        if self.global_retry:
            raise RuntimeError("Download failed after maximum retry attempts.")

        return temp_dir_download


# =============================================================================
# Point downloader: Earthdata Cloud OPeNDAP / CSV
# =============================================================================

_EARTHDATA_LOGIN_LOCK = threading.Lock()


class MERRA2CSVManager:
    """
    Single-point MERRA2 downloader backed by Earthdata Cloud OPeNDAP.

    The public point-mode input and output stay unchanged. Internally, the
    manager discovers daily granules through CMR, asks the OPeNDAP server for
    one MERRA-2 grid cell, and converts the result to the original DataFrame
    and CSV shape.
    """

    def __init__(self, username: str, password: str, timeout: Tuple[int, int] = (30, 300)):
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session = self._create_authenticated_session()

    def _create_authenticated_session(self):
        if earthaccess is None or get_cmr_urls is None or create_session is None:
            raise ImportError(
                "MERRA2 point mode requires current earthaccess and pydap packages. "
                "Install them with: python -m pip install -U earthaccess \"pydap>=3.5.9\""
            )

        environment_keys = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN")

        with _EARTHDATA_LOGIN_LOCK:
            previous_values = {key: os.environ.get(key) for key in environment_keys}
            try:
                os.environ["EARTHDATA_USERNAME"] = self.username
                os.environ["EARTHDATA_PASSWORD"] = self.password
                os.environ.pop("EARTHDATA_TOKEN", None)

                earthdata_auth = earthaccess.Auth()
                earthdata_auth.login(strategy="environment")
                return create_session(session=earthdata_auth.get_session())
            except Exception as exc:
                raise RuntimeError(
                    "Earthdata Login failed. Check the username/password and confirm "
                    "that the GES DISC application is authorized in Earthdata Login."
                ) from exc
            finally:
                for key, value in previous_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    @staticmethod
    def _dataset_identity(dataset: str):
        parts = dataset.split(".", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MERRA2 dataset identifier: {dataset}")
        return parts[0], parts[1]

    @staticmethod
    def _nearest_grid_point(lat: float, lon: float):
        if not -90 <= lat <= 90:
            raise ValueError("lat must be between -90 and 90 degrees.")

        normalized_lon = ((float(lon) + 180.0) % 360.0) - 180.0
        latitude_index = int(np.argmin(np.abs(np.arange(-90.0, 90.5, 0.5) - float(lat))))
        longitude_index = int(np.argmin(np.abs(np.arange(-180.0, 180.0, 0.625) - normalized_lon)))
        matched_latitude = -90.0 + latitude_index * 0.5
        matched_longitude = -180.0 + longitude_index * 0.625

        return latitude_index, longitude_index, matched_latitude, matched_longitude

    def _get_opendap_urls(self, dataset: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
        short_name, version = self._dataset_identity(dataset)
        urls = []

        for sub_start, sub_end in _split_year_ranges(start_time, end_time):
            number_of_days = (sub_end.normalize() - sub_start.normalize()).days + 1
            limit = max(1, number_of_days + 2)
            logging.info(
                "Discovering Earthdata Cloud granules for %s from %s to %s",
                short_name,
                sub_start,
                sub_end,
            )
            discovered = get_cmr_urls(
                short_name=short_name,
                version=version,
                time_range=[sub_start.to_pydatetime(), sub_end.to_pydatetime()],
                session=self.session,
                limit=limit,
            )
            urls.extend(discovered)

        urls = sorted(dict.fromkeys(urls))
        if not urls:
            raise RuntimeError(
                f"No Earthdata Cloud OPeNDAP granules were found for {dataset} "
                f"between {start_time} and {end_time}."
            )

        return urls

    @staticmethod
    def _build_dap4_url(url: str, variables: List[str]):
        selections = ["/time", "/lat", "/lon"]
        selections.extend(f"/{variable}" for variable in variables)

        dap4_url = re.sub(r"^https://", "dap4://", url, count=1)

        return dap4_url + "?dap4.ce=" + ";".join(selections)

    @staticmethod
    def _normalize_dap4_dataset(dataset):
        """Use coordinate variables to normalize DAP4 root dimension names.

        Depending on the PyDAP/Xarray versions, a variable named ``lat``
        can have a dimension named ``/lat``. Coordinate variables may also
        arrive as data variables, or retain a leading slash themselves.
        Promote the actual time/latitude/longitude arrays to coordinates
        before swapping dimensions so point extraction keeps real UTC times.
        Data variables remain lazy; only coordinate indexes are constructed.
        """
        variable_renames = {}
        for name in dataset.variables:
            if not isinstance(name, str) or not name.startswith("/"):
                continue
            normalized_name = name.lstrip("/")
            # Only normalize root variables, not variables in nested groups.
            if not normalized_name or "/" in normalized_name:
                continue
            if normalized_name in dataset.variables:
                raise ValueError(
                    f"Ambiguous DAP4 variable names: {name!r} and {normalized_name!r}."
                )
            if normalized_name in variable_renames.values():
                raise ValueError(f"Duplicate DAP4 root variable: {normalized_name!r}.")
            variable_renames[name] = normalized_name

        if variable_renames:
            dataset = dataset.rename_vars(variable_renames)

        coordinates = ("time", "lat", "lon")
        missing_coordinates = [name for name in coordinates if name not in dataset.variables]
        if missing_coordinates:
            raise ValueError(f"MERRA2 coordinate variables are missing: {missing_coordinates}")

        # Existing indexes can still reference /lat, /lon or /time after a
        # dimension swap. Rebuild these indexes from the coordinate arrays.
        indexed_coordinates = [name for name in coordinates if name in dataset.xindexes]
        if indexed_coordinates:
            dataset = dataset.drop_indexes(indexed_coordinates)
        dataset = dataset.set_coords(list(coordinates))
        dimension_swaps = {}
        for name in coordinates:
            coordinate = dataset[name]
            if coordinate.ndim != 1:
                raise ValueError(f"MERRA2 coordinate {name!r} must be one-dimensional.")
            dimension = coordinate.dims[0]
            if dimension != name:
                if dimension in dimension_swaps:
                    raise ValueError(f"MERRA2 coordinates share dimension {dimension!r}.")
                dimension_swaps[dimension] = name

        if dimension_swaps:
            dataset = dataset.swap_dims(dimension_swaps)
        for name in coordinates:
            if name not in dataset.xindexes:
                dataset = dataset.set_xindex(name)
        return dataset

    @staticmethod
    def _replace_missing_values(data: pd.DataFrame, variable_attributes: dict):
        for variable, attributes in variable_attributes.items():
            if variable not in data.columns:
                continue
            for attribute_name in ("fmissing_value", "_FillValue", "missing_value"):
                missing_values = attributes.get(attribute_name)
                if missing_values is None:
                    continue
                for missing_value in np.asarray(missing_values).reshape(-1):
                    data[variable] = data[variable].mask(data[variable] == missing_value)
        return data

    def _read_granule_point(self, url: str, latitude_index: int, longitude_index: int, variables: List[str]):
        dap4_url = self._build_dap4_url(url, variables)

        try:
            with xr.open_dataset(
                dap4_url,
                engine="pydap",
                session=self.session,
                timeout=self.timeout,
            ) as remote_dataset:
                remote_dataset = self._normalize_dap4_dataset(remote_dataset)
                missing_variables = [variable for variable in variables if variable not in remote_dataset]
                if missing_variables:
                    raise RuntimeError(f"Variables missing from the MERRA2 granule: {missing_variables}")
                for variable in variables:
                    if set(remote_dataset[variable].dims) != {"time", "lat", "lon"}:
                        raise ValueError(
                            f"Unexpected dimensions for {variable}: "
                            f"{remote_dataset[variable].dims}; expected time, lat and lon."
                        )

                matched_latitude = float(
                    remote_dataset["lat"].isel(lat=latitude_index).values
                )
                matched_longitude = float(
                    remote_dataset["lon"].isel(lon=longitude_index).values
                )
                variable_attributes = {
                    variable: dict(remote_dataset[variable].attrs) for variable in variables
                }
                point_dataset = remote_dataset[variables].isel(
                    lat=latitude_index,
                    lon=longitude_index,
                ).load()

            squeeze_dimensions = [
                dimension for dimension in ("lat", "lon") if dimension in point_dataset.dims
            ]
            if squeeze_dimensions:
                point_dataset = point_dataset.squeeze(dim=squeeze_dimensions, drop=True)

            data = point_dataset.to_dataframe()[variables]
            data.index = pd.to_datetime(data.index, errors="coerce", utc=True).tz_convert(None)
            data.index.name = "time"
            data = data[~data.index.isna()]
            data = self._replace_missing_values(data, variable_attributes)
            units = {
                variable: variable_attributes[variable].get("units") for variable in variables
            }

            return data, units, matched_latitude, matched_longitude
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read MERRA2 point data from {url}: {type(exc).__name__}: {exc}"
            ) from exc

    def download_collection_single_point(self, collection_name: str, lat: float, lon: float, start_time: pd.Timestamp, end_time: pd.Timestamp, variables=None):
        if collection_name not in var_list:
            raise ValueError(f"Unsupported collection name: {collection_name}. Available collections are: {list(var_list.keys())}")
        if collection_name in POINT_UNSUPPORTED_COLLECTIONS:
            raise ValueError(f"collection '{collection_name}' is not supported by point CSV mode yet.")

        dataset = var_list[collection_name]["esdt_dir"]
        selected_variables = _resolve_variables_for_collection(collection_name, variables)
        latitude_index, longitude_index, matched_latitude, matched_longitude = self._nearest_grid_point(lat, lon)
        urls = self._get_opendap_urls(dataset, start_time, end_time)
        data_frames = []
        units = {}

        for url in urls:
            logging.info(
                "Requesting MERRA2 collection %s variables %s from %s",
                collection_name,
                selected_variables,
                url,
            )
            data_sub, units_sub, matched_latitude, matched_longitude = self._read_granule_point(
                url,
                latitude_index,
                longitude_index,
                selected_variables,
            )
            data_frames.append(data_sub)
            units.update(units_sub)

        df = pd.concat(data_frames, axis=0)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        df = df.loc[(df.index >= start_time) & (df.index <= end_time), selected_variables]

        if df.empty:
            raise RuntimeError(
                f"No MERRA2 point records were returned for {collection_name} "
                f"between {start_time} and {end_time}."
            )

        meta = {
            "collection_name": collection_name,
            "dataset": dataset,
            "requested_latitude": lat,
            "requested_longitude": lon,
            "matched_latitude": matched_latitude,
            "matched_longitude": matched_longitude,
            "variables": selected_variables,
            "standard_name": var_list[collection_name].get("standard_name"),
            "collection": var_list[collection_name].get("collection"),
            "start": str(start_time),
            "end": str(end_time),
            "units": units,
        }

        return df, meta


# =============================================================================
# Public interface
# =============================================================================

def download_merra2(collection_names: Union[str, List[str]] = None, initial_year: Optional[int] = None, final_year: Optional[int] = None, initial_month: int = 1, final_month: int = 12, initial_day: int = 1, final_day: Optional[int] = None, lat_1: float = -90, lon_1: float = -180, lat_2: float = 90, lon_2: float = 180, lat: Optional[float] = None, lon: Optional[float] = None, mode: str = "area", start=None, end=None, variables=None, output_dir: Union[str, Path] = "MERRA2_data", save_path: Optional[Union[str, Path]] = None, auth: dict = None, username: str = None, password: str = None, merge_timelapse: str = "all", thread_num: int = 1, map_variables: bool = True, timeout: Tuple[int, int] = (30, 300), *, convert_units: bool = True):
    """
    Unified MERRA2 downloader.

    mode="area"
        Download the bounding box into one NetCDF by default and return its directory.
        When collections have different time axes, they keep separate dimensions.
        The legacy merge_timelapse argument is accepted for compatibility; the
        public area downloader always writes one file. Use convert_units=False
        for provider units.

    mode="point"
        Download single-point CSV time series by lat and lon. By default, mapped
        outputs use degrees Celsius, hPa and mm/h for temperature, pressure and
        water-equivalent fluxes. Radiation is already in W/m2. Set
        convert_units=False to preserve product units with mapped names. With
        map_variables=False, names and units stay native. Exported CSVs have a
        JSON metadata sidecar.
    """
    if collection_names is None:
        if mode == "point":
            collection_names = ["rad", "slv", "aer"]
        elif mode == "area":
            raise ValueError("collection_names must be provided in area mode to avoid downloading large default area data.")
        else:
            raise ValueError("mode must be either 'area' or 'point'.")

    collection_names = _normalize_collection_names(collection_names)
    auth = _get_auth(auth=auth, username=username, password=password)

    if mode == "area":
        if start is not None or end is not None:
            first, last = _resolve_date_range(start=start, end=end)
            initial_year, initial_month, initial_day = first.year, first.month, first.day
            final_year, final_month, final_day = last.year, last.month, last.day
        return _download_merra2_area(collection_names=collection_names, initial_year=initial_year, final_year=final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day, lat_1=lat_1, lon_1=lon_1, lat_2=lat_2, lon_2=lon_2, variables=variables, output_dir=output_dir, auth=auth, merge_timelapse=merge_timelapse, thread_num=thread_num, map_variables=map_variables, convert_units=convert_units)

    if mode == "point":
        return _download_merra2_point(collection_names=collection_names, start=start, end=end, initial_year=initial_year, final_year=final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day, lat=lat, lon=lon, variables=variables, save_path=save_path, auth=auth, map_variables=map_variables, timeout=timeout, convert_units=convert_units)

    raise ValueError("mode must be either 'area' or 'point'.")


def _download_merra2_area(collection_names, initial_year, final_year, initial_month=1, final_month=12, initial_day=1, final_day=None, lat_1=-90, lon_1=-180, lat_2=90, lon_2=180, variables=None, output_dir="MERRA2_data", auth=None, merge_timelapse="all", thread_num=1, map_variables=True, convert_units=True):
    if initial_year is None or final_year is None:
        raise ValueError("initial_year and final_year must be provided in area mode.")
    if final_day is None:
        raise ValueError("final_day must be provided in area mode.")

    merra2_var_dicts = _build_merra2_var_dicts(collection_names, variables)

    products = {}
    selected = {}
    for alias, spec in merra2_var_dicts.items():
        first = products.setdefault(spec["esdt_dir"], alias)
        if first == alias:
            selected[alias] = spec
        else:
            variables = selected[first]["var_name"] + spec["var_name"]
            selected[first]["var_name"] = list(dict.fromkeys(variables))
    if merge_timelapse not in {"all", "none", "daily", "monthly", "yearly"}:
        raise ValueError("merge_timelapse must be 'all', 'none', 'daily', 'monthly' or 'yearly'.")

    first = datetime.date(initial_year, initial_month, initial_day)
    last = datetime.date(final_year, final_month, final_day)
    if last < first:
        raise ValueError("end must be later than or equal to start.")
    if not selected:
        raise ValueError("At least one MERRA-2 area collection must be requested.")

    metadata = {
        "mode": "area", "dataset": "MERRA-2", "start_utc": first.isoformat(), "end_utc": last.isoformat(),
        "area": [lat_1, lon_1, lat_2, lon_2],
        "collections": {spec["collection"]: sorted(spec["var_name"]) for spec in selected.values()},
        "convert_units": convert_units, "map_variables": map_variables,
    }
    key = _area_request_tag(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"merra2_{first:%Y%m%d}_{last:%Y%m%d}_{key}"
    socket = SocketManager()

    with TemporaryDirectory(prefix=".merra2-", dir=output_dir) as folder:
        work = Path(folder)
        directory = Path(socket.daily_download_and_convert(
            collection_names=list(selected), initial_year=initial_year, final_year=final_year,
            initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day,
            lat_1=lat_1, lon_1=lon_1, lat_2=lat_2, lon_2=lon_2, merra2_var_dicts=selected, output_dir=work,
            auth=auth, merge_timelapse="none",
            thread_num=thread_num, map_variables=False,
        ))

        staged = work / "merra2.nc"
        with ExitStack() as stack:
            datasets = []
            sources = metadata["collections"]
            mixed_time = len({collection.split("_")[0] for collection in sources}) > 1
            time_names = {
                "tavg3": "time_3h", "inst1": "time_instant",
                "statD": "time_daily", "const": "time_static",
            }
            for collection in sources:
                paths = sorted(directory.glob(f"*.{collection}.*.nc*"), key=lambda path: path.name.split(".")[2])
                if not paths:
                    raise RuntimeError(f"MERRA-2 returned no NetCDF files for {collection}.")
                parts = [ds for path in paths for ds in _open_netcdf_parts(path, work, stack)]
                dataset = _concat_netcdf_time(parts, "time")
                if convert_units:
                    dataset, _ = _standardize_area([dataset], map_variables)
                elif map_variables:
                    dataset = _rename_dataset_with_variable_map(dataset)
                time_name = time_names.get(collection.split("_")[0], "time")
                if mixed_time and time_name != "time" and "time" in dataset.dims:
                    dataset = dataset.rename({"time": time_name})
                datasets.append(dataset)

            data = xr.merge(datasets, join="exact", compat="no_conflicts", combine_attrs="drop_conflicts")
            metadata["units"] = {name: var.attrs["units"] for name, var in data.data_vars.items() if "units" in var.attrs}
            _write_netcdf(data, staged)

        staged_meta = work / "merra2_metadata.json"
        _write_json(metadata, staged_meta)
        destination.mkdir(parents=True, exist_ok=True)
        staged.replace(destination / staged.name)
        staged_meta.replace(destination / staged_meta.name)
    return str(destination)


def _download_merra2_point(collection_names, start=None, end=None, initial_year=None, final_year=None, initial_month=1, final_month=12, initial_day=1, final_day=None, lat=None, lon=None, variables=None, save_path=None, auth=None, map_variables=True, timeout=(30, 300), *, convert_units=True):
    if lat is None or lon is None:
        raise ValueError("lat and lon must be provided in point mode.")

    start_time, end_time = _resolve_date_range(start=start, end=end, initial_year=initial_year, final_year=final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day)

    manager = MERRA2CSVManager(username=auth["uid"], password=auth["password"], timeout=timeout)

    df_list = []
    collection_meta = {}

    for collection_name in collection_names:
        collection_variables = _collection_variables(collection_names, variables, collection_name)
        df_collection, meta_collection = manager.download_collection_single_point(collection_name=collection_name, lat=lat, lon=lon, start_time=start_time, end_time=end_time, variables=collection_variables)
        df_list.append(df_collection)
        collection_meta[collection_name] = meta_collection

    data = pd.concat(df_list, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.sort_index()
    data.index.name = "time"
    data = data.reset_index()

    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    units = {}
    for meta_item in collection_meta.values():
        units.update(meta_item.get("units", {}))

    meta = {"mode": "point", "requested_latitude": lat, "requested_longitude": lon, "start": str(start_time), "end": str(end_time), "collections": collection_names, "collection_meta": collection_meta, "units": units, "map_variables": map_variables, "convert_units": map_variables and convert_units}

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
        mapped_units = {}
        for original_name, unit in units.items():
            mapped_units[VARIABLE_MAP.get(original_name, original_name)] = unit
        if convert_units:
            for name in _UNIT_CONVERSIONS.keys() & set(data.columns):
                target, factor, offset = _normalize_units(name, mapped_units.get(name))
                if factor != 1.0 or offset:
                    data[name] = pd.to_numeric(data[name], errors="coerce") * factor + offset
                mapped_units[name] = target
            if "precipitation_rate" in data:
                data["precipitation"] = data["precipitation_rate"]
                mapped_units["precipitation"] = "mm"
        meta["units"] = mapped_units

    if any(name in units for name in ("SWGDN", "SWGDNCLR")):
        meta["irradiance_timing"] = POINT_RADIATION_TIMING.copy()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, encoding="utf-8-sig", index=False)
        metadata_path = save_path.with_name(f"{save_path.stem}_metadata.json")
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(meta, stream, indent=2, default=str)

    data.attrs["units"] = dict(meta["units"])
    return data, meta


__all__ = [
    "var_list",
    "VARIABLE_MAP",
    "list_merra2_collections",
    "download_merra2",
]
