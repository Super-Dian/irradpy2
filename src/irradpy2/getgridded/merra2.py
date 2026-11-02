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
import logging
import multiprocessing
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http import cookiejar
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import netCDF4
import numpy as np
import pandas as pd
import requests
import xarray as xr

try:
    import earthaccess
    from pydap.client import get_cmr_urls
    from pydap.net import create_session
except ImportError:
    earthaccess = None
    get_cmr_urls = None
    create_session = None

ssl._create_default_https_context = ssl._create_unverified_context


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
    "LWGEM": "longwave_up",
    "LWGAB": "longwave_down",
    "CLDTOT": "cloud_fraction",
    "TAUTOT": "cloud_optical_thickness",
    "T2M": "temp_air",
    "T2MDEW": "temp_dew",
    "T2MMAX": "temp_air_max",
    "T2MMIN": "temp_air_min",
    "TQV": "water_vapour",
    "TO3": "ozone",
    "PS": "pressure",
    "QV2M": "specific_humidity",
    "TOTSCATAU": "aod_scattering",
    "TOTEXTTAU": "aod550",
    "TOTANGSTR": "angstrom_exponent",
    "EVAP": "evaporation",
    "PRECTOT": "precipitation",
    "PRECCON": "convective_precipitation",
    "PRECTOTCORR": "precipitation_corrected",
    "TPRECMAX": "precipitation_max",
    "PRECSNO": "snowfall",
    "U10M": "wind_u_10m",
    "V10M": "wind_v_10m",
    "FRSEAICE": "sea_ice_fraction",
    "PHIS": "surface_geopotential",
}


POINT_UNSUPPORTED_COLLECTIONS = {"asm", "phis", "sftgif", "sftkf", "sftlf", "sftof", "shg", "hur"}


# =============================================================================
# Common helpers
# =============================================================================

def list_merra2_collections():
    """Return a copy of the built-in MERRA2 collection registry.

    Returns
    -------
    dict
        A copy of ``{collection_name: {...}}``; mutating it does not affect
        the module-level table.
    """
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
        """
        Return the configured download URL.

        Returns
        -------
        str
            The download URL, or an empty string when unset.
        """
        return self.__download_url

    @download_url.setter
    def download_url(self, link):
        """
        Set the download URL after validating its structure.

        Parameters
        ----------
        link : str
            Download URL; ``None`` is stored as an empty string.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When the URL does not contain a recognisable filename.
        """
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
        """
        Return the local download directory.

        Returns
        -------
        str
            Directory that receives downloaded files.
        """
        return self.__download_path

    @download_path.setter
    def download_path(self, file_path):
        """
        Set the local download directory.

        Parameters
        ----------
        file_path : str
            Target directory; created on demand by ``start_download``.

        Returns
        -------
        None
        """
        self.__download_path = file_path

    def set_username_and_password(self, username, password):
        """
        Store Earthdata Login credentials for later downloads.

        Parameters
        ----------
        username : str
            Earthdata Login user name.
        password : str
            Earthdata Login password.

        Returns
        -------
        None
        """
        self.__username = username
        self.__password = password

    def _mp_download_wrapper(self, url_item, filename=None):
        file_path = os.path.join(self.download_path, filename or self.get_filename(url_item))
        self.__download_and_save_file(url_item, file_path)

    def start_download(self, filename=None):
        """
        Download ``download_url`` into ``download_path``.

        Creates the target directory and an authenticated session on first use.

        Parameters
        ----------
        filename : str, optional
            Output filename. When ``None``, it is derived from the download URL.

        Returns
        -------
        None
            The file is written to ``download_path``; HTML responses and empty
            bodies raise ``RuntimeError``.
        """
        if self._authenticated_session is None:
            self._authenticated_session = self.__create_authenticated_sesseion()
        os.makedirs(self.download_path, exist_ok=True)
        self._mp_download_wrapper(self.download_url, filename=filename)

    @staticmethod
    def get_filename(url):
        """
        Extract the file name from a URL.

        Parameters
        ----------
        url : str
            URL whose last path component carries the file name.

        Returns
        -------
        str
            The file name including its extension.

        Raises
        ------
        ValueError
            When no file name can be extracted.
        """
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

        urllib.request.install_opener(opener)
        opener.open(self.download_url)

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

    def generate_url_params(self, parameter, time_para, lat_para, lon_para):
        """
        Build an OPeNDAP constraint expression for the requested variables.

        Parameters
        ----------
        parameter : sequence of str
            Base variable names (``{var}{time}{lat}{lon}`` suffixes are appended).
        time_para : str
            Time-dimension index slice, e.g. ``"[0:23]"``.
        lat_para : str
            Latitude-dimension index slice.
        lon_para : str
            Longitude-dimension index slice.

        Returns
        -------
        str
            Comma-joined variable selections plus the ``time``/``lat``/``lon``
            dimension selections.
        """
        parameter = map(lambda x: x + time_para, parameter)
        parameter = map(lambda x: x + lat_para, parameter)
        parameter = map(lambda x: x + lon_para, parameter)
        addition = ["time", "lat" + lat_para, "lon" + lon_para]
        return ",".join(parameter) + "," + ",".join(addition)

    def translate_lat_to_geos5_native(self, latitude):
        """
        Translate a latitude in degrees to the GEOS-5 native index.

        Parameters
        ----------
        latitude : float
            Latitude in degrees.

        Returns
        -------
        float
            Native grid index ``(latitude + 90) / 0.5``.
        """
        return ((latitude + 90) / 0.5)

    def translate_lon_to_geos5_native(self, longitude):
        """
        Translate a longitude in degrees to the GEOS-5 native index.

        Parameters
        ----------
        longitude : float
            Longitude in degrees.

        Returns
        -------
        float
            Native grid index ``(longitude + 180) / 0.625``.
        """
        return ((longitude + 180) / 0.625)

    def find_closest_coordinate(self, calc_coord, coord_array):
        """
        Return the grid coordinate nearest to a computed index.

        Parameters
        ----------
        calc_coord : float
            Target coordinate value.
        coord_array : numpy.ndarray
            Candidate grid coordinates.

        Returns
        -------
        float
            The candidate closest to ``calc_coord``.
        """
        index = np.abs(coord_array - calc_coord).argmin()
        return coord_array[index]

    def build_remote_url(self, merra2_collection, date):
        """
        Build the OPeNDAP directory URL for a collection and date.

        Parameters
        ----------
        merra2_collection : dict
            Collection registry entry (uses ``esdt_dir`` and ``collection``).
        date : datetime.date
            Granule date; constant collections use the fixed 1980 monthly path.

        Returns
        -------
        str
            Remote directory URL.

        Notes
        -----
        Legacy URL builder; live area downloads use ``_build_cloud_area_url``.
        """
        if merra2_collection["collection"].startswith("const"):
            return ("https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/{db_name}/1980/").format(db_name=merra2_collection["esdt_dir"])
        return ("https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{db_name}/{date:%Y}/{date:%m}/").format(db_name=merra2_collection["esdt_dir"], date=date)

    def reconstruct_filename(self, name, params):
        """
        Rebuild the ``name?params`` key used in download logs.

        Parameters
        ----------
        name : str
            File name.
        params : str or None
            OPeNDAP constraint expression appended after ``?``.

        Returns
        -------
        str
            ``name?params`` (``name`` alone when ``params`` is None).
        """
        return "{name}?{params}".format(name=name, params=params)

    def build_remote_filename(self, merra2_collection, date, params):
        """
        Build the MERRA-2 granule file name (optionally with constraints).

        Parameters
        ----------
        merra2_collection : dict
            Collection registry entry (uses ``collection``).
        date : datetime.date
            Granule date; selects the MERRA-2 stream (100/200/300/400) from the
            year. Constant collections use stream 101, date ``00000000`` and the
            ``nc4`` extension.
        params : str or None
            OPeNDAP constraint expression appended after ``?`` when given.

        Returns
        -------
        str
            File name such as ``MERRA2_400.tavg1_2d_rad_Nx.20250101.nc4.nc``.
        """
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
        """
        Download one daily MERRA-2 granule with a single retry attempt.

        Parameters
        ----------
        date : datetime.date
            Granule date.
        merra2_collection : dict
            Collection registry entry.
        output_directory : str or pathlib.Path
            Destination directory (existing file for the date is replaced).
        params : str or None
            OPeNDAP constraint expression.
        auth : dict
            Earthdata credentials ``{"uid": ..., "password": ...}``.

        Returns
        -------
        bool
            True when the download succeeded; False after the failed attempt
            (``global_retry`` is set unless the cause was authentication).
        """
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
        """
        Yield every calendar day from ``first`` to ``last`` inclusive.

        Parameters
        ----------
        first : datetime.date
            Inclusive range start.
        last : datetime.date
            Inclusive range end.

        Returns
        -------
        generator of datetime.date
        """
        current = first
        while current <= last:
            yield current
            current += datetime.timedelta(1)

    def iter_months(self, first: datetime.date, last: datetime.date):
        """
        Yield every month from ``first`` to ``last`` as ``YYYY-MM`` strings.

        Parameters
        ----------
        first : datetime.date
            Inclusive range start.
        last : datetime.date
            Inclusive range end.

        Returns
        -------
        generator of str
        """
        ym_start = 12 * first.year + first.month - 1
        ym_end = 12 * last.year + last.month - 1
        for ym in range(ym_start, ym_end + 1):
            y, m = divmod(ym, 12)
            yield "{:0>4d}-{:0>2d}".format(y, m + 1)

    def iter_years(self, first: datetime.date, last: datetime.date):
        """
        Yield one item per year from ``first`` to ``last`` inclusive.

        Parameters
        ----------
        first : datetime.date
            Inclusive range start.
        last : datetime.date
            Inclusive range end.

        Returns
        -------
        generator of str
            Each item is the unfilled template string ``{:0>4d}``; formatting it
            with the year is left to the caller.
        """
        for y in range(first.year, last.year + 1):
            yield "{:0>4d}".format(y)

    def subdaily_universal_download(self, merra2_collection: dict, initial_year: int, final_year: int, initial_month: int = 1, final_month: int = 12, initial_day: int = 1, final_day: Optional[int] = None, params: Optional[str] = None, auth: dict = None, output_directory: Union[str, Path] = None, thread_num: int = 5, merge: str = "monthly"):
        """
        Download all daily granules of a collection for a date range.

        Skips files already recorded in the ``index.npy`` resume log, verifies
        existing files by opening them, downloads missing days in batches of
        ``thread_num`` (process pool when greater than 1) and marks integrity
        failures on ``global_retry``.

        Parameters
        ----------
        merra2_collection : dict
            Collection registry entry.
        initial_year : int
            First year (with ``initial_month``/``initial_day``).
        final_year : int
            Last year (with ``final_month``/``final_day``).
        initial_month : int, default 1
            First month.
        final_month : int, default 12
            Last month.
        initial_day : int, default 1
            First day.
        final_day : int, optional
            Last day; required to build the end date.
        params : str or None, optional
            OPeNDAP constraint expression.
        auth : dict, optional
            Earthdata credentials.
        output_directory : str or pathlib.Path, optional
            Destination directory (also anchors the resume log).
        thread_num : int, default 5
            Parallel download workers.
        merge : str, default "monthly"
            Merge policy recorded in the resume log.

        Returns
        -------
        None
        """
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
                    ds = xr.open_dataset(os.path.join(output_directory, name), engine="netcdf4")
                    lat = ds.lat
                    lon = ds.lon
                    intact = bool((lat != 0).any() or (lon != 0).any())
                    ds.close()
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
                            ds = xr.open_dataset(os.path.join(output_directory, name), engine="netcdf4")
                            lat = ds.lat
                            lon = ds.lon
                            intact = bool((lat != 0).any() or (lon != 0).any())
                            ds.close()
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
        """
        Merge the daily NetCDF files of one date into a single dataset.

        Parameters
        ----------
        path_data : str or pathlib.Path
            Directory searched recursively for ``*YYYYMMDD.nc4.nc`` files.
        collection_names : list of str
            Collections whose files participate in the merge.
        final_year : int
            Last year of the period used to build the output time range.
        final_month : int
            Last month of that period.
        final_day : int
            Last day of that period.
        date : datetime.date
            Day to merge; days without files are skipped.
        map_variables : bool, default True
            Rename variables to standard names in the merged output.

        Returns
        -------
        delete_set : list of str
            Paths of the consumed source files (removed by the caller after a
            successful merge).
        """
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
                final_ds = xr.merge([final_ds, remote_ds])
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
        """
        Merge the daily NetCDF files of one month into a single dataset.

        Parameters
        ----------
        path_data : str or pathlib.Path
            Directory searched recursively for ``*YYYY-MM-*.nc`` files.
        collection_names : list of str
            Collections whose files participate in the merge.
        final_year : int
            Last year of the period used to build the output time range.
        final_month : int
            Last month of that period.
        final_day : int
            Last day of that period.
        date : str
            Month to merge in ``YYYY-MM`` format; months without files are
            skipped.
        map_variables : bool, default True
            Rename variables to standard names in the merged output.

        Returns
        -------
        delete_set : list of str
            Paths of the consumed source files.
        """
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
        """
        Merge the monthly NetCDF files of one year into a single dataset.

        Parameters
        ----------
        path_data : str or pathlib.Path
            Directory searched recursively for ``*YYYY-MM-*.nc`` files.
        collection_names : list of str
            Collections whose files participate in the merge.
        final_year : int
            Last year of the period used to build the output time range.
        final_month : int
            Last month of that period.
        final_day : int
            Last day of that period.
        date : str
            Year to merge; years without files are skipped.
        map_variables : bool, default True
            Rename variables to standard names in the merged output.

        Returns
        -------
        delete_set : list of str
            Paths of the consumed source files.
        """
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
        """
        Download and convert an area subset of MERRA-2 collections.

        End-to-end area workflow: download every daily granule in the date
        range, then optionally merge per day/month/year. Up to three attempts
        are made when the integrity check fails.

        Parameters
        ----------
        collection_names : list of str
            Collections to download.
        initial_year : int
            First year (with ``initial_month``/``initial_day``).
        final_year : int, optional
            Last year; defaults to the current year at import time.
        initial_month : int, default 1
            First month.
        final_month : int, default 12
            Last month.
        initial_day : int, default 1
            First day.
        final_day : int, optional
            Last day; defaults to today at import time.
        lat_1 : float, default -90
            First corner latitude [degrees].
        lon_1 : float, default -180
            First corner longitude [degrees].
        lat_2 : float, default 90
            Opposite corner latitude [degrees]; must satisfy ``lat_1 <= lat_2``.
        lon_2 : float, default 180
            Opposite corner longitude [degrees]; must satisfy ``lon_1 <= lon_2``.
        merra2_var_dicts : dict, optional
            Per-collection variable selections (built from the registry when
            None).
        output_dir : str or pathlib.Path, optional
            Output root; defaults to the working directory.
        auth : dict, optional
            Earthdata credentials ``{"uid": ..., "password": ...}``.
        merge_timelapse : str, default "monthly"
            One of ``none``/``daily``/``monthly``/``yearly``.
        thread_num : int, default 5
            Parallel download workers.
        map_variables : bool, default True
            Rename variables during merging.

        Returns
        -------
        str
            Path of the dated subdirectory holding the downloaded (and merged)
            NetCDF files.

        Raises
        ------
        RuntimeError
            For an illegal bounding box or merge policy, or after three failed
            download attempts.
        """
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

                if merra2_var_dict["collection"].startswith("const"):
                    requested_time = "[0:0]"
                else:
                    requested_time = "[0:23]"

                parameter = self.generate_url_params(requested_params, requested_time, requested_lat, requested_lon)

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
        """
        Fetch one collection as a single-point CSV-ready time series.

        Parameters
        ----------
        collection_name : str
            Registry key (e.g. ``rad``); must support point mode.
        lat : float
            Point latitude [degrees], snapped to the nearest grid point.
        lon : float
            Point longitude [degrees], snapped to the nearest grid point.
        start_time : pd.Timestamp
            Inclusive range start (UTC).
        end_time : pd.Timestamp
            Inclusive range end (UTC).
        variables : str, list or dict, optional
            Variable subset; defaults to the collection's full variable list.

        Returns
        -------
        data : pandas.DataFrame
            Time-indexed variables for the matched grid point.
        meta : dict
            Coordinates (requested and matched), selected variables, units and
            the covered range.

        Raises
        ------
        ValueError
            For unknown collections, or collections unsupported in point mode.
        RuntimeError
            When the request returns no records.
        """
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

def download_merra2(collection_names: Union[str, List[str]] = None, initial_year: Optional[int] = None, final_year: Optional[int] = None, initial_month: int = 1, final_month: int = 12, initial_day: int = 1, final_day: Optional[int] = None, lat_1: float = -90, lon_1: float = -180, lat_2: float = 90, lon_2: float = 180, lat: Optional[float] = None, lon: Optional[float] = None, mode: str = "area", start=None, end=None, variables=None, output_dir: Union[str, Path] = "MERRA2_data", save_path: Optional[Union[str, Path]] = None, auth: dict = None, username: str = None, password: str = None, merge_timelapse: str = "none", thread_num: int = 1, map_variables: bool = True, timeout: Tuple[int, int] = (30, 300)):
    """
    Unified MERRA2 downloader for gridded NetCDF subsets and single-point CSV time series.

    mode="area" downloads NetCDF subset files by four area parameters (lat_1, lon_1, lat_2, lon_2); mode="point" downloads a single-point CSV time series by lat and lon.

    Parameters
    ----------
    collection_names : str or list of str, optional
        MERRA-2 collection(s) from the built-in registry (`rad`, `slv`, `aer`, `asm`, `dewpt`, `evspsbl`, `hur`, `huss`, `phis`, `pr`, `prc`, `prbc`, `prmax`, `prsn`, `rls`, `sftgif`, `sftkf`, `sftlf`, `sftof`, `shg`, `sic`, `tas`, `tasmax`, `tasmin`, `uas`, `vas`). Point mode defaults to `["rad", "slv", "aer"]`; area mode requires it explicitly (`ValueError`) to avoid huge default downloads.
    initial_year : int, optional
        First year. Required in area mode (`ValueError`); in point mode used only when `start`/`end` are not given.
    final_year : int, optional
        Last year; same rules as `initial_year`.
    initial_month : int, default 1
        First month of the year/month/day range.
    final_month : int, default 12
        Last month of the year/month/day range.
    initial_day : int, default 1
        First day of the year/month/day range.
    final_day : int, optional
        Last day. Required in area mode and whenever the year/month/day range form is used (`ValueError`).
    lat_1 : float, default -90
        Area corner latitude [degrees] (area mode); must satisfy `lat_1 <= lat_2` (`RuntimeError` otherwise).
    lon_1 : float, default -180
        Area corner longitude [degrees] (area mode); must satisfy `lon_1 <= lon_2` (`RuntimeError` otherwise).
    lat_2 : float, default 90
        Opposite area corner latitude [degrees] (area mode).
    lon_2 : float, default 180
        Opposite area corner longitude [degrees] (area mode).
    lat : float, optional
        Point latitude [degrees]. Required in point mode (`ValueError`).
    lon : float, optional
        Point longitude [degrees]. Required in point mode (`ValueError`).
    mode : str, default "area"
        `"area"` (NetCDF subset by four corners) or `"point"` (single-point CSV time series); `ValueError` otherwise.
    start : Any, optional
        Point-mode period start (e.g. `"YYYY-MM-DD"`); `start` and `end` must be given together. Date-only values start at 00:00:00 UTC.
    end : Any, optional
        Point-mode period end; date-only values extend to 23:59:59.
    variables : sequence or dict, optional
        Variable subset. A string or list is allowed for a single collection; with multiple collections pass a dict `{collection: ["VAR", ...]}` (`ValueError` otherwise). Names must exist in the collection's registry (`ValueError`).
    output_dir : str or pathlib.Path, default "MERRA2_data"
        Area-mode output directory (created if missing); downloads land in a dated subdirectory.
    save_path : str or pathlib.Path, optional
        Point mode: also write the resulting table to this CSV. Ignored in area mode.
    auth : dict, optional
        Earthdata credentials `{"uid": ..., "password": ...}`; alternative to `username`/`password`. Required in both modes (`ValueError` when absent).
    username : str, optional
        Earthdata Login username (combined with `password` into `auth`).
    password : str, optional
        Earthdata Login password.
    merge_timelapse : str, default "none"
        Area mode: `"none"`, `"daily"`, `"monthly"` or `"yearly"` — merge downloaded files afterwards (`RuntimeError` otherwise).
    thread_num : int, default 1
        Area mode: parallel download threads.
    map_variables : bool, default True
        Rename columns to standard names (`VARIABLE_MAP`).
    timeout : tuple of float, default (30, 300)
        Point mode: `(connect, read)` timeouts in seconds for OPeNDAP requests.

    Returns
    -------
    Depends on `mode`: area mode returns `str` — the directory holding the downloaded (and merged) NetCDF files; point mode returns `(pandas.DataFrame, dict)` — the time-series table (with `time` as the first column) and metadata including per-variable units.

    Notes
    -----
    Requires NASA Earthdata Login credentials. Area mode validates `collection_names`, `initial_year`, `final_year` and `final_day` up front (`ValueError`) and raises `RuntimeError` when the bounding box or `merge_timelapse` is invalid, or after three failed download retries.
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
        return _download_merra2_area(collection_names=collection_names, initial_year=initial_year, final_year=final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day, lat_1=lat_1, lon_1=lon_1, lat_2=lat_2, lon_2=lon_2, variables=variables, output_dir=output_dir, auth=auth, merge_timelapse=merge_timelapse, thread_num=thread_num, map_variables=map_variables)

    if mode == "point":
        return _download_merra2_point(collection_names=collection_names, start=start, end=end, initial_year=initial_year, final_year=final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day, lat=lat, lon=lon, variables=variables, save_path=save_path, auth=auth, map_variables=map_variables, timeout=timeout)

    raise ValueError("mode must be either 'area' or 'point'.")


def _download_merra2_area(collection_names, initial_year, final_year, initial_month=1, final_month=12, initial_day=1, final_day=None, lat_1=-90, lon_1=-180, lat_2=90, lon_2=180, variables=None, output_dir="MERRA2_data", auth=None, merge_timelapse="none", thread_num=1, map_variables=True):
    if initial_year is None or final_year is None:
        raise ValueError("initial_year and final_year must be provided in area mode.")
    if final_day is None:
        raise ValueError("final_day must be provided in area mode.")

    merra2_var_dicts = _build_merra2_var_dicts(collection_names, variables)

    socket = SocketManager()
    return socket.daily_download_and_convert(collection_names=collection_names, initial_year=initial_year, final_year=final_year, initial_month=initial_month, final_month=final_month, initial_day=initial_day, final_day=final_day, lat_1=lat_1, lon_1=lon_1, lat_2=lat_2, lon_2=lon_2, merra2_var_dicts=merra2_var_dicts, output_dir=output_dir, auth=auth, merge_timelapse=merge_timelapse, thread_num=thread_num, map_variables=map_variables)


def _download_merra2_point(collection_names, start=None, end=None, initial_year=None, final_year=None, initial_month=1, final_month=12, initial_day=1, final_day=None, lat=None, lon=None, variables=None, save_path=None, auth=None, map_variables=True, timeout=(30, 300)):
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

    meta = {"mode": "point", "requested_latitude": lat, "requested_longitude": lon, "start": str(start_time), "end": str(end_time), "collections": collection_names, "collection_meta": collection_meta, "units": units, "map_variables": map_variables}

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
        mapped_units = {}
        for original_name, unit in units.items():
            mapped_units[VARIABLE_MAP.get(original_name, original_name)] = unit
        meta["units"] = mapped_units

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, encoding="utf-8-sig", index=False)

    return data, meta


__all__ = [
    "var_list",
    "VARIABLE_MAP",
    "list_merra2_collections",
    "download_merra2",
]
