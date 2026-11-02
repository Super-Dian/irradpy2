"""
SAURAN data downloader.

This module provides a unified downloader for SAURAN
(Southern African Universities Radiometric Network) data.

Main capabilities
-----------------
- Download SAURAN data by site and date range
- Support multiple timestamp formats
- Export processed data to CSV

Notes
-----
This module depends on shared utility functions defined in ``necessary_main``.
"""

from pathlib import Path
from io import StringIO
from datetime import timezone, timedelta
import pandas as pd
import requests
import urllib3

from .necessary_main import init_str_date, url_contact, fmt_offset


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


COLUMN_ALIAS_MAP = {
    "TmStamp": "local_time",
    "RecNum": "record_number",

    "GHI_Avg": "ghi",
    "CMP11_2SlrW": "ghi",
    "CMP11_2SlrW_Avg": "ghi",
    "SunWM_Avg": "ghi",
    "Sol_Global1_Avg": "ghi",
    "GHI1_Avg": "ghi",

    "DIF_Avg": "dhi",
    "CMP11_1SlrW": "dhi",
    "CMP11_1SlrW_Avg": "dhi",
    "ShadowWM_Avg": "dhi",
    "Sol_Global2_Avg": "dhi",
    "ShadowbandWM_Avg": "dhi",

    "DNI_Avg": "dni",
    "CHP1SlrW": "dni",
    "CHP1SlrW_Avg": "dni",
    "TrackerWM_Avg": "dni",
    "Sol_Direct_Avg": "dni",
    "DNICalc_Avg": "dni_calc",

    "GHI2_Avg": "ghi_2",
    "Tracker2WM_Avg": "dni_2",
    "DNI_2_Avg": "dni_2",
    "SPN1dig_Total_Avg": "ghi_spn1",
    "SPN1dig_Diffuse_Avg": "dhi_spn1",

    "Temp_Avg": "temp_air",
    "AirTC": "temp_air",
    "AirTC_Avg": "temp_air",
    "RH": "rh",
    "RH_Avg": "rh",
    "RH_Min": "rh",
    "BP_Avg": "pressure",
    "BP_mbar_Avg": "pressure",
    "BP_mbar_Min": "pressure",
    "BP_mB_Avg": "pressure",
    "LoggerTemp_Avg": "logger_temp",
    "SolarTemp_Avg": "solar_temp",

    "WS_Avg": "wind_speed",
    "WSpd_Avg": "wind_speed",
    "WS_ms_S_WVT": "wind_speed",
    "WS_Max": "wind_speed_max",
    "WS_ms_Max": "wind_speed_max",
    "WVec_Mag_Avg": "wind_vector_mag",
    "WD_Avg": "wind_dir",
    "WDir_Avg": "wind_dir",
    "WindDir_D1_WVT": "wind_dir",
    "WD_StdDev": "wind_dir_std",
    "WindDir_SD1_WVT": "wind_dir_std",
    "WindGust_Avg": "wind_gust_avg",
    "WindGust_Max": "wind_gust_max",
    "WindGust_Min": "wind_gust_min",

    "Rain_Tot": "precipitation",
    "Rain_mm_Tot": "precipitation",
    "Rainfall_Tot": "precipitation",

    "CUV5SlrW": "uv_total",
    "UVA_Avg": "uva",
    "UVB_Avg": "uvb",
    "UVA_Temp_Avg": "uva_temp",

    "Calc_Azimuth": "calc_azimuth",
    "Calc_Azimuth_Avg": "calc_azimuth",
    "Calc_Tilt": "calc_tilt",
    "Calc_Tilt_Avg": "calc_tilt",
    "Solys_Azimuth": "solys_azimuth",
    "Solys_Tilt": "solys_tilt",
    "Solys_Lat": "solys_lat",
    "Solys_Long": "solys_long",

    "LoggerStationName": "station_name",
    "LoggerSerialNumber": "logger_serial_number",
    "GHI_SN": "ghi_sn",
    "DIF_SN": "dhi_sn",
    "DNI_SN": "dni_sn",
    "SOLYS2_SN": "solys2_sn",
}


def _normalize_sauran_column(col: str) -> str:
    col = str(col).strip()
    return COLUMN_ALIAS_MAP.get(col, col)


def _parse_tmstamp_series(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip()
    formats = ["%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"]

    ts = pd.to_datetime(s, format=formats[0], errors="coerce")

    for fmt in formats[1:]:
        ts = ts.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))

    ts = ts.fillna(pd.to_datetime(s, errors="coerce", dayfirst=True))
    return ts


def download_sauran(site: str, start: str, end: str, save_path: str) -> None:
    """
    Download SAURAN data and save the processed result as a CSV file.

    Parameters
    ----------
    site : str
        SAURAN station identifier, for example "NMU" or "UNV".
    start : str
        Start date in YYYYMMDD or YYYY-MM-DD format.
    end : str
        End date in YYYYMMDD or YYYY-MM-DD format.
    save_path : str
        Absolute output CSV file path.

    Returns
    -------
    None
        The processed data are written directly to save_path.
    """
    save_path = Path(save_path)
    tz_offset_hours=2.0
    if not save_path.is_absolute():
        raise ValueError("save_path must be a non-empty absolute path.")

    site = site.upper().strip()
    base_url_https = "https://sauran.ac.za/api/DataDownload"
    base_url_http = "http://sauran.ac.za/api/DataDownload"
    user_type = "Minute"

    start_str = init_str_date(start)
    end_str = init_str_date(end)

    print(f"[SAURAN] site={site}, start={start_str.replace('-', '')}, end={end_str.replace('-', '')}")
    print(f"[SAURAN] output={save_path}")

    def build_url(base):
        url = url_contact(base, site)
        url = url_contact(url, user_type)
        url = url_contact(url, start_str)
        url = url_contact(url, end_str)
        return url

    url = build_url(base_url_https)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, verify=False)
        resp.raise_for_status()
    except Exception:
        print("[SAURAN] https_failed=True, retry_http=True")
        url = build_url(base_url_http)
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()

    lines = resp.text.splitlines()
    if len(lines) < 5:
        raise RuntimeError(f"Response too short: only {len(lines)} lines.")

    field_row = [x.strip() for x in lines[1].split(",")]
    unit_row = [x.strip() for x in lines[2].split(",")]

    data_text = "\n".join(lines[4:])
    df = pd.read_csv(StringIO(data_text), header=None)

    data_cols = df.shape[1]

    if len(unit_row) < len(field_row):
        unit_row += [""] * (len(field_row) - len(unit_row))
    if len(field_row) < len(unit_row):
        field_row += [""] * (len(unit_row) - len(field_row))

    header = field_row

    if len(header) < data_cols:
        header += [f"Extra_{i}" for i in range(len(header), data_cols)]
    elif len(header) > data_cols:
        header = header[:data_cols]

    first_col = header[0]
    if not first_col.lower().startswith("tmstamp"):
        if "tmstamp" not in first_col.lower():
            raise RuntimeError(f"Invalid format: expected TmStamp, got: {first_col}")

    df.columns = header
    df.columns = [_normalize_sauran_column(col) for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    naive_ts = _parse_tmstamp_series(df["local_time"]) - pd.Timedelta(minutes=1)

    tz = timezone(timedelta(hours=tz_offset_hours))
    local_ts = naive_ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    local_str = local_ts.dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
    utc_str = local_ts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")
    offset_str = fmt_offset(tz_offset_hours)

    df.drop(columns=["local_time"], inplace=True)
    df.insert(0, "UTC_time", utc_str)
    df.insert(1, "local_time", local_str)
    df.insert(2, "utc_offset", offset_str)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[SAURAN] rows={len(df)}")
    print(f"[SAURAN] saved={save_path}")
