"""
SRML data downloader.

This module provides a unified downloader for SRML
(Solar Radiation Monitoring Laboratory) data.

Main capabilities
-----------------
- Download SRML data by site and date range
- Decode variable codes and quality flags
- Export processed data to CSV

Notes
-----
This module depends on shared utility functions defined in ``necessary_main``.
"""

import os
import re
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from .necessary_main import (http_get_with_retry,ensure_dir,url_contact,tz_localize_safely,attach_local_and_offset,clip_by_local_date,)

# SRML station mapping.
# This mapping allows users to work with commonly used SRML site names.
# See http://solardata.uoregon.edu/StationIDCodes.html for detailed station codes.
STATION_MAP: Dict[str, str] = {
    "BUO": "BU",
    "CYW": "CY",
    "EUO": "EU",
    "HEO": "HE",
    "MDO": "MD",
    "PDO": "PD",
    "PSO": "PS",
    "SAO": "SA",
    "SIO": "SI",
    "STW": "ST",
}

# All SRML sites use the same local timezone.
SRML_TZ = "America/Los_Angeles"

# Regular SRML variable prefix mapping.
# For most SRML variables, the first three digits indicate data type,
# and the fourth digit differentiates duplicate sensors.
VARIABLE_MAP = {
    "100": "ghi",
    "109": "global_ground_facing",
    "200": "direct_beam",
    "201": "dni",
    "300": "dhi",
    "910": "sky_condition",
    "911": "ceiling_height",
    "912": "visibility",
    "913": "weather_code",
    "915": "precipitation",
    "917": "pressure",
    "920": "wind_dir",
    "921": "wind_speed",
    "922": "wind_dir_std",
    "930": "temp_air",
    "931": "temp_dew",
    "933": "rh",
    "937": "solar_cell_temp",
    "938": "temp_air_min",
    "939": "temp_air_max",
    "940": "pressure_avg",
    "951": "sky_cover_total",
    "952": "sky_cover_opaque",
    "953": "precipitable_water",
    "954": "aerosol_optical_depth",
    "955": "snow_depth",
    "956": "albedo",
    "965": "days_since_last_snowfall",
}

# Exact SRML variable mapping.
# 7xxx spectral data use all four digits as the data type.
# A few non-standard exact codes are also included.
EXACT_VARIABLE_MAP = {
    "7000": "og570_horizontal",
    "7001": "rg630_horizontal",
    "7002": "rg695_horizontal",
    "7003": "uva_horizontal",
    "7004": "uvb_horizontal",
    "7005": "zenith_illuminance",
    "7006": "diffuse_illuminance",
    "7007": "global_illuminance",
    "7008": "max_illuminance_horizontal",
    "7009": "min_illuminance_horizontal",
    "7100": "og570_beam",
    "7101": "rg630_beam",
    "7102": "rg695_beam",
    "7103": "uva_beam",
    "7104": "uvb_beam",
    "7107": "beam_illuminance",
    "7108": "max_illuminance_beam",
    "7109": "min_illuminance_beam",
    "4006": "battery_current_avg",
    "5006": "battery_power_avg",
    "6006": "battery_voltage_avg",
}

# SRML file type priority.
# Different file types represent different time intervals and processing levels.
# The downloader tries these file types in order until one valid file is found.
FILETYPE_PRIORITY = ["PO", "RO", "PF", "RF", "PQ", "RQ", "PH", "RH"]

BASE_URL = "http://solardata.uoregon.edu/download/archive"


def _decode_srml_code(code: str) -> str:
    """
    Decode one SRML element code into a unified column name.

    Rules
    -----
    - For 7xxx spectral variables, all four digits define the type.
    - For most other variables, the first three digits define the type.
    - The fourth digit is usually used to distinguish duplicate sensors.
      For example, 1000 -> ghi, 1001 -> ghi_1, 1002 -> ghi_2.
    """
    code = str(code).strip()

    if code in EXACT_VARIABLE_MAP:
        return EXACT_VARIABLE_MAP[code]

    base = code[:3]
    if base in VARIABLE_MAP:
        name = VARIABLE_MAP[base]
        if len(code) == 4 and code[3].isdigit() and code[3] != "0":
            return f"{name}_{code[3]}"
        return name

    return f"var_{code}"


def _parse_srml_text_to_df(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty SRML text")

    header = re.split(r"\s+", lines[0].strip())
    year = int(header[1])
    pairs = header[2:]
    if len(pairs) % 2 != 0:
        pairs = pairs[:-1]

    codes = pairs[0::2]
    var_names = []

    # Convert SRML variable codes into unified column names.
    seen_names = {}
    for code in codes:
        name = _decode_srml_code(code)
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_dup{seen_names[name]}"
        else:
            seen_names[name] = 0
        var_names.append(name)

    doy_list, raw_list = [], []
    data = {v: [] for v in var_names}

    for ln in lines[1:]:
        parts = re.split(r"\s+", ln.strip())
        if len(parts) < 2 + 2 * len(var_names):
            continue

        doy = int(float(parts[0]))
        hhmm = int(float(parts[1]))
        doy_list.append(doy)
        raw_list.append(hhmm)

        for j, name in enumerate(var_names):
            v = parts[2 + 2 * j]
            f = parts[3 + 2 * j]

            try:
                val = float(v)
            except Exception:
                val = np.nan

            try:
                flag = int(float(f))
            except Exception:
                flag = 0

            if flag == 99:
                val = -999.99

            data[name].append(val)

    # Shift time labels back by one interval.
    if len(raw_list) > 1:
        interval = raw_list[1] - raw_list[0]
    else:
        interval = 100

    adj_times = []
    for t in raw_list:
        t_adj = t - interval
        if interval != 100 and (t_adj % 100) > 60:
            t_adj -= 40
        adj_times.append(t_adj)

    timestamps = [
        f"{year}-{doy:03d}-{hhmm:04d}"
        for doy, hhmm in zip(doy_list, adj_times)
    ]

    dt = pd.to_datetime(timestamps, format="%Y-%j-%H%M")
    df = pd.DataFrame(data, index=dt)
    df.index.name = "datetime"
    return df


def _download_one_month_df(site_dir: str, prefix: str, year: int, month: int):
    """
    Download and parse one month of SRML data.

    Multiple SRML file types are tried according to ``FILETYPE_PRIORITY``.
    """
    yy = f"{year % 100:02d}"
    mm = f"{month:02d}"

    for ft in FILETYPE_PRIORITY:
        fname = f"{prefix}{ft}{yy}{mm}.txt"

        url = url_contact(
            url_contact(
                url_contact(BASE_URL, site_dir),
                f"{site_dir}_{year}"
            ),
            fname
        )

        try:
            text = http_get_with_retry(url)
        except Exception:
            continue

        try:
            df = _parse_srml_text_to_df(text)
            return df
        except Exception:
            continue

    return None


def download_srml(site: str, start: str, end: str, save_path: str):
    """
    Download SRML data for a given site and save the processed result as a CSV file.

    This is the public high-level SRML download interface.

    Parameters
    ----------
    site : str
        SRML site code (for example, ``"BUO"``).
    start : str
        Start date in ``YYYYMMDD`` format.
    end : str
        End date in ``YYYYMMDD`` format.
    save_path : str
        Absolute output CSV file path.

    Returns
    -------
    None
        The merged table is written to ``save_path``.

    Raises
    ------
    ValueError
        When ``save_path`` is empty or not an absolute path, or when
        ``site`` is unknown.

    References
    ----------
    SRML homepage:
    http://solardata.uoregon.edu/

    SRML data archive:
    http://solardata.uoregon.edu/download/archive/
    """
    if not save_path or not os.path.isabs(save_path):
        raise ValueError("save_path must be a non-empty absolute path.")

    site = site.upper().strip()
    if site not in STATION_MAP:
        raise ValueError(f"Unknown SRML site: {site}")

    prefix = STATION_MAP[site]
    tz_name = SRML_TZ

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    print(f"[SRML] site={site}, start={start}, end={end}")
    print(f"[SRML] output={save_path}")

    # Build the month list covering the requested date range.
    months = []
    cur = datetime(start_dt.year, start_dt.month, 1)
    last = datetime(end_dt.year, end_dt.month, 1)

    while cur <= last:
        months.append((cur.year, cur.month))
        cur = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)

    dfs = []
    for y, m in months:
        df_m = _download_one_month_df(site, prefix, y, m)
        if df_m is not None:
            dfs.append(df_m)

    print(f"[SRML] months={len(months)}, downloaded={len(dfs)}")

    if not dfs:
        print("[SRML] warning=no data available")
        return

    df = pd.concat(dfs).sort_index()

    # Localize naive timestamps to the SRML site timezone.
    naive = df.index.to_series()
    local_ts = tz_localize_safely(naive, tz_name)

    # Clip records to the requested local date range.
    mask, clipped = clip_by_local_date(local_ts, start, end, tz_name)
    df = df.loc[mask].reset_index(drop=True)
    clipped = clipped.reset_index(drop=True)

    # Add local time and UTC offset columns.
    df = attach_local_and_offset(
        df,
        clipped,
        local_col="local_time",
        offset_col="utc_offset"
    )

    # Add UTC time column.
    utc_ts = clipped.dt.tz_convert("UTC")
    df.insert(0, "UTC_time", utc_ts.dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Save output.
    ensure_dir(os.path.dirname(save_path))
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[SRML] rows={len(df)}")
    print(f"[SRML] saved={save_path}")