# midc.py
"""
MIDC data downloader.

This module provides a unified downloader and parser for NREL MIDC radiation data.

Main capabilities
-----------------
- Download MIDC data by site and date range
- Resolve local time, UTC time, and UTC offset
- Rename raw station columns to standardized variable names
- Generate standardized CSV output

Notes
-----
This module is built on top of shared utility functions defined in ``necessary_main.py``.
"""

from datetime import datetime, timedelta
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
from .necessary_main import http_get_with_retry, read_flexible_csv, attach_local_and_offset, clip_by_local_date, expand_unnamed_columns


SITE_COLUMN_MAP = {
    "BMS": {
        'Global CMP22 (vent/cor) [W/m^2]': 'ghi',
        'Direct CHP1-1 [W/m^2]': 'dni_chp1',
        'Direct NIP [W/m^2]': 'dni_nip',
        'Diffuse CM22-1 (vent/cor) [W/m^2]': 'dhi',
        'Avg Wind Speed @ 6ft [m/s]': 'wind_speed',
        'Tower Dry Bulb Temp [deg C]': 'temp_air',
        'Tower RH [%]': 'relative_humidity',
        "Total Cloud Cover [%]": "cloud_fraction",
        "Opaque Cloud Cover [%]": "opaque_cloud_fraction"
    },
    "UOSMRL": {
        "Global CMP22 [W/m^2]": "ghi",
        "Direct CHP1 [W/m^2]": "dni",
        "Diffuse CMP22 [W/m^2]": "dhi",
        "Air Temperature [deg C]": "temp_air",
        "Relative Humidity [%]": "rh",
        "Station Pressure [mBar]": "pressure",
        "Avg Wind Speed @ 10m [m/s]": "wind_speed",
        "Avg Wind Direction @ 10m [deg from N]": "wind_dir",
        "Peak Wind Speed @ 10m [m/s]": "wind_speed_max",
        "Zenith Angle [degrees]": "zenith_angle",
        "Azimuth Angle [degrees]": "azimuth_angle",
        "Downwelling IR PIR [W/m^2]": "longwave_down",
        "Instrument Net PIR [W/m^2]": "longwave_net",
        "Global UV [W/m^2]": "uv_total",
        "Upwelling Shortwave [W/m^2]": "shortwave_up",
    },
    "HSU": {
        "Global Horiz [W/m^2]": "ghi",
        "Diffuse Horiz (shadowband) [W/m^2]": "dhi",
        "Diffuse Horiz (band corr) [W/m^2]": "dhi_2",
        "Zenith Angle [degrees]": "zenith_angle",
        "Azimuth Angle [degrees]": "azimuth_angle",
        "Airmass": "airmass",
    },
    "UTPASRL": {
        "Global Horizontal [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "CHP1 Temp [deg C]": "chp1_temp",
    },
    "UAT": {
        "Global Horiz (platform) [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horiz [W/m^2]": "dhi",
        "Air Temperature [deg C]": "temp_air",
        "Rel Humidity [%]": "rh",
        "Avg Wind Speed @ 3m [m/s]": "wind_speed",
    },
    "STAC": {
        "Global Horizontal [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "Avg Wind Speed @ 10m [m/s]": "wind_speed",
        "Air Temperature [deg C]": "temp_air",
        "Rel Humidity [%]": "rh",
    },
    "UNLV": {
        "Direct Normal [W/m^2]": "dni",
        "Global Horiz [W/m^2]": "ghi",
        "Diffuse Horiz (calc) [W/m^2]": "dhi",
        "Dry Bulb Temp [deg C]": "temp_air",
        "Avg Wind Speed @ 30ft [m/s]": "wind_speed",
        "Avg Wind Direction @ 30ft [deg from N]": "wind_dir",
        "Peak Wind Speed @ 30ft [m/s]": "wind_speed_max",
        "Zenith Angle [degrees]": "zenith_angle",
        "Azimuth Angle [degrees]": "azimuth_angle",
        "Airmass": "airmass",
        "Global UVA [W/m^2]": "uva",
        "Global UVE [W/m^2]": "uve",
    },
    "ORNL": {
        "Global Horizontal [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "Air Temperature [deg C]": "temp_air",
        "Rel Humidity [%]": "rh",
        "Avg Wind Speed @ 42ft [m/s]": "wind_speed",
    },
    "NELHA": {
        "Global Horizontal [W/m^2]": "ghi",
        "Air Temperature [W/m^2]": "temp_air",
        "Avg Wind Speed @ 10m [m/s]": "wind_speed",
        "Rel Humidity [%]": "rh",
    },
    "ULL": {
        "Direct Normal [W/m^2]": "dni",
        "Global Horizontal [W/m^2]": "ghi",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "Air Temperature [deg C]": "temp_air",
        "Rel Humidity [%]": "rh",
        "Station Pressure [mBar]": "pressure",
        "Avg Wind Speed @ 3m [m/s]": "wind_speed",
        "Avg Wind Direction @ 3m [deg from N]": "wind_dir",
        "Precipitation [mm]": "precipitation",
    },
    "NWTC": {
        "Global Horizontal [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "Temperature @ 2m [deg C]": "temp_air",
        "Avg Wind Speed @ 2m [m/s]": "wind_speed",
        "Relative Humidity [%]": "rh",
    },
    "ESIF": {
        "Global Horizontal [W/m^2]": "ghi",
        "Air Temperature [deg C]": "temp_air",
        "Rel Humidity [%]": "rh",
        "Avg Wind Speed @ 0m [m/s]": "wind_speed",
        "Peak Wind Speed @ 0m [m/s]": "wind_speed_max",
        "Station Pressure [mBar]": "pressure",
        "Avg Wind Direction @ 0m [deg from N]": "wind_dir",
    },
    "SRRLASI": {
        "BRBG Total Cloud Cover [%]": "cloud_fraction",
        "CDOC Total Cloud Cover [%]": "cloud_fraction_total",
        "CDOC Thick Cloud Cover [%]": "cloud_fraction_thick",
        "CDOC Thin Cloud Cover [%]": "cloud_fraction_thin",
        "Sun Flag": "sun_flag",
        "HCF Value": "hcf",
    },
    "IRRSP": {
        "Global Horizontal [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "Air Temperature [deg C]": "temp_air",
        "Pressure [mBar]": "pressure",
        "Rel Humidity [%]": "rh",
        "Avg Wind Speed @ 19ft [m/s]": "wind_speed",
        "Avg Wind Direction @ 19ft [deg from N]": "wind_dir",
        "Peak Wind Speed @ 19ft [m/s]": "wind_speed_max",
        "Zenith Angle [degrees]": "zenith_angle",
        "Azimuth Angle [degrees]": "azimuth_angle",
        "Airmass": "airmass",
        "Precipitation [mm]": "precipitation",
    },
    "RAZON": {
        "Global Horizontal [W/m^2]": "ghi",
        "Direct Normal [W/m^2]": "dni",
        "Diffuse Horizontal [W/m^2]": "dhi",
        "PR1 Temperature [deg C]": "pr1_temp",
        "PH1 Temperature [deg C]": "ph1_temp",
        "Pressure [mBar]": "pressure",
        "Zenith Angle [degrees]": "zenith_angle",
        "Azimuth Angle [degrees]": "azimuth_angle",
    },
}


def download_midc(site: str, start: str, end: str, save_path: str):
    """
    Public high-level MIDC download interface.

    Parameters
    ----------
    site : str
        MIDC site ID, for example "BMS".
    start : str
        Start date in YYYYMMDD format.
    end : str
        End date in YYYYMMDD format.
    save_path : str
        Output CSV file path.
    """
    max_workers = 2
    site = site.upper().strip()

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    end_plus1 = end_dt + timedelta(days=1)

    months = []
    cur = start_dt.replace(day=1)
    while cur.date() <= end_dt.date():
        nxt = cur.replace(year=cur.year + 1, month=1, day=1) if cur.month == 12 else cur.replace(month=cur.month + 1, day=1)
        months.append((cur, nxt - timedelta(days=1)))
        cur = nxt

    base_url = "https://midcdmz.nlr.gov/apps/data_api.pl"
    print(f"[MIDC] site={site}, start={start}, end={end}")
    print(f"[MIDC] output={save_path}")

    def fast_read_csv(text: str) -> pd.DataFrame:
        try:
            return pd.read_csv(StringIO(text), sep=",", engine="c", on_bad_lines="skip")
        except Exception:
            return read_flexible_csv(text)

    def fetch_one(month_start: datetime, month_end: datetime) -> pd.DataFrame:
        text = http_get_with_retry(base_url, params={"site": site, "begin": month_start.strftime("%Y%m%d"), "end": month_end.strftime("%Y%m%d")})
        df = fast_read_csv(text)

        if df.empty or df.shape[1] < 4 or "Year" not in df.columns or "DOY" not in df.columns:
            return pd.DataFrame()

        time_col = df.columns[3]

        year_i = df["Year"].astype("int32")
        doy_i = df["DOY"].astype("int32")
        hhmm_i = df[time_col].astype("Int64").fillna(0).astype("int32")
        hours = (hhmm_i // 100).astype("int16")
        mins = (hhmm_i % 100).astype("int16")

        year_start = pd.to_datetime(year_i.astype(str) + "-01-01", format="%Y-%m-%d")
        naive_local = year_start + pd.to_timedelta(doy_i - 1, unit="D") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(mins, unit="m")

        mask = (naive_local >= start_dt) & (naive_local < end_plus1)
        if not mask.any():
            return pd.DataFrame()

        out = df.loc[mask].copy()
        out.insert(0, "_naive_local", naive_local.loc[mask].values)
        return out

    parts = []
    failed_months = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, ms, me): (ms, me) for (ms, me) in months}
        for fut in as_completed(futs):
            try:
                df_part = fut.result()
                if not df_part.empty:
                    parts.append((futs[fut][0], df_part))
            except Exception:
                failed_months += 1

    print(f"[MIDC] months={len(months)}, downloaded={len(parts)}, failed={failed_months}")

    if not parts:
        raise RuntimeError(f"No MIDC data available for {site} within the specified range ({start}~{end}).")

    parts.sort(key=lambda x: x[0])
    final_df = pd.concat([p[1] for p in parts], ignore_index=True)

    ns = pd.to_datetime(final_df["_naive_local"].values)
    ns = pd.Series(ns, index=final_df.index)

    key_min = ns.dt.strftime("%Y-%m-%d %H:%M")
    dup_mask = key_min.duplicated(keep=False)
    first_of_dup = ~key_min.duplicated(keep="first") & dup_mask

    ambiguous_vec = pd.Series(False, index=ns.index)
    ambiguous_vec[first_of_dup] = True

    local_ts = ns.dt.tz_localize("America/Denver", nonexistent="shift_forward", ambiguous=ambiguous_vec.values)
    final_df = final_df.drop(columns=["_naive_local"])
    final_df = attach_local_and_offset(final_df, local_ts)
    final_df = final_df.rename(columns={"Local_Time": "local_time", "UTC_Offset": "utc_offset"})

    utc_time_str = local_ts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")
    final_df.insert(0, "UTC_time", utc_time_str)
    if site in SITE_COLUMN_MAP:
        raw_keep = expand_unnamed_columns(final_df.columns) + [c for c in SITE_COLUMN_MAP[site] if c in final_df.columns]
        base_cols = ["UTC_time", "local_time", "utc_offset"]
        final_df = final_df[base_cols + raw_keep] if raw_keep else final_df[base_cols]
        final_df = final_df.rename(columns=SITE_COLUMN_MAP[site])
        final_df = final_df.loc[:, ~final_df.columns.duplicated(keep="last")]
    else:
        final_df = final_df[["UTC_time", "local_time", "utc_offset"]]

    mask, _ = clip_by_local_date(local_ts, start, end, "America/Denver")
    final_df = final_df[mask].reset_index(drop=True)
    final_df = final_df.loc[:, ~final_df.columns.str.contains(r"^Unnamed", regex=True)]
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[MIDC] rows={len(final_df)}")
    print(f"[MIDC] saved={save_path}")
