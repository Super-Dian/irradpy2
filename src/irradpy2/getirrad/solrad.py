"""
SOLRAD radiation data downloader.

This module provides a unified downloader and parser for NOAA SOLRAD radiation data.

Main capabilities
-----------------
- Download SOLRAD data by site and date range
- Apply quality-control filtering
- Generate standardized CSV output

Notes
-----
This module is built on top of shared utility functions defined in ``necessary_main.py``.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List
import pandas as pd

from .necessary_main import http_get_with_retry, init_date, url_contact, attach_local_and_offset

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Field index mapping following the NOAA SOLRAD file layout.
FIELD_MAP = {
    "dw_psp": (8, 9),    # global PSP
    "direct": (10, 11),  # direct beam
    "diffuse": (12, 13), # diffuse
    "uvb": (14, 15),     # UV-B
    "uvb_temp": (16, 17),# UV-B temperature
}

# Standard deviation fields.
STD_FIELDS = {
    "std_dw_psp": 18,
    "std_direct": 19,
    "std_diffuse": 20,
    "std_uvb": 21,
}


def _parse_meta_line(line: str) -> dict:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid metadata line: {line}")

    meta = {"lat": float(parts[0]), "lon": float(parts[1]), "elev": float(parts[2]), "tz_hours": int(parts[3])}
    for i, p in enumerate(parts):
        if p.lower().startswith("version") and i + 1 < len(parts):
            meta["version"] = parts[i + 1]
    return meta


def _build_solrad_urls(station: str, start, end) -> List[str]:
    base = "https://gml.noaa.gov/aftp/data/radiation/solrad"
    today_utc = datetime.now(timezone.utc).date()
    urls = []
    total_days = (end - start).days + 1

    for i in range(total_days):
        d = start + timedelta(days=i)
        yy = d.strftime("%y")
        year = str(d.year)
        doy = f"{d.timetuple().tm_yday:03d}"
        fname = f"{station}{yy}{doy}.dat"

        if d >= today_utc - timedelta(days=2):
            u = url_contact(base, "realtime")
            u = url_contact(u, station)
            u = url_contact(u, year)
            u = url_contact(u, fname)
            urls.append(u)

        if d <= today_utc - timedelta(days=1):
            u = url_contact(base, station)
            u = url_contact(u, year)
            u = url_contact(u, fname)
            urls.append(u)

    return urls


def _parse_data_block(lines: List[str], tz_hours: int) -> pd.DataFrame:
    if not lines:
        return pd.DataFrame()

    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 22:
            continue
        rows.append(parts[:22])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(22)])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    utc_dt = pd.to_datetime(df["c0"].astype(int).astype(str) + "-" + df["c2"].astype(int).astype(str) + "-" + df["c3"].astype(int).astype(str) + " " + df["c4"].astype(int).astype(str) + ":" + df["c5"].astype(int).astype(str), errors="coerce")
    local_dt = utc_dt + pd.Timedelta(hours=tz_hours)
    df["UTC_Time"] = utc_dt.dt.strftime("%Y/%m/%d %H:%M:%S")

    for key, (val_i, qc_i) in FIELD_MAP.items():
        raw = df[f"c{val_i}"]
        qc = df[f"c{qc_i}"]
        df[key] = raw.where((qc == 0) & (raw > -9000), -999.9)

    for key, idx in STD_FIELDS.items():
        df[key] = df[f"c{idx}"].astype(float)

    df = attach_local_and_offset(df, local_ts=local_dt.dt.tz_localize(f"Etc/GMT{(-tz_hours):+d}"), local_col="Local_Time", offset_col="UTC_Offset")
    df = df.sort_values("UTC_Time")
    return df[["UTC_Time", "Local_Time", "UTC_Offset"] + list(FIELD_MAP.keys()) + list(STD_FIELDS.keys())]


def _print_progress(current: int, total: int, downloaded: int, failed: int) -> None:
    width = 30
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "=" * filled + "." * (width - filled)
    print(f"\r[SOLRAD] [{bar}] {current}/{total} ({ratio * 100:5.1f}%) downloaded={downloaded} failed={failed}", end="", flush=True)
    if current >= total:
        print()


def download_solrad(site: str, start: str, end: str, save_path: str) -> None:
    """
    Download SOLRAD data for a given site and save the processed result as a CSV file.

    This is the public high-level SOLRAD download interface.

    Parameters
    ----------
    site : str
        SOLRAD station identifier.
        See the NOAA SOLRAD page for available station information:
        https://gml.noaa.gov/grad/solrad/
    start : str
        Start date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
    end : str
        End date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
    save_path : str
        Absolute output CSV file path.

    Returns
    -------
    None
        The merged table is written to ``save_path``.

    Raises
    ------
    ValueError
        When ``save_path`` is empty or not an absolute path.

    References
    ----------
    NOAA SOLRAD homepage:
    https://gml.noaa.gov/grad/solrad/

    NOAA SOLRAD data access:
    https://gml.noaa.gov/aftp/data/radiation/solrad/
    """
    if not save_path or not os.path.isabs(save_path):
        raise ValueError("save_path must be a non-empty absolute path.")

    start_d = init_date(start)
    end_d = init_date(end)
    station = site.lower().strip()
    urls = _build_solrad_urls(station, start_d, end_d)
    frames = []
    downloaded_files = 0
    failed_files = 0

    print(f"[SOLRAD] site={station}, start={start_d.strftime('%Y%m%d')}, end={end_d.strftime('%Y%m%d')}")
    print(f"[SOLRAD] output={save_path}")
    _print_progress(0, len(urls), downloaded_files, failed_files)

    for i, url in enumerate(urls, start=1):
        success = False
        try:
            text = http_get_with_retry(url)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) >= 3:
                meta = _parse_meta_line(lines[1])
                tz_hours = meta["tz_hours"]
                df = _parse_data_block(lines[2:], tz_hours)
                if not df.empty:
                    frames.append(df)
                    downloaded_files += 1
                    success = True
        except Exception:
            pass
        if not success:
            failed_files += 1
        _print_progress(i, len(urls), downloaded_files, failed_files)
        time.sleep(0.4)

    if not frames:
        print("[SOLRAD] No data available")
        return

    df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["UTC_Time"]).sort_values("UTC_Time")
    df_all.to_csv(save_path, index=False, encoding="utf-8")
    print(f"[SOLRAD] files={len(urls)}, downloaded={downloaded_files}, failed={failed_files}")
    print(f"[SOLRAD] rows={len(df_all)}")
    print(f"[SOLRAD] saved={save_path}")