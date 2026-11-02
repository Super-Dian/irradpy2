"""
SURFRAD radiation data downloader.

This module provides a unified downloader and parser for NOAA SURFRAD radiation data.

Main capabilities
-----------------
- Download SURFRAD data by site and date range
- Apply quality-control filtering
- Generate standardized CSV output

Notes
-----
This module is built on top of shared utility functions defined in ``necessary_main.py``.
"""

import csv
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from .necessary_main import init_date, http_get_with_retry, url_contact

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def _build_surfrad_urls(site: str, start_date, end_date) -> List[str]:
    """Build a list of SURFRAD URLs for the given date range."""
    base = "https://gml.noaa.gov/aftp/data/radiation/surfrad"
    site = site.lower()
    today_utc = datetime.now(timezone.utc).date()
    urls = set()
    delta = (end_date - start_date).days + 1

    for i in range(delta):
        d = start_date + timedelta(days=i)
        yy = d.strftime("%y")
        year = d.year
        doy = d.timetuple().tm_yday
        fname = f"{site}{yy}{doy:03d}.dat"

        if d >= today_utc - timedelta(days=2):
            u = url_contact(base, "realtime")
            u = url_contact(u, site)
            u = url_contact(u, str(year))
            u = url_contact(u, fname)
            urls.add(u)

        if d <= today_utc - timedelta(days=1):
            u = url_contact(base, site)
            u = url_contact(u, str(year))
            u = url_contact(u, fname)
            urls.add(u)

    return sorted(urls)


def _print_progress(current: int, total: int, downloaded: int, failed: int) -> None:
    width = 30
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "=" * filled + "." * (width - filled)
    print(f"\r[SURFRAD] [{bar}] {current}/{total} ({ratio * 100:5.1f}%) downloaded={downloaded} failed={failed}", end="", flush=True)
    if current >= total:
        print()


def _parse_surfrad_line(fields: List[str]) -> Optional[Dict]:
    """
    Parse a single SURFRAD data row.

    The field positions match the NOAA SURFRAD .dat format.
    QC logic:
        value = raw_value if (qc == 0 and value != -9999.9)
                else -999.9
    """
    try:
        year = int(fields[0])
        month = int(fields[2])
        day = int(fields[3])
        hour = int(fields[4])
        minute = int(fields[5])
        utc_dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        record = {"UTC_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S")}

        var_map = [
            ("ghi", 8), ("shortwave_up", 10), ("dni", 12), ("dhi", 14),
            ("longwave_down", 16), ("dw_case_temp", 18), ("dw_dome_temp", 20),
            ("longwave_up", 22), ("uw_case_temp", 24), ("uw_dome_temp", 26),
            ("uvb", 28), ("par", 30), ("net_solar", 32), ("longwave_net", 34),
            ("total_net", 36), ("temp_air", 38), ("rh", 40), ("wind_speed", 42),
            ("wind_dir", 44), ("pressure", 46),
        ]

        for key, idx in var_map:
            val = float(fields[idx])
            qc = int(fields[idx + 1])
            record[key] = val if (qc == 0 and val != -9999.9) else -999.9

        return record
    except Exception:
        return None


def download_surfrad(site: str, start: str, end: str, save_path: str):
    """
    Public SURFRAD download function (replaces fetch_surfrad_data).

    Parameters
    ----------
    site : str
        SURFRAD site ID (e.g., "bon", "dra", "fpk").
    start : str
        Start date (YYYY-MM-DD or YYYYMMDD).
    end : str
        End date (YYYY-MM-DD or YYYYMMDD).
    save_path : str, optional
        Output CSV path. If None, the filename is auto-generated.
    """
    start_date = init_date(start)
    end_date = init_date(end)
    if not save_path:
        save_path = f"surfrad_{start_date}_{end_date}.csv"

    site = site.lower()
    urls = _build_surfrad_urls(site, start_date, end_date)
    print(f"[SURFRAD] site={site}, start={start_date.strftime('%Y%m%d')}, end={end_date.strftime('%Y%m%d')}")
    print(f"[SURFRAD] output={save_path}")

    all_rows = []
    failed_files = 0
    downloaded_files = 0
    headers = {"User-Agent": "Mozilla/5.0"}
    _print_progress(0, len(urls), downloaded_files, failed_files)

    for i, url in enumerate(urls, start=1):
        success = False
        try:
            text = http_get_with_retry(url, headers=headers, timeout=20, max_retries=3)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) >= 3:
                site_name = lines[0]
                meta_info = lines[1].split()
                file_rows = 0
                for ln in lines[2:]:
                    rec = _parse_surfrad_line(ln.split())
                    if rec:
                        all_rows.append(rec)
                        file_rows += 1
                if file_rows > 0:
                    downloaded_files += 1
                    success = True
        except Exception:
            pass
        if not success:
            failed_files += 1
        _print_progress(i, len(urls), downloaded_files, failed_files)
        time.sleep(0.25)

    seen = set()
    dedup = []
    for r in all_rows:
        k = r["UTC_time"]
        if k not in seen:
            seen.add(k)
            dedup.append(r)
    dedup.sort(key=lambda x: x["UTC_time"])

    csv_cols = [
        "UTC_time", "ghi", "shortwave_up", "dni", "dhi", "longwave_down", "dw_case_temp",
        "dw_dome_temp", "longwave_up", "uw_case_temp", "uw_dome_temp", "uvb", "par",
        "net_solar", "longwave_net", "total_net", "temp_air", "rh", "wind_speed", "wind_dir", "pressure",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        w.writerows(dedup)

    print(f"[SURFRAD] files={len(urls)}, downloaded={downloaded_files}, failed={failed_files}")
    print(f"[SURFRAD] rows={len(dedup)}")
    print(f"[SURFRAD] saved={save_path}")