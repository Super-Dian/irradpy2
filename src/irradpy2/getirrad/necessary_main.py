# necessary_main.py
"""
General utility functions for:
- HTTP requests
- CSV parsing
- time and timezone handling
- date conversion
"""

from datetime import datetime, timedelta
import pytz
import time
import csv
from io import StringIO
from typing import Optional, Sequence, Tuple
import requests
import pandas as pd
import os
# ======================
# HTTP & CSV
# ======================
def http_get_with_retry(url: str,
                        params: Optional[dict] = None,
                        headers: Optional[dict] = None,
                        timeout: int = 30,
                        max_retries: int = 3,
                        backoff: float = 1.5) -> str:
    """
    Send an HTTP GET request with retry support and return the response text.

    Parameters
    ----------
    url : str
        Request URL.
    params : dict, optional
        Query-string parameters.
    headers : dict, optional
        Extra headers merged over the built-in defaults.
    timeout : int, default 30
        Socket timeout in seconds (the connect timeout is capped at 10 s).
    max_retries : int, default 3
        Number of attempts before the last exception is re-raised.
    backoff : float, default 1.5
        Base for the exponential sleep between retries.

    Returns
    -------
    str
        Response text with any leading BOM stripped.

    Raises
    ------
    requests.RequestException
        Re-raised from the final attempt when all retries fail.
    """
    hdr = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    if headers:
        hdr.update(headers)

    last_exc = None
    for i in range(max_retries):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=hdr,
                timeout=(min(10, timeout // 2 or 5), timeout),
            )
            resp.raise_for_status()
            if not resp.encoding:
                resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text.lstrip("\ufeff")
        except Exception as e:
            last_exc = e
            if i < max_retries - 1:
                sleep_s = (backoff ** i) * (1 + 0.1 * (i % 3))
                print(f"[warn] HTTP failed, retry {i+1}, waiting {sleep_s:.1f}s... Details: {e}")
                time.sleep(sleep_s)
    raise last_exc


def read_flexible_csv(text: str) -> pd.DataFrame:
    """
    Parse CSV text into a DataFrame with flexible delimiter handling.

    Parameters
    ----------
    text : str
        Raw CSV (or delimited) text.

    Returns
    -------
    pandas.DataFrame
        Parsed table; delimiter is sniffed when the fast comma parse fails,
        and BOM-like artifacts are stripped from column names.
    """
    raw = StringIO(text)
    try:
        df_fast = pd.read_csv(raw, sep=",", engine="c", on_bad_lines="skip")
        df_fast.columns = [c.replace("\ufeff", "").replace("ï»¿", "") for c in df_fast.columns]
        return df_fast
    except Exception:
        pass

    raw.seek(0)
    try:
        sample = raw.read(4096)
        raw.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = None
        raw.seek(0)

    try:
        if sep:
            head_df = pd.read_csv(raw, nrows=0, sep=sep, engine="python")
        else:
            head_df = pd.read_csv(raw, nrows=0, sep=r'[,\t]+|\s{2,}', engine="python")
        ncols = len(head_df.columns)
    except Exception:
        raw.seek(0)
        head_df = pd.read_csv(raw, nrows=0, sep=",", engine="python")
        ncols = len(head_df.columns)
    raw.seek(0)

    df = pd.read_csv(
        raw,
        usecols=range(ncols),
        sep=(sep if sep else r'[,\t]+|\s{2,}'),
        engine="python",
        on_bad_lines="skip",
        skip_blank_lines=True,
    )
    df.columns = [c.replace("\ufeff", "").replace("ï»¿", "") for c in df.columns]
    return df

# ======================
# Time & localization
# ======================
def tz_localize_safely(naive_ts: pd.Series,
                       tz: str,
                       nonexistent: str = "shift_forward",
                       ambiguous: str = "infer") -> pd.Series:
    """
    Localize naive timestamps to a timezone-aware series.

    Parameters
    ----------
    naive_ts : pandas.Series
        Timezone-naive datetime series.
    tz : str
        Target timezone name.
    nonexistent : str, default "shift_forward"
        Policy for nonexistent local times.
    ambiguous : str, default "infer"
        Policy for ambiguous local times.

    Returns
    -------
    pandas.Series
        Timezone-aware series.
    """
    return naive_ts.dt.tz_localize(tz, nonexistent=nonexistent, ambiguous=ambiguous)


def format_local_str(local_ts: pd.Series,
                     fmt: str = "%Y/%m/%d %H:%M:00") -> pd.Series:
    """
    Format a timestamp series as strings.

    Parameters
    ----------
    local_ts : pandas.Series
        Datetime series.
    fmt : str, default "%Y/%m/%d %H:%M:00"
        strftime format string.

    Returns
    -------
    pandas.Series
        Formatted strings.
    """
    return local_ts.dt.strftime(fmt)


def attach_local_and_offset(df: pd.DataFrame,
                            local_ts: pd.Series,
                            local_col: str = "Local_Time",
                            offset_col: str = "UTC_Offset") -> pd.DataFrame:
    """
    Add local time and UTC offset columns to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table; copied before columns are added.
    local_ts : pandas.Series
        Timezone-aware local timestamps aligned with ``df``.
    local_col : str, default "Local_Time"
        Name of the formatted local-time column.
    offset_col : str, default "UTC_Offset"
        Name of the UTC-offset column (format "+HH:MM").

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the two new columns placed first.
    """
    local_str = local_ts.dt.strftime("%Y/%m/%d %H:%M:00")
    z = local_ts.dt.strftime("%z").str.replace(r"([+-]\d{2})(\d{2})", r"\1:\2", regex=True)

    out = df.copy()
    out = out.assign(**{local_col: local_str, offset_col: z})
    cols = [local_col, offset_col] + [c for c in out.columns if c not in (local_col, offset_col)]
    return out[cols]


def clip_by_local_date(local_ts: pd.Series,
                       begin: str,
                       end: str,
                       tz: str) -> Tuple[pd.Series, pd.Series]:
    """
    Filter timestamps by a local date range.

    Parameters
    ----------
    local_ts : pandas.Series
        Timezone-aware local timestamps.
    begin : str
        Inclusive start date in ``YYYYMMDD`` format.
    end : str
        Inclusive end date in ``YYYYMMDD`` format (the window extends to the
        end of that day).
    tz : str
        Timezone used to localize the ``begin``/``end`` bounds.

    Returns
    -------
    mask : pandas.Series of bool
        Boolean mask over ``local_ts``.
    filtered : pandas.Series
        The timestamps inside the window.
    """
    beg = pd.to_datetime(begin, format="%Y%m%d").tz_localize(tz)
    endt = (pd.to_datetime(end, format="%Y%m%d") + pd.Timedelta(days=1)).tz_localize(tz)
    mask = (local_ts >= beg) & (local_ts < endt)
    return mask, local_ts[mask]

# ======================
# Other utilities
# ======================
def expand_unnamed_columns(actual_cols: Sequence[str],
                           prefix: str = "Unnamed:") -> list:
    """
    Return all column names that start with the given prefix.

    Parameters
    ----------
    actual_cols : sequence of str
        Column names to filter.
    prefix : str, default "Unnamed:"
        Prefix to match.

    Returns
    -------
    list of str
        Matching column names, in input order.
    """
    return [c for c in actual_cols if c.startswith(prefix)]


def init_date(date_str):
    """
    Parse a date string into a date object.

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD``, ``YYYY.MM.DD``, ``YYYY-MM-DD`` or
        ``YYYY/MM/DD`` format.

    Returns
    -------
    datetime.date
        Parsed date.

    Raises
    ------
    ValueError
        When the string matches none of the supported formats.
    """
    formats = ["%Y%m%d", "%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}")


def init_str_date(date_str):
    """
    Parse a date string and return it in YYYY-MM-DD format.

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD``, ``YYYY.MM.DD``, ``YYYY-MM-DD`` or
        ``YYYY/MM/DD`` format.

    Returns
    -------
    str
        Date normalized to ``YYYY-MM-DD``.

    Raises
    ------
    ValueError
        When the string matches none of the supported formats.
    """
    formats = ["%Y%m%d", "%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"]
    for fmt in formats:
        try:
            date = datetime.strptime(date_str, fmt).date()
            return datetime.strftime(date, '%Y-%m-%d')
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}")


def get_yesterday_utc_minus_7():
    """
    Return yesterday's date in the America/Denver timezone.

    Returns
    -------
    datetime.date
        Calendar date one day before "today" in America/Denver (UTC-6/-7).
    """
    utc_minus_7 = pytz.timezone('America/Denver')
    now_utc_minus_7 = datetime.now(utc_minus_7)
    yesterday_utc_minus_7 = now_utc_minus_7 - timedelta(1)
    return yesterday_utc_minus_7.date()


def url_contact(url, content):
    """
    Join a base URL and path segment safely.

    Parameters
    ----------
    url : str
        Base URL; trailing slashes are stripped.
    content : str
        Path segment; surrounding slashes are stripped.

    Returns
    -------
    str
        ``url + '/' + content`` with exactly one separator.
    """
    url = url.rstrip("/")
    content = str(content).strip("/")
    return f"{url}/{content}"


def ensure_dir(path: str):
    """
    Create the directory if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path; ignored when empty. Parent directories are created
        as needed.

    Returns
    -------
    None
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def fmt_offset(hours: float) -> str:
    """
    Format a timezone offset as +HH:MM or -HH:MM.

    Parameters
    ----------
    hours : float
        Offset in decimal hours (e.g. ``-7.0`` or ``5.5``).

    Returns
    -------
    str
        Offset string such as ``"+05:30"`` or ``"-07:00"``.
    """
    sign = "+" if hours >= 0 else "-"
    h = abs(int(hours))
    m = abs(int((hours - int(hours)) * 60))
    return f"{sign}{h:02d}:{m:02d}"

