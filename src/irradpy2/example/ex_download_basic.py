"""Example 1: credential-free download, data preview, and a first plot.

Downloads one recent day of SURFRAD data (no account required), cleans the
-999.9 QC sentinels, computes the solar zenith angle, and plots the GHI
series. Data is cached in ./data/ so re-runs skip the download.
"""

from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from irradpy2.getirrad import download_surfrad
from irradpy2.tool import cal_zen, plot_columns, read_csv

SITE, LAT, LON = "bon", 40.05, -88.37  # SURFRAD Bondville, IL
DATA_DIR = Path("data")  # written under the current working directory


def latest_archive_day(site: str, lookback: int = 14) -> str:
    """Return the most recent YYYYMMDD available in the SURFRAD archive."""
    for back in range(2, 2 + lookback):
        day = date.today() - timedelta(days=back)
        url = (f"https://gml.noaa.gov/aftp/data/radiation/surfrad/"
               f"{site}/{day.year}/{site}{day:%y%j}.dat")
        try:
            urlopen(Request(url, method="HEAD"), timeout=10).close()
            return day.strftime("%Y%m%d")
        except (HTTPError, URLError):
            continue  # archive publication lags a few days; try an earlier day
    raise RuntimeError(f"no SURFRAD {site} data in the last {lookback} days")


day = latest_archive_day(SITE)
DATA_DIR.mkdir(exist_ok=True)
csv_path = DATA_DIR / f"surfrad_{SITE}_{day}.csv"
if not csv_path.exists():  # cache: skip the download on re-runs
    download_surfrad(SITE, day, day, str(csv_path))

data = read_csv(csv_path).replace(-999.9, np.nan).dropna(subset=["ghi", "dni", "dhi"])
assert not data.empty, f"{csv_path} has no usable rows; delete it and re-run"

data = cal_zen(data, LAT, LON)  # time column auto-detected (UTC_time)
print(f"site={SITE} day={day} rows={len(data)} zenith max={data.zen.max():.1f} deg")
print(data[["UTC_time", "ghi", "dni", "dhi", "zen"]].head())
print(data[["ghi", "dni", "dhi"]].describe().round(1))

plot_columns(data, ["ghi"], x_col="UTC_time", title=f"SURFRAD {SITE} {day} GHI",
             ylabel="GHI (W m$^{-2}$)")
