"""Example 2: clear-sky modelling and clear-sky detection (CSD).

Builds a TJ-1957 clear-sky GHI curve for a recent SURFRAD day, then runs the
Ellis-2018 clear-sky detection method against the observations. Reuses the
./data/ cache from ex_download_basic.py when present.
"""

from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from irradpy2.clear_sky import tj
from irradpy2.csd import Ellis2018CSD
from irradpy2.getirrad import download_surfrad
from irradpy2.tool import cal_zen, merge_columns, plot_csd, read_csv

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

data = cal_zen(data, LAT, LON)
datetime = pd.to_datetime(data["UTC_time"]).to_numpy()
daytime = data["zen"].to_numpy() < 90  # clear-sky models are undefined at night

# tj expects the zenith angle in radians and returns clear-sky [GHI, DNI, DHI]
ghics = np.full(len(data), np.nan)
ghics[daytime] = np.ravel(tj(np.radians(data.loc[daytime, "zen"]), datetime[daytime])[0])
data = merge_columns(data, [ghics], ["ghi_clear"])

csd = np.zeros(len(data))  # Ellis2018: 1 = clear sky, 0 = not clear
csd[data.index.get_indexer(data[daytime].index)] = np.ravel(
    Ellis2018CSD(data.loc[daytime, "ghi"].to_numpy(), ghics[daytime]))
data["csd"] = csd

print(f"site={SITE} day={day} daytime rows={daytime.sum()}")
print(f"clear-sky fraction (daytime): {csd[daytime].mean():.2%}")
plot_csd(data, ghi_col="ghi", clear_col="ghi_clear", csd_col="csd")
