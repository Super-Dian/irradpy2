"""Example 3: DNI/DHI separation and GHI persistence forecasting.

Separates a recent SURFRAD day into DNI/DHI with Engerer2, evaluates the
separation against the measured components, then forecasts GHI with a
persistence model. Reuses the ./data/ cache when present.
"""

from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from irradpy2.forecast import forecast_persistence
from irradpy2.getirrad import download_surfrad
from irradpy2.separation import engerer2
from irradpy2.tool import cal_metrics, filter_sep_data, merge_columns, plot_columns, read_csv

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

observations = read_csv(csv_path).replace(-999.9, np.nan)
data = observations.dropna(subset=["ghi", "dni", "dhi"])
assert not data.empty, f"{csv_path} has no usable rows for separation; delete it and re-run"

datetime = pd.to_datetime(data["UTC_time"]).to_numpy()

# SURFRAD bon is 1-minute data -> averaging_period=1 (minutes)
kd, dhi_pred, dni_pred = engerer2(data["ghi"].to_numpy(), datetime, LAT, LON,
                                  averaging_period=1)
data = merge_columns(data, [np.ravel(dni_pred), np.ravel(dhi_pred)],
                     ["dni_pred", "dhi_pred"])

sep = filter_sep_data(data, LAT, LON, time_col="UTC_time", ghi_col="ghi",
                      dni_col="dni", dhi_col="dhi",
                      pred_cols=["dni_pred", "dhi_pred"])
print(f"site={SITE} day={day} separation evaluation rows={len(sep)}")
print("Engerer2 vs measured:",
      {k: round(float(v), 1) for k, v in
       cal_metrics(sep, obs_col="dni", pred_col="dni_pred").items()},
      "(DNI)")

# Forecast using all available GHI, including rows with missing DNI or DHI.
_, future = forecast_persistence(
    observations["UTC_time"], observations["ghi"], forecast_duration="6min"
)
print(future.to_string(index=False))
plot_columns(future, ["pre_Global"], labels=["Persistence forecast"], x_col="Timestamp",
             title=f"SURFRAD {SITE} {day} GHI persistence forecast",
             ylabel="GHI (W m$^{-2}$)")
