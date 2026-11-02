# example_download_and_plot.py

from bsrn import download_bsrn
from merra2 import download_merra2
from cams import download_cams
from tool import read_csv, select_series, resample_series, merge_series, plot_columns

site, lat, lon = "cab", 51.968, 4.928
start, end = "2024-01-01", "2024-01-01"
save_dir = r"D:\example_down"

bsrn_path = fr"{save_dir}\cab_bsrn.csv"
merra2_path = fr"{save_dir}\cab_merra2_rad.csv"
cams_path = fr"{save_dir}\cab_cams_point.csv"

# download_bsrn(site=site, begin=start, end=end, username="your_bsrn_username", password="your_bsrn_password", save_path=bsrn_path)
# download_merra2(collection_names="rad", mode="point", lat=lat, lon=lon, start=start, end=end, username="your_earthdata_username", password="your_earthdata_password", save_path=merra2_path)
# download_cams(mode="point", lat=lat, lon=lon, start=start, end=end, save_path=cams_path, time_step="15min", api_url="https://ads.atmosphere.copernicus.eu/api", api_key="your_ads_api_key")

bsrn = resample_series(read_csv(bsrn_path)[["UTC", "ghi"]], time_col="UTC", cols=["ghi"], freq="1h", agg="mean", lower_limit=0, label_offset="30min").rename(columns={"ghi": "bsrn_ghi"})
merra2 = select_series(read_csv(merra2_path)[["time", "ghi"]], time_col="time", cols=["ghi"], lower_limit=0).rename(columns={"ghi": "merra2_ghi"})
cams = resample_series(read_csv(cams_path)[["time", "ghi"]], time_col="time", cols=["ghi"], freq="1h", agg="sum", lower_limit=0, label_offset="30min").rename(columns={"ghi": "cams_ghi"})

data = merge_series([bsrn, merra2, cams], time_col="UTC", dropna=True)

plot_columns(data, y_cols=["bsrn_ghi", "merra2_ghi", "cams_ghi"], labels=["BSRN GHI", "MERRA-2 GHI", "CAMS GHI"], title="GHI comparison from BSRN, MERRA-2 and CAMS", ylabel=r"GHI (W m$^{-2}$)", x_col="UTC")