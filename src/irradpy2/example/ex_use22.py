import numpy as np

from midc import download_midc
from merra2 import download_merra2
from tool import read_csv, prepare_clearsky, merge_columns, plot_clearsky
from TJ import TJ
from Simplified_Solis import Simplified_Solis
from rest2v5 import rest2v5
from Ellis2018CSD import Ellis2018CSD


# 1. Download MIDC ground observation and MERRA-2 auxiliary data
lat, lon = 39.742, -105.18
begin, end = "2023-01-01", "2023-01-01"

bms_path = r"D:\example_down\MIDC_BMS_20230101_20230102.csv"
merra2_path = r"D:\example_down\bms_merra2_clear_sky.csv"

# download_midc(site="BMS", begin=begin, end=end, save_path=bms_path)
# download_merra2(collection_names=["rad", "slv", "aer"], mode="point", lat=lat, lon=lon, begin=begin, end=end, variables={"rad": ["ALBEDO", "SWGDN", "SWGDNCLR"], "slv": ["PS", "TQV", "TO3"], "aer": ["TOTEXTTAU", "TOTANGSTR"]}, username="your_username", password="your_password", save_path=merra2_path)

# 2. Read data and prepare clear-sky model inputs
midc_data = read_csv(bms_path)
merra2_data = read_csv(merra2_path)
data = prepare_clearsky(midc_data, merra2_data, lat, lon, begin, end)

# 3. Keep daytime samples for clear-sky modeling and detection
# 4. Extract required variables
datetime = data["UTC"]
ghi = data["ghi"]
sza = data["sza"]
press = data["press"]
albedo = data["albedo"]
wv = data["wv"]
aod700 = data["aod700"]
ang_alpha = data["ang_alpha"]
ang_beta = data["ang_beta"]
ozone = data["ozone_input"]
NO2 = 2e-4

# 5. Run clear-sky irradiance models
ghi_tj, dni_tj, dhi_tj = TJ(sza, datetime)
ghi_solis, dni_solis, dhi_solis = Simplified_Solis(sza, datetime, press, aod700, wv)
ghi_rest2v5, dni_rest2v5, dhi_rest2v5 = rest2v5(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, NO2, wv)

# 6. Detect clear-sky periods
csd_ellis2018 = Ellis2018CSD(ghi, ghi_rest2v5)

# 7. Merge model outputs and plot clear-sky model comparison
data = merge_columns(
    data,
    values=[ghi_tj, dni_tj, dhi_tj, ghi_solis, dni_solis, dhi_solis, ghi_rest2v5, dni_rest2v5, dhi_rest2v5],
    columns=["ghi_clear_tj", "dni_clear_tj", "dhi_clear_tj", "ghi_clear_solis", "dni_clear_solis", "dhi_clear_solis", "ghi_clear_rest2v5", "dni_clear_rest2v5", "dhi_clear_rest2v5"]
)

plot_clearsky(
    data,
    model_cols={
        "TJ": ("ghi_clear_tj", "dni_clear_tj", "dhi_clear_tj"),
        "Simplified Solis": ("ghi_clear_solis", "dni_clear_solis", "dhi_clear_solis"),
        "REST2v5": ("ghi_clear_rest2v5", "dni_clear_rest2v5", "dhi_clear_rest2v5")
    },
    x_col="UTC"
)