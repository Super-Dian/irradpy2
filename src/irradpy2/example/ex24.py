import numpy as np
import pandas as pd

from bsrn import download_bsrn
from pre import forecast_lstm
from engerer2 import engerer2
from tool import read_csv, clean_negative, calculate_metrics, build_time_array, filter_separation_evaluation, plot_two_series


# 1. Download BSRN ground observation data
lat, lon = 47.37, 123.92
begin, end = "2025-01-01", "2025-01-31"
bsrn_path = r"D:\example_down\BSRN_QIQ_20250101_20250131.csv"

# download_bsrn(site="QIQ", begin=begin, end=end, username="bsrnftp", password="bsrn1", save_path=bsrn_path)

# 2. Read data and forecast GHI using LSTM
data = read_csv(bsrn_path).replace([-999.9, -9999, -999, -99], np.nan)
data = clean_negative(data, cols=["ghi", "dni", "dhi"], lower=0)

datetime, ghi = data["UTC"], data["ghi"]
ori_data, pre_data = forecast_lstm(datetime, ghi, seq_len=72, horizon=6, epochs=10, batch_size=64)

# 3. Prepare observation and prediction data
base_data = pd.DataFrame({
    "Timestamp": pd.to_datetime(data["UTC"], errors="coerce"),
    "Direct": pd.to_numeric(data["dni"], errors="coerce"),
    "Diffuse": pd.to_numeric(data["dhi"], errors="coerce")
})

base_data = base_data.dropna(subset=["Timestamp"])
base_data[["Direct", "Diffuse"]] = base_data[["Direct", "Diffuse"]].ffill().bfill().clip(lower=0)
base_data = base_data.sort_values("Timestamp").reset_index(drop=True)

result_data = pd.merge(ori_data, pre_data, on="Timestamp", how="inner")
result_data = pd.merge(result_data, base_data, on="Timestamp", how="left")
result_data["pre_Global_for_engerer2"] = result_data["pre_Global"].clip(lower=0)

result_data, _ = build_time_array(result_data, time_col="Timestamp", required_cols=["Global", "pre_Global", "pre_Global_for_engerer2", "Direct", "Diffuse"])

# 4. Run Engerer2 using observed and predicted GHI
for suffix, ghi_col in {"obsghi": "Global", "predghi": "pre_Global_for_engerer2"}.items():
    kd, dhi, dni = engerer2(result_data[ghi_col], result_data["Timestamp"], lat, lon, averaging_period=1)
    result_data[f"kd_engerer2_{suffix}"] = np.asarray(kd).reshape(-1)
    result_data[f"dhi_engerer2_{suffix}"] = np.asarray(dhi).reshape(-1)
    result_data[f"dni_engerer2_{suffix}"] = np.asarray(dni).reshape(-1)

# 5. Print metrics
ghi_metrics_df = pd.DataFrame({
    "GHI_LSTM_all_samples": calculate_metrics(result_data, obs_col="Global", pred_col="pre_Global")
}).T

sep_eval_data = filter_separation_evaluation(
    result_data,
    latitude=lat,
    longitude=lon,
    time_col="Timestamp",
    zen_col="zen",
    ghi_col="Global",
    dni_col="Direct",
    dhi_col="Diffuse",
    pred_cols=["dhi_engerer2_obsghi", "dni_engerer2_obsghi", "dhi_engerer2_predghi", "dni_engerer2_predghi"],
    max_zen=85,
    min_ghi=50,
    require_positive=True
)

separation_metrics_df = pd.DataFrame({
    "DHI_Engerer2_obsGHI": calculate_metrics(sep_eval_data, obs_col="Diffuse", pred_col="dhi_engerer2_obsghi"),
    "DNI_Engerer2_obsGHI": calculate_metrics(sep_eval_data, obs_col="Direct", pred_col="dni_engerer2_obsghi"),
    "DHI_Engerer2_predGHI": calculate_metrics(sep_eval_data, obs_col="Diffuse", pred_col="dhi_engerer2_predghi"),
    "DNI_Engerer2_predGHI": calculate_metrics(sep_eval_data, obs_col="Direct", pred_col="dni_engerer2_predghi")
}).T

print("\nGHI prediction metrics")
print(ghi_metrics_df.round(3))

print("\nDHI/DNI separation metrics")
print(separation_metrics_df.round(3))

# 6. Plot GHI prediction
plot_two_series(result_data, obs_col="Global", pred_col="pre_Global", obs_label="Observed GHI", pred_label="Predicted GHI", title="LSTM GHI Prediction", ylabel=r"GHI [W m$^{-2}$]", threshold=20, max_value=1400)