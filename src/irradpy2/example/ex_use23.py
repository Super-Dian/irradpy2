import numpy as np
import pandas as pd

from bsrn import download_bsrn
from pre import forecast_lstm
from engerer2 import engerer2
from tool import read_csv, cal_metrics, plot_series


# 1. Download BSRN ground observation data
lat, lon = 47.37, 123.92
begin, end = "2025-01-01", "2025-01-31"
bsrn_path = r"D:\example_down\BSRN_QIQ_20250101_20250131.csv"
out_path = r"D:\example_down\BSRN_QIQ_20250101_20250131_lstm_engerer2_test.csv"

# download_bsrn(site="QIQ", begin=begin, end=end, username="bsrnftp", password="bsrn1", save_path=bsrn_path)

# 2. Read data and forecast GHI using LSTM
data = read_csv(bsrn_path)
datetime, ghi = data["UTC"], data["ghi"]
ori_data, pre_data = forecast_lstm(datetime, ghi, seq_len=72, horizon=6, epochs=10, batch_size=64)

# 3. Prepare prediction results
result_data = ori_data.merge(pre_data, on="Timestamp", how="inner")
result_data["pre_Global_for_engerer2"] = result_data["pre_Global"].clip(lower=0)

# 4. Run Engerer2 using observed GHI
kd_obs, dhi_obs, dni_obs = engerer2(result_data["Global"], result_data["Timestamp"], lat, lon, averaging_period=1)

# 5. Run Engerer2 using predicted GHI
kd_pred, dhi_pred, dni_pred = engerer2(result_data["pre_Global_for_engerer2"], result_data["Timestamp"], lat, lon, averaging_period=1)

# 6. Store separation results
result_data["kd_engerer2_obsghi"] = np.asarray(kd_obs).reshape(-1)
result_data["dhi_engerer2_obsghi"] = np.asarray(dhi_obs).reshape(-1)
result_data["dni_engerer2_obsghi"] = np.asarray(dni_obs).reshape(-1)
result_data["kd_engerer2_predghi"] = np.asarray(kd_pred).reshape(-1)
result_data["dhi_engerer2_predghi"] = np.asarray(dhi_pred).reshape(-1)
result_data["dni_engerer2_predghi"] = np.asarray(dni_pred).reshape(-1)

# 7. Add observed DNI and DHI for testing
obs_data = data[["UTC", "dni", "dhi"]].rename(columns={"UTC": "Timestamp", "dni": "Direct", "dhi": "Diffuse"})
obs_data["Timestamp"] = pd.to_datetime(obs_data["Timestamp"])
result_data["Timestamp"] = pd.to_datetime(result_data["Timestamp"])
result_data = result_data.merge(obs_data, on="Timestamp", how="left")

# 8. Print test data
print("\n==============================")
print("Result data preview")
print("==============================")
print(result_data.head(10))

print("\n==============================")
print("Result data columns")
print("==============================")
print(result_data.columns.tolist())

print("\n==============================")
print("Sample numbers")
print("==============================")
print(f"Original data samples: {len(data)}")
print(f"Prediction result samples: {len(result_data)}")

# 9. Print metrics
metrics_df = pd.DataFrame({
    "GHI_LSTM": cal_metrics(result_data, obs_col="Global", pred_col="pre_Global"),
    "DHI_Engerer2_obsGHI": cal_metrics(result_data, obs_col="Diffuse", pred_col="dhi_engerer2_obsghi"),
    "DNI_Engerer2_obsGHI": cal_metrics(result_data, obs_col="Direct", pred_col="dni_engerer2_obsghi"),
    "DHI_Engerer2_predGHI": cal_metrics(result_data, obs_col="Diffuse", pred_col="dhi_engerer2_predghi"),
    "DNI_Engerer2_predGHI": cal_metrics(result_data, obs_col="Direct", pred_col="dni_engerer2_predghi")
}).T

print("\n==============================")
print("Evaluation metrics")
print("==============================")
print(metrics_df.round(3))

# 10. Plot GHI prediction
plot_series(result_data, obs_col="Global", pred_col="pre_Global", obs_label="Observed GHI", pred_label="Predicted GHI", title="LSTM GHI Prediction", ylabel=r"GHI [W m$^{-2}$]", threshold=20, max_value=1400)

# 11. Save test result
result_data.to_csv(out_path, index=False)
print(f"\n[DONE] Saved: {out_path}")