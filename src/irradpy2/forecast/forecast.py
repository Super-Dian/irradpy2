"""
Forecasting module for solar radiation prediction.

This module provides classical forecasting methods and deep learning
forecasting methods for Global Horizontal Irradiance prediction.

Main functions
--------------
Classical forecasting methods:
- forecast_persistence(datetime, ghi)
- forecast_arima(datetime, ghi)

Deep learning forecasting methods:
- forecast_rnn(datetime, ghi)
- forecast_lstm(datetime, ghi)
- forecast_informer(datetime, ghi)

Notes
-----
The input must contain two numpy.ndarray objects:
- datetime: datetime sequence
- ghi: Global Horizontal Irradiance sequence

Supported input shapes:
- datetime: (n,) or (n, 1)
- ghi: (n,) or (n, 1)

Each public function returns two pandas DataFrames:
- ori_data: the historical input used, with Timestamp and Global
- pre_data: future timestamps with pre_Global in W/m2

Timestamps are UTC; naive inputs are interpreted as UTC.
Choose forecast_duration (e.g. "30min" or "1h") or horizon (number of steps).
The default is six steps at the input cadence. The forecast origin is the latest
usable sample at or before as_of, or the latest usable sample if as_of is omitted.
"""

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _forecast_window(datetime, ghi, horizon, forecast_duration, as_of):
    """Select the latest contiguous history and construct the future timestamps."""
    time = np.asarray(datetime)
    y = np.asarray(ghi, dtype=float)
    if time.ndim == 2 and time.shape[1] == 1:
        time = time[:, 0]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if time.ndim != 1 or y.ndim != 1 or len(time) != len(y):
        raise ValueError("datetime and ghi must be equally sized one-dimensional series.")

    history = pd.DataFrame({
        "Timestamp": pd.to_datetime(time, errors="coerce", utc=True).tz_convert(None),
        "Global": y,
    })
    history["Global"] = history["Global"].replace(
        [np.inf, -np.inf, -9999, -999.99, -999.9, -999, -99], np.nan
    ).clip(lower=0)
    history = history.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="stable")
    history = history.drop_duplicates("Timestamp", keep="last")

    cutoff = None
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, utc=True)
        if pd.isna(cutoff):
            raise ValueError("as_of must be a valid timestamp.")
        cutoff = cutoff.tz_convert(None)
        history = history[history["Timestamp"] <= cutoff]

    intervals = history["Timestamp"].diff().value_counts()
    if intervals.empty:
        raise ValueError("At least two historical timestamps are required to infer the sampling interval.")
    if len(intervals) > 1 and intervals.iloc[0] == intervals.iloc[1]:
        raise ValueError("The historical sampling interval is ambiguous. Supply a regularly sampled series.")
    interval = intervals.index[0]

    history = history.dropna(subset=["Global"])
    if history.empty:
        raise ValueError("No usable historical GHI is available at the forecast origin.")
    breaks = history["Timestamp"].diff().ne(interval).to_numpy()
    history = history.iloc[np.flatnonzero(breaks)[-1]:].reset_index(drop=True)
    origin = history["Timestamp"].iloc[-1]

    if cutoff is not None and origin + interval <= cutoff:
        raise ValueError("The latest usable GHI is at least one sampling interval older than as_of.")

    if forecast_duration is not None:
        if horizon is not None:
            raise ValueError("Specify either forecast_duration or horizon.")
        if not isinstance(forecast_duration, (str, pd.Timedelta)):
            raise ValueError("Use a duration such as '30min' or '1h'.")
        duration = pd.Timedelta(forecast_duration)
        if pd.isna(duration) or duration <= pd.Timedelta(0) or duration % interval:
            raise ValueError(f"forecast_duration must be a positive multiple of {interval}.")
        horizon = duration // interval
    elif horizon is None:
        horizon = 6

    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, (bool, np.bool_)) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of steps.")
    future = pd.date_range(start=origin + interval, periods=horizon, freq=interval)
    return history, future


def _prediction_frame(future, values):
    return pd.DataFrame({"Timestamp": future, "pre_Global": np.clip(values, 0, None)})


def forecast_persistence(datetime, ghi, horizon=None, *, forecast_duration=None, as_of=None):
    """Forecast future GHI by holding the last usable observation constant."""
    history, future = _forecast_window(datetime, ghi, horizon, forecast_duration, as_of)
    return history, _prediction_frame(future, np.full(len(future), history["Global"].iloc[-1]))


def forecast_arima(datetime, ghi, order=(2, 1, 2), horizon=None, *, forecast_duration=None, as_of=None):
    """Fit ARIMA to historical GHI and forecast the requested future steps."""
    from statsmodels.tsa.arima.model import ARIMA

    history, future = _forecast_window(datetime, ghi, horizon, forecast_duration, as_of)
    y = history["Global"].to_numpy()
    if len(y) < max(20, sum(order) + 2):
        raise ValueError("Contiguous historical GHI is too short for the selected ARIMA order.")
    values = np.repeat(y[-1], len(future)) if np.ptp(y) == 0 else ARIMA(y, order=order).fit().forecast(steps=len(future))
    return history, _prediction_frame(future, values)


class SeqDataset(Dataset):
    """Historical input windows with the next observed GHI as the target."""

    def __init__(self, scaled_ghi, seq_len):
        self.scaled_ghi = torch.as_tensor(scaled_ghi, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.scaled_ghi) - self.seq_len

    def __getitem__(self, index):
        end = index + self.seq_len
        return self.scaled_ghi[index:end, None], self.scaled_ghi[end]


class RNNModel(nn.Module):
    """Basic RNN forecasting model."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()

        self.rnn = nn.RNN(input_size=in_dim, hidden_size=hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out.squeeze(-1)


class LSTMModel(nn.Module):
    """Basic LSTM forecasting model."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number.")

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class InformerModel(nn.Module):
    """
    Lightweight Transformer-style forecasting model.

    This is an Informer-style model, not a full original Informer implementation.
    """

    def __init__(self, in_dim, d_model=128, nhead=4, num_layers=2, max_len=5000):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.pos(x)

        out = self.encoder(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out.squeeze(-1)


def _train_model(model, train_loader, epochs=20, lr=1e-3):
    """Fit a one-step model using all historical training windows."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def _forecast_deep_learning(
    history, future, model, seq_len=72, epochs=20, batch_size=64, lr=1e-3,
):
    """Train once on the historical series, then predict one step at a time."""
    y = history["Global"].to_numpy()
    if not isinstance(seq_len, int) or seq_len < 1 or len(y) <= seq_len:
        raise ValueError("Contiguous historical GHI must contain more samples than seq_len.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if np.ptp(y) == 0:
        return history, _prediction_frame(future, np.full(len(future), y[-1]))

    scale = y.max()
    scaled = y / scale
    dataset = SeqDataset(scaled, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = _train_model(model, loader, epochs=epochs, lr=lr)
    model.eval()

    window = torch.tensor(scaled[-seq_len:], dtype=torch.float32, device=DEVICE)
    predictions = []
    with torch.no_grad():
        for _ in future:
            value = model(window[None, :, None]).reshape(1).clamp_min(0)
            predictions.append(value)
            window = torch.cat((window[1:], value))
        values = torch.cat(predictions).cpu().numpy() * scale

    return history, _prediction_frame(future, values)


def forecast_rnn(
    datetime, ghi, seq_len=72, horizon=None, epochs=20, batch_size=64, lr=1e-3,
    hidden=128, *, forecast_duration=None, as_of=None,
):
    """Forecast future GHI recursively with an RNN."""
    history, future = _forecast_window(datetime, ghi, horizon, forecast_duration, as_of)
    model = RNNModel(in_dim=1, hidden=hidden)
    return _forecast_deep_learning(history, future, model, seq_len, epochs, batch_size, lr)


def forecast_lstm(
    datetime, ghi, seq_len=72, horizon=None, epochs=20, batch_size=64, lr=1e-3,
    hidden=128, *, forecast_duration=None, as_of=None,
):
    """Forecast future GHI recursively with an LSTM."""
    history, future = _forecast_window(datetime, ghi, horizon, forecast_duration, as_of)
    model = LSTMModel(in_dim=1, hidden=hidden)
    return _forecast_deep_learning(history, future, model, seq_len, epochs, batch_size, lr)


def forecast_informer(
    datetime, ghi, seq_len=72, horizon=None, epochs=20, batch_size=64, lr=1e-3,
    d_model=128, nhead=4, num_layers=2, *, forecast_duration=None, as_of=None,
):
    """Forecast future GHI recursively with the Informer-style model."""
    history, future = _forecast_window(datetime, ghi, horizon, forecast_duration, as_of)
    model = InformerModel(in_dim=1, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=seq_len)
    return _forecast_deep_learning(history, future, model, seq_len, epochs, batch_size, lr)
