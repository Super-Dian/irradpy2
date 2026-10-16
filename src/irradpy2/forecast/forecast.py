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

The output contains two pandas DataFrame objects:
- ori_data: original Global irradiance values with Timestamp
- pre_data: predicted Global irradiance values with Timestamp
"""

import copy
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_ghi_array(datetime, ghi):
    """
    Prepare datetime and GHI arrays.

    Parameters
    ----------
    datetime : numpy.ndarray
        Datetime sequence with shape (n,) or (n, 1).
    ghi : numpy.ndarray
        GHI sequence with shape (n,) or (n, 1).

    Returns
    -------
    time : pandas.Series
        Cleaned datetime sequence.
    y : numpy.ndarray
        Cleaned one-dimensional GHI array.
    """
    if not isinstance(datetime, np.ndarray):
        raise TypeError("datetime must be a numpy ndarray.")

    if not isinstance(ghi, np.ndarray):
        raise TypeError("ghi must be a numpy ndarray.")

    datetime_arr = np.asarray(datetime)

    if datetime_arr.ndim == 1:
        time = datetime_arr
    elif datetime_arr.ndim == 2 and datetime_arr.shape[1] == 1:
        time = datetime_arr[:, 0]
    else:
        raise ValueError("datetime must have shape (n,) or (n, 1).")

    ghi_arr = np.asarray(ghi, dtype=float)

    if ghi_arr.ndim == 1:
        y = ghi_arr
    elif ghi_arr.ndim == 2 and ghi_arr.shape[1] == 1:
        y = ghi_arr[:, 0]
    else:
        raise ValueError("ghi must have shape (n,) or (n, 1).")

    if len(time) != len(y):
        raise ValueError("datetime and ghi must have the same length.")

    df = pd.DataFrame({"Timestamp": pd.to_datetime(time, errors="coerce"), "Global": y})
    df = df.dropna(subset=["Timestamp"])
    df["Global"] = df["Global"].replace([-9999, -999, -99], np.nan)
    df["Global"] = df["Global"].ffill().bfill()

    if df["Global"].isna().any():
        raise ValueError("ghi contains only missing or invalid values.")

    df["Global"] = df["Global"].clip(lower=0)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    time = df["Timestamp"]
    y = df["Global"].to_numpy(dtype=float)

    return time, y


def forecast_persistence(datetime, ghi, horizon=6, test_ratio=0.15):
    """
    Forecast Global irradiance using persistence.

    Persistence assumes that future irradiance remains equal to
    the most recent observed irradiance.

    Parameters
    ----------
    datetime : numpy.ndarray
        Datetime sequence, shape (n,) or (n, 1); parsed with
        ``pandas.to_datetime`` (unparseable rows are dropped).
    ghi : numpy.ndarray
        Global horizontal irradiance, shape (n,) or (n, 1), same length as
        ``datetime`` [W m-2].
    horizon : int, default 6
        Forecast lead in samples: the prediction at time ``t`` is the
        observation at ``t - horizon``.
    test_ratio : float, default 0.15
        Trailing fraction of the time-sorted series used as the evaluation
        window.

    Returns
    -------
    ori_data : pandas.DataFrame
        Observed values over the evaluation window: columns ``Timestamp``,
        ``Global``.
    pre_data : pandas.DataFrame
        Predictions on the same timestamps: columns ``Timestamp``,
        ``pre_Global``.

    Notes
    -----
    The evaluation window starts at
    ``max(int(n * (1 - test_ratio)), horizon)``; predictions are clipped
    at 0. Both arguments must be ``numpy.ndarray`` (TypeError otherwise)
    with matching lengths and shapes (n,) or (n, 1); invalid timestamps are
    dropped, sentinel values -9999/-999/-99 are gap-filled forward then
    backward, negative GHI is clipped to 0, and the series is sorted by
    time. ``ValueError`` is raised when GHI is entirely missing or the
    series is too short.
    """
    time, y = prepare_ghi_array(datetime, ghi)

    test_start = int(len(y) * (1 - test_ratio))
    start = max(test_start, horizon)

    if start >= len(y):
        raise ValueError("Data length is too short for the current horizon and test_ratio.")

    y_true = y[start:]
    y_pred = y[start - horizon:len(y) - horizon]
    y_pred = np.clip(y_pred, 0, None)

    target_time = time.iloc[start:].to_numpy()

    ori_data = pd.DataFrame({"Timestamp": target_time, "Global": y_true})
    pre_data = pd.DataFrame({"Timestamp": target_time, "pre_Global": y_pred})

    return ori_data, pre_data


def forecast_arima(datetime, ghi, order=(2, 1, 2), test_ratio=0.15):
    """
    Forecast Global irradiance using ARIMA.

    Parameters
    ----------
    datetime : numpy.ndarray
        Datetime sequence, shape (n,) or (n, 1); parsed with
        ``pandas.to_datetime`` (unparseable rows are dropped).
    ghi : numpy.ndarray
        Global horizontal irradiance, shape (n,) or (n, 1), same length as
        ``datetime`` [W m-2].
    order : tuple of int, default (2, 1, 2)
        ARIMA (p, d, q) order passed to ``statsmodels.tsa.arima.model.ARIMA``.
    test_ratio : float, default 0.15
        Trailing fraction of the time-sorted series used as the test window.

    Returns
    -------
    ori_data : pandas.DataFrame
        Observed values over the test window: columns ``Timestamp``,
        ``Global``.
    pre_data : pandas.DataFrame
        Predictions on the same timestamps: columns ``Timestamp``,
        ``pre_Global``.

    Notes
    -----
    Fits ARIMA on the first ``(1 - test_ratio)`` fraction (requires
    ``statsmodels`` and at least 20 training samples, ``ValueError``
    otherwise) and forecasts exactly ``len(test)`` steps; predictions are
    clipped at 0. Both arguments must be ``numpy.ndarray`` (TypeError
    otherwise) with matching lengths and shapes (n,) or (n, 1); invalid
    timestamps are dropped, sentinel values -9999/-999/-99 are gap-filled
    forward then backward, negative GHI is clipped to 0, and the series is
    sorted by time. ``ValueError`` is raised when GHI is entirely missing
    or the series is too short.
    """
    from statsmodels.tsa.arima.model import ARIMA

    time, y = prepare_ghi_array(datetime, ghi)

    test_start = int(len(y) * (1 - test_ratio))
    train_y = y[:test_start]
    test_y = y[test_start:]

    if len(train_y) < 20:
        raise ValueError("Training data is too short for ARIMA.")

    model = ARIMA(train_y, order=order)
    fitted = model.fit()

    y_pred = fitted.forecast(steps=len(test_y))
    y_pred = np.clip(np.array(y_pred), 0, None)

    target_time = time.iloc[test_start:].to_numpy()

    ori_data = pd.DataFrame({"Timestamp": target_time, "Global": test_y})
    pre_data = pd.DataFrame({"Timestamp": target_time, "pre_Global": y_pred})

    return ori_data, pre_data


class SeqDataset(Dataset):
    """Sequence-to-one dataset."""

    def __init__(self, X, y, seq_len, horizon):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return max(0, len(self.X) - self.seq_len - self.horizon + 1)

    def __getitem__(self, index):
        x = self.X[index:index + self.seq_len]
        y = self.y[index + self.seq_len + self.horizon - 1]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class RNNModel(nn.Module):
    """Basic RNN forecasting model."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()

        self.rnn = nn.RNN(input_size=in_dim, hidden_size=hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """Map a batch of input windows to one scalar prediction each.

        Parameters
        ----------
        x : torch.Tensor
            Input windows with shape (batch, seq_len, in_dim).

        Returns
        -------
        torch.Tensor
            Predictions with shape (batch,).
        """
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
        """Map a batch of input windows to one scalar prediction each.

        Parameters
        ----------
        x : torch.Tensor
            Input windows with shape (batch, seq_len, in_dim).

        Returns
        -------
        torch.Tensor
            Predictions with shape (batch,).
        """
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
        """Add sinusoidal positional encoding to the embedded input.

        Parameters
        ----------
        x : torch.Tensor
            Embedded sequence with shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Input plus the positional encoding, same shape as ``x``.
        """
        return x + self.pe[:, :x.size(1)].to(x.device)


class InformerModel(nn.Module):
    """
    Lightweight Transformer-style forecasting model.

    This is an Informer-style model, not a full original Informer implementation.
    """

    def __init__(self, in_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)

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
        """Run projection, positional encoding and the transformer encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input windows with shape (batch, seq_len, in_dim).

        Returns
        -------
        torch.Tensor
            Predictions with shape (batch,).
        """
        x = self.in_proj(x)
        x = self.pos(x)

        out = self.encoder(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out.squeeze(-1)


def _evaluate_model(model, loader):
    """Evaluate model using validation loss."""
    model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            total_loss += loss.item() * len(xb)

    return total_loss / len(loader.dataset)


def _train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    """Train model and keep the best model state in memory."""
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = _evaluate_model(model, val_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Train={train_loss:.4f} | Val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return model


def _forecast_deep_learning(datetime, ghi, model, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=1e-3):
    """
    Shared deep learning forecasting process.

    datetime is used for sorting and output alignment.
    ghi is used as the actual model input.
    """
    time, y = prepare_ghi_array(datetime, ghi)

    X = y.reshape(-1, 1)

    X_train, X_temp, y_train, y_temp, time_train, time_temp = train_test_split(
        X, y, time, test_size=0.3, shuffle=False
    )

    X_val, X_test, y_val, y_test, time_val, time_test = train_test_split(
        X_temp, y_temp, time_temp, test_size=0.5, shuffle=False
    )

    x_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_max = np.nanmax(y_train)

    if not np.isfinite(y_max) or y_max <= 0:
        raise ValueError("Global irradiance maximum must be greater than 0.")

    y_train = y_train / y_max
    y_val = y_val / y_max
    y_test = y_test / y_max

    train_set = SeqDataset(X_train, y_train, seq_len, horizon)
    val_set = SeqDataset(X_val, y_val, seq_len, horizon)
    test_set = SeqDataset(X_test, y_test, seq_len, horizon)

    if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
        raise ValueError("Data length is too short for the current seq_len and horizon.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = _train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=epochs, lr=lr)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)

            pred = model(xb).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true) * y_max
    y_pred = np.clip(np.array(y_pred), 0, None) * y_max

    time_test = pd.Series(time_test).reset_index(drop=True)

    target_index = np.arange(len(y_true)) + seq_len + horizon - 1
    target_time = time_test.iloc[target_index].to_numpy()

    ori_data = pd.DataFrame({"Timestamp": target_time, "Global": y_true})
    pre_data = pd.DataFrame({"Timestamp": target_time, "pre_Global": y_pred})

    return ori_data, pre_data


def forecast_rnn(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=1e-3, hidden=128):
    """
    Forecast Global irradiance using an RNN.

    Parameters
    ----------
    datetime : numpy.ndarray
        Datetime sequence, shape (n,) or (n, 1); parsed with
        ``pandas.to_datetime`` (unparseable rows are dropped).
    ghi : numpy.ndarray
        Global horizontal irradiance, shape (n,) or (n, 1), same length as
        ``datetime`` [W m-2].
    seq_len : int, default 72
        Input window length in samples fed to the model.
    horizon : int, default 6
        Gap in samples between the end of the input window and the target
        sample.
    epochs : int, default 20
        Training epochs; the best validation-loss state is kept.
    batch_size : int, default 64
        Mini-batch size.
    lr : float, default 0.001
        Adam learning rate.
    hidden : int, default 128
        Hidden units of the RNN layer.

    Returns
    -------
    ori_data : pandas.DataFrame
        Observed values over the test window: columns ``Timestamp``,
        ``Global``.
    pre_data : pandas.DataFrame
        Predictions on the same timestamps: columns ``Timestamp``,
        ``pre_Global``.

    Notes
    -----
    Requires ``torch``. The shared pipeline makes a chronological 70/15/15
    train/validation/test split (no shuffling), scales inputs with
    ``MinMaxScaler``, divides targets by the training-set maximum, trains
    with Adam and keeps the best validation-loss epoch (per-epoch progress
    is printed to stdout). The returned frames cover the test window;
    predictions are clipped at 0. Both arguments must be ``numpy.ndarray``
    (TypeError otherwise) with matching lengths and shapes (n,) or (n, 1);
    invalid timestamps are dropped, sentinel values -9999/-999/-99 are
    gap-filled forward then backward, negative GHI is clipped to 0, and the
    series is sorted by time. ``ValueError`` is raised when GHI is entirely
    missing or the series is too short.
    """
    model = RNNModel(in_dim=1, hidden=hidden)

    return _forecast_deep_learning(
        datetime=datetime,
        ghi=ghi,
        model=model,
        seq_len=seq_len,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )


def forecast_lstm(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=1e-3, hidden=128):
    """
    Forecast Global irradiance using an LSTM.

    Parameters
    ----------
    datetime : numpy.ndarray
        Datetime sequence, shape (n,) or (n, 1); parsed with
        ``pandas.to_datetime`` (unparseable rows are dropped).
    ghi : numpy.ndarray
        Global horizontal irradiance, shape (n,) or (n, 1), same length as
        ``datetime`` [W m-2].
    seq_len : int, default 72
        Input window length in samples fed to the model.
    horizon : int, default 6
        Gap in samples between the end of the input window and the target
        sample.
    epochs : int, default 20
        Training epochs; the best validation-loss state is kept.
    batch_size : int, default 64
        Mini-batch size.
    lr : float, default 0.001
        Adam learning rate.
    hidden : int, default 128
        Hidden units of the LSTM layer.

    Returns
    -------
    ori_data : pandas.DataFrame
        Observed values over the test window: columns ``Timestamp``,
        ``Global``.
    pre_data : pandas.DataFrame
        Predictions on the same timestamps: columns ``Timestamp``,
        ``pre_Global``.

    Notes
    -----
    Requires ``torch``. The shared pipeline makes a chronological 70/15/15
    train/validation/test split (no shuffling), scales inputs with
    ``MinMaxScaler``, divides targets by the training-set maximum, trains
    with Adam and keeps the best validation-loss epoch (per-epoch progress
    is printed to stdout). The returned frames cover the test window;
    predictions are clipped at 0. Both arguments must be ``numpy.ndarray``
    (TypeError otherwise) with matching lengths and shapes (n,) or (n, 1);
    invalid timestamps are dropped, sentinel values -9999/-999/-99 are
    gap-filled forward then backward, negative GHI is clipped to 0, and the
    series is sorted by time. ``ValueError`` is raised when GHI is entirely
    missing or the series is too short.
    """
    model = LSTMModel(in_dim=1, hidden=hidden)

    return _forecast_deep_learning(
        datetime=datetime,
        ghi=ghi,
        model=model,
        seq_len=seq_len,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )


def forecast_informer(
    datetime,
    ghi,
    seq_len=72,
    horizon=6,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    d_model=128,
    nhead=4,
    num_layers=2
):
    """
    Forecast Global irradiance using an Informer-style model.

    Parameters
    ----------
    datetime : numpy.ndarray
        Datetime sequence, shape (n,) or (n, 1); parsed with
        ``pandas.to_datetime`` (unparseable rows are dropped).
    ghi : numpy.ndarray
        Global horizontal irradiance, shape (n,) or (n, 1), same length as
        ``datetime`` [W m-2].
    seq_len : int, default 72
        Input window length in samples fed to the model.
    horizon : int, default 6
        Gap in samples between the end of the input window and the target
        sample.
    epochs : int, default 20
        Training epochs; the best validation-loss state is kept.
    batch_size : int, default 64
        Mini-batch size.
    lr : float, default 0.001
        Adam learning rate.
    d_model : int, default 128
        Transformer embedding dimension; must be divisible by ``nhead``
        (``ValueError`` otherwise).
    nhead : int, default 4
        Number of attention heads.
    num_layers : int, default 2
        Number of transformer encoder layers.

    Returns
    -------
    ori_data : pandas.DataFrame
        Observed values over the test window: columns ``Timestamp``,
        ``Global``.
    pre_data : pandas.DataFrame
        Predictions on the same timestamps: columns ``Timestamp``,
        ``pre_Global``.

    Notes
    -----
    A lightweight Informer-style Transformer, not the full original
    Informer implementation. Requires ``torch``. The shared pipeline makes
    a chronological 70/15/15 train/validation/test split (no shuffling),
    scales inputs with ``MinMaxScaler``, divides targets by the
    training-set maximum, trains with Adam and keeps the best
    validation-loss epoch (per-epoch progress is printed to stdout). The
    returned frames cover the test window; predictions are clipped at 0.
    Both arguments must be ``numpy.ndarray`` (TypeError otherwise) with
    matching lengths and shapes (n,) or (n, 1); invalid timestamps are
    dropped, sentinel values -9999/-999/-99 are gap-filled forward then
    backward, negative GHI is clipped to 0, and the series is sorted by
    time. ``ValueError`` is raised when GHI is entirely missing or the
    series is too short.
    """
    model = InformerModel(in_dim=1, d_model=d_model, nhead=nhead, num_layers=num_layers)

    return _forecast_deep_learning(
        datetime=datetime,
        ghi=ghi,
        model=model,
        seq_len=seq_len,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )