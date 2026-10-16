# `irradpy2.forecast` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (5): `forecast_arima`, `forecast_informer`, `forecast_lstm`, `forecast_persistence`, `forecast_rnn`

---

### `forecast_arima(datetime, ghi, order=(2, 1, 2), test_ratio=0.15)`

Forecast Global irradiance using ARIMA.

**Parameters**

- `datetime` (*numpy.ndarray*) — Datetime sequence, shape (n,) or (n, 1); parsed with ``pandas.to_datetime`` (unparseable rows are dropped).
- `ghi` (*numpy.ndarray*) — Global horizontal irradiance, shape (n,) or (n, 1), same length as ``datetime`` [W m-2].
- `order` (*tuple of int, default (2, 1, 2)*) — ARIMA (p, d, q) order passed to ``statsmodels.tsa.arima.model.ARIMA``.
- `test_ratio` (*float, default 0.15*) — Trailing fraction of the time-sorted series used as the test window.

**Returns**

- `ori_data` (*pandas.DataFrame*) — Observed values over the test window: columns ``Timestamp``, ``Global``.
- `pre_data` (*pandas.DataFrame*) — Predictions on the same timestamps: columns ``Timestamp``, ``pre_Global``.

**Notes**

Fits ARIMA on the first ``(1 - test_ratio)`` fraction (requires
``statsmodels`` and at least 20 training samples, ``ValueError``
otherwise) and forecasts exactly ``len(test)`` steps; predictions are
clipped at 0. Both arguments must be ``numpy.ndarray`` (TypeError
otherwise) with matching lengths and shapes (n,) or (n, 1); invalid
timestamps are dropped, sentinel values -9999/-999/-99 are gap-filled
forward then backward, negative GHI is clipped to 0, and the series is
sorted by time. ``ValueError`` is raised when GHI is entirely missing
or the series is too short.

### `forecast_informer(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=0.001, d_model=128, nhead=4, num_layers=2)`

Forecast Global irradiance using an Informer-style model.

**Parameters**

- `datetime` (*numpy.ndarray*) — Datetime sequence, shape (n,) or (n, 1); parsed with ``pandas.to_datetime`` (unparseable rows are dropped).
- `ghi` (*numpy.ndarray*) — Global horizontal irradiance, shape (n,) or (n, 1), same length as ``datetime`` [W m-2].
- `seq_len` (*int, default 72*) — Input window length in samples fed to the model.
- `horizon` (*int, default 6*) — Gap in samples between the end of the input window and the target sample.
- `epochs` (*int, default 20*) — Training epochs; the best validation-loss state is kept.
- `batch_size` (*int, default 64*) — Mini-batch size.
- `lr` (*float, default 0.001*) — Adam learning rate.
- `d_model` (*int, default 128*) — Transformer embedding dimension; must be divisible by ``nhead`` (``ValueError`` otherwise).
- `nhead` (*int, default 4*) — Number of attention heads.
- `num_layers` (*int, default 2*) — Number of transformer encoder layers.

**Returns**

- `ori_data` (*pandas.DataFrame*) — Observed values over the test window: columns ``Timestamp``, ``Global``.
- `pre_data` (*pandas.DataFrame*) — Predictions on the same timestamps: columns ``Timestamp``, ``pre_Global``.

**Notes**

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

### `forecast_lstm(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=0.001, hidden=128)`

Forecast Global irradiance using an LSTM.

**Parameters**

- `datetime` (*numpy.ndarray*) — Datetime sequence, shape (n,) or (n, 1); parsed with ``pandas.to_datetime`` (unparseable rows are dropped).
- `ghi` (*numpy.ndarray*) — Global horizontal irradiance, shape (n,) or (n, 1), same length as ``datetime`` [W m-2].
- `seq_len` (*int, default 72*) — Input window length in samples fed to the model.
- `horizon` (*int, default 6*) — Gap in samples between the end of the input window and the target sample.
- `epochs` (*int, default 20*) — Training epochs; the best validation-loss state is kept.
- `batch_size` (*int, default 64*) — Mini-batch size.
- `lr` (*float, default 0.001*) — Adam learning rate.
- `hidden` (*int, default 128*) — Hidden units of the LSTM layer.

**Returns**

- `ori_data` (*pandas.DataFrame*) — Observed values over the test window: columns ``Timestamp``, ``Global``.
- `pre_data` (*pandas.DataFrame*) — Predictions on the same timestamps: columns ``Timestamp``, ``pre_Global``.

**Notes**

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

### `forecast_persistence(datetime, ghi, horizon=6, test_ratio=0.15)`

Forecast Global irradiance using persistence.  Persistence assumes that future irradiance remains equal to the most recent observed irradiance.

**Parameters**

- `datetime` (*numpy.ndarray*) — Datetime sequence, shape (n,) or (n, 1); parsed with ``pandas.to_datetime`` (unparseable rows are dropped).
- `ghi` (*numpy.ndarray*) — Global horizontal irradiance, shape (n,) or (n, 1), same length as ``datetime`` [W m-2].
- `horizon` (*int, default 6*) — Forecast lead in samples: the prediction at time ``t`` is the observation at ``t - horizon``.
- `test_ratio` (*float, default 0.15*) — Trailing fraction of the time-sorted series used as the evaluation window.

**Returns**

- `ori_data` (*pandas.DataFrame*) — Observed values over the evaluation window: columns ``Timestamp``, ``Global``.
- `pre_data` (*pandas.DataFrame*) — Predictions on the same timestamps: columns ``Timestamp``, ``pre_Global``.

**Notes**

The evaluation window starts at
``max(int(n * (1 - test_ratio)), horizon)``; predictions are clipped
at 0. Both arguments must be ``numpy.ndarray`` (TypeError otherwise)
with matching lengths and shapes (n,) or (n, 1); invalid timestamps are
dropped, sentinel values -9999/-999/-99 are gap-filled forward then
backward, negative GHI is clipped to 0, and the series is sorted by
time. ``ValueError`` is raised when GHI is entirely missing or the
series is too short.

### `forecast_rnn(datetime, ghi, seq_len=72, horizon=6, epochs=20, batch_size=64, lr=0.001, hidden=128)`

Forecast Global irradiance using an RNN.

**Parameters**

- `datetime` (*numpy.ndarray*) — Datetime sequence, shape (n,) or (n, 1); parsed with ``pandas.to_datetime`` (unparseable rows are dropped).
- `ghi` (*numpy.ndarray*) — Global horizontal irradiance, shape (n,) or (n, 1), same length as ``datetime`` [W m-2].
- `seq_len` (*int, default 72*) — Input window length in samples fed to the model.
- `horizon` (*int, default 6*) — Gap in samples between the end of the input window and the target sample.
- `epochs` (*int, default 20*) — Training epochs; the best validation-loss state is kept.
- `batch_size` (*int, default 64*) — Mini-batch size.
- `lr` (*float, default 0.001*) — Adam learning rate.
- `hidden` (*int, default 128*) — Hidden units of the RNN layer.

**Returns**

- `ori_data` (*pandas.DataFrame*) — Observed values over the test window: columns ``Timestamp``, ``Global``.
- `pre_data` (*pandas.DataFrame*) — Predictions on the same timestamps: columns ``Timestamp``, ``pre_Global``.

**Notes**

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

