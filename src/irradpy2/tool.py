import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(file_path):
    """
    Read a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to read.

    Returns
    -------
    pandas.DataFrame — the parsed CSV table.

    Notes
    -----
    Thin wrapper around `pandas.read_csv`. Downstream helpers such as `prepare_clearsky` auto-detect the time column among `UTC`, `UTC_time`, `time`.
    """
    return pd.read_csv(file_path)


def cal_zen(data, latitude, longitude, time_col=None, zen_col="zen"):
    """
    Calculate solar zenith angle in degrees.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table; copied before the new column is added.
    latitude : float
        Site latitude [degrees].
    longitude : float
        Site longitude [degrees].
    time_col : str, optional
        Datetime column. When `None`, auto-detects among `Timestamp`, `UTC`, `UTC_time`, `time`.
    zen_col : str, default 'zen'
        Output column name for the zenith angle.

    Returns
    -------
    pandas.DataFrame — copy of `data` with `zen_col` added (zenith angle in degrees).

    Notes
    -----
    Returns a copy of `data` with `zen_col` filled in **degrees**. The clear-sky models take radians — convert with `np.radians(zen)`. Raises `ValueError` when no time column can be found.
    """
    data = data.copy()

    if time_col is None:
        for col in ["Timestamp", "UTC", "UTC_time", "time"]:
            if col in data.columns:
                time_col = col
                break

    if time_col is None:
        raise ValueError("No time column found. Expected one of: Timestamp, UTC, UTC_time, time.")

    time = pd.to_datetime(data[time_col], utc=True)
    doy = time.dt.dayofyear
    hour = time.dt.hour + time.dt.minute / 60 + time.dt.second / 3600
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)

    declination = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)

    equation_time = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))

    true_solar_time = hour * 60 + equation_time + 4 * longitude
    hour_angle = np.deg2rad(true_solar_time / 4 - 180)
    latitude_rad = np.deg2rad(latitude)

    cos_zen = np.sin(latitude_rad) * np.sin(declination) + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    data[zen_col] = np.rad2deg(np.arccos(np.clip(cos_zen, -1, 1)))

    return data

def cal_exth(data, time_col="UTC", zen_col="zen", exth_col="exth"):
    """
    Calculate horizontal extraterrestrial irradiance [W/m²].

    Parameters
    ----------
    data : pandas.DataFrame
        Input table containing a zenith-angle column.
    time_col : str, default 'UTC'
        Datetime column (day-of-year is derived from it).
    zen_col : str, default 'zen'
        Zenith angle column [degrees].
    exth_col : str, default 'exth'
        Output column name for horizontal extraterrestrial irradiance.

    Returns
    -------
    pandas.DataFrame — copy of `data` with `exth_col` added [W m-2].

    Notes
    -----
    Computes the extraterrestrial DNI (1366.1 W m-2 with the annual equation) times `cos(zen)`; set to 0 where `zen > 90°`.
    """
    data = data.copy()
    time = pd.to_datetime(data[time_col], errors="coerce", utc=True)
    doy = time.dt.dayofyear.to_numpy(dtype=float)
    zen = np.deg2rad(pd.to_numeric(data[zen_col], errors="coerce").to_numpy(dtype=float))

    beta = 2 * np.pi * doy / 365
    extn = 1366.1 * (1.00011 + 0.034221 * np.cos(beta) + 0.00128 * np.sin(beta) + 0.000719 * np.cos(2 * beta) + 0.000077 * np.sin(2 * beta))
    exth = extn * np.cos(zen)
    exth[zen > np.pi / 2] = 0

    data[exth_col] = exth
    return data


def _cal_sza(data, latitude, longitude, time_col="UTC", sza_col="sza"):
    """Calculate solar zenith angle in radians."""
    data = data.copy()

    time = pd.to_datetime(data[time_col], utc=True)
    doy = time.dt.dayofyear
    hour = time.dt.hour + time.dt.minute / 60 + time.dt.second / 3600
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)

    declination = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)

    equation_time = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))

    true_solar_time = hour * 60 + equation_time + 4 * longitude
    hour_angle = np.deg2rad(true_solar_time / 4 - 180)
    latitude_rad = np.deg2rad(latitude)

    cos_sza = np.sin(latitude_rad) * np.sin(declination) + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    data[sza_col] = np.arccos(np.clip(cos_sza, -1, 1))

    return data


def cal_doy_year(data, time_col="UTC", doy_col="doy", year_col="year"):
    """
    Calculate day of year and year.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table.
    time_col : str, default 'UTC'
        Datetime column.
    doy_col : str, default 'doy'
        Output column name for day-of-year.
    year_col : str, default 'year'
        Output column name for the year.

    Returns
    -------
    pandas.DataFrame — copy of `data` with `doy_col` and `year_col` added.
    """
    data = data.copy()
    time = pd.to_datetime(data[time_col], utc=True)
    data[doy_col] = time.dt.dayofyear
    data[year_col] = time.dt.year
    return data


def cal_mse(data, obs_col="ghi", pred_col="ghi_pred"):
    """
    Calculate mean squared error.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding observed and predicted columns.
    obs_col : str, default 'ghi'
        Observed-values column.
    pred_col : str, default 'ghi_pred'
        Predicted-values column.

    Returns
    -------
    float — mean squared error.

    Notes
    -----
    Rows where either column is NaN are dropped before the mean.
    """
    valid_data = data[[obs_col, pred_col]].dropna()
    obs = valid_data[obs_col].values
    pred = valid_data[pred_col].values
    return np.mean((pred - obs) ** 2)


def cal_rmse(data, obs_col="ghi", pred_col="ghi_pred"):
    """
    Calculate root mean squared error.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding observed and predicted columns.
    obs_col : str, default 'ghi'
        Observed-values column.
    pred_col : str, default 'ghi_pred'
        Predicted-values column.

    Returns
    -------
    float — root mean squared error.
    """
    return np.sqrt(cal_mse(data, obs_col=obs_col, pred_col=pred_col))


def cal_mae(data, obs_col="ghi", pred_col="ghi_pred"):
    """
    Calculate mean absolute error.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding observed and predicted columns.
    obs_col : str, default 'ghi'
        Observed-values column.
    pred_col : str, default 'ghi_pred'
        Predicted-values column.

    Returns
    -------
    float — mean absolute error.
    """
    valid_data = data[[obs_col, pred_col]].dropna()
    obs = valid_data[obs_col].values
    pred = valid_data[pred_col].values
    return np.mean(np.abs(pred - obs))


def cal_mbe(data, obs_col="ghi", pred_col="ghi_pred"):
    """
    Calculate mean bias error.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding observed and predicted columns.
    obs_col : str, default 'ghi'
        Observed-values column.
    pred_col : str, default 'ghi_pred'
        Predicted-values column.

    Returns
    -------
    float — mean bias error (`predicted - observed` convention).

    Notes
    -----
    Sign convention: `MBE = mean(predicted - observed)`. Positive MBE means overestimation, negative MBE underestimation. Some literature uses the opposite sign — check before comparing published values.
    """
    valid_data = data[[obs_col, pred_col]].dropna()
    obs = valid_data[obs_col].values
    pred = valid_data[pred_col].values
    return np.mean(pred - obs)


def cal_r2(data, obs_col="ghi", pred_col="ghi_pred"):
    """
    Calculate coefficient of determination.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding observed and predicted columns.
    obs_col : str, default 'ghi'
        Observed-values column.
    pred_col : str, default 'ghi_pred'
        Predicted-values column.

    Returns
    -------
    float — coefficient of determination, or `NaN` when observed variance is zero.

    Notes
    -----
    Returns `NaN` when the observed variance is zero.
    """
    valid_data = data[[obs_col, pred_col]].dropna()
    obs = valid_data[obs_col].values
    pred = valid_data[pred_col].values

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)

    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


def cal_metrics(data, obs_col="ghi", pred_col="ghi_pred"):
    """
    Calculate common evaluation metrics.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding observed and predicted columns.
    obs_col : str, default 'ghi'
        Observed-values column.
    pred_col : str, default 'ghi_pred'
        Predicted-values column.

    Returns
    -------
    dict — keys `MSE`, `RMSE`, `MAE`, `MBE`, `R2`.
    """
    return {
        "MSE": cal_mse(data, obs_col=obs_col, pred_col=pred_col),
        "RMSE": cal_rmse(data, obs_col=obs_col, pred_col=pred_col),
        "MAE": cal_mae(data, obs_col=obs_col, pred_col=pred_col),
        "MBE": cal_mbe(data, obs_col=obs_col, pred_col=pred_col),
        "R2": cal_r2(data, obs_col=obs_col, pred_col=pred_col)
    }


def merge_columns(data, values, columns, mask=None, fill_value=np.nan):
    """
    Merge arrays into data columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Destination table; copied before columns are added.
    values : sequence of array-like
        Arrays merged in order; matched positionally with `columns`.
    columns : sequence of str
        Destination column names.
    mask : numpy.ndarray of bool, optional
        Rows that receive `values`; rows outside the mask get `fill_value`. Defaults to all rows.
    fill_value : float, default np.nan
        Value written to rows outside `mask` (default `NaN`).

    Returns
    -------
    pandas.DataFrame — copy of `data` with the merged columns.
    """
    data = data.copy()

    if mask is None:
        mask = np.ones(len(data), dtype=bool)

    for value, col in zip(values, columns):
        data[col] = np.full(len(data), fill_value, dtype=float)
        data.loc[mask, col] = np.asarray(value, dtype=float).reshape(-1)

    return data


def _build_index(start, end, freq="1min"):
    """Build a complete time index."""
    return pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=freq)


def _parse_time(data, time_col="UTC"):
    """Parse time column to timezone-naive UTC datetime."""
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col], utc=True, errors="coerce").dt.tz_convert(None)
    data = data.dropna(subset=[time_col]).sort_values(time_col)
    return data


def _clean_negative(data, cols, lower=0):
    """Replace values lower than the limit."""
    data = data.copy()

    for col in cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data.loc[data[col] < lower, col] = lower

    return data


def _reindex_time(data, time_col, target_index):
    """Reindex data to a target time index without interpolation."""
    data = _parse_time(data, time_col=time_col)
    data = data.set_index(time_col).reindex(target_index)
    data.index.name = time_col
    return data.reset_index()


def _interpolate_time(data, time_col, target_index):
    """Interpolate data to a target time index."""
    data = _parse_time(data, time_col=time_col)

    for col in data.columns:
        if col != time_col:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.set_index(time_col)
    data = data.reindex(target_index.union(data.index)).sort_index()
    data = data.interpolate(method="time").reindex(target_index)
    data.index.name = time_col

    return data.reset_index()


def _convert_clearsky(data):
    """Convert atmospheric variables to clear-sky model inputs."""
    data = data.copy()

    if "pressure" in data.columns:
        data["press"] = pd.to_numeric(data["pressure"], errors="coerce") / 100

    if "water_vapour" in data.columns:
        data["wv"] = pd.to_numeric(data["water_vapour"], errors="coerce") / 10

    if "ozone" in data.columns:
        data["ozone_input"] = pd.to_numeric(data["ozone"], errors="coerce") / 1000

    if "angstrom_exponent" in data.columns:
        data["ang_alpha"] = pd.to_numeric(data["angstrom_exponent"], errors="coerce")

    if "aod550" in data.columns and "ang_alpha" in data.columns:
        data["aod550"] = pd.to_numeric(data["aod550"], errors="coerce")
        data["ang_beta"] = data["aod550"] * (0.55 ** data["ang_alpha"])
        data["aod700"] = data["aod550"] * ((0.70 / 0.55) ** (-data["ang_alpha"]))

    if "albedo" in data.columns:
        data["albedo"] = pd.to_numeric(data["albedo"], errors="coerce").interpolate(limit_direction="both")

    return data


def _get_time_col(data):
    """Find the time column automatically."""
    for col in ["UTC", "UTC_time", "time"]:
        if col in data.columns:
            return col

    raise ValueError("No time column found. Expected one of: UTC, UTC_time, time.")


def select_series(data, time_col="UTC", cols=None, lower_limit=None):
    """
    Select numeric time-series columns and set datetime index.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table.
    time_col : str, default 'UTC'
        Datetime column; parsed to timezone-naive UTC and used as index.
    cols : sequence of str, optional
        Columns to keep; defaults to every non-time column.
    lower_limit : float, optional
        When given, values below it are raised to it (clipped).

    Returns
    -------
    pandas.DataFrame — time-indexed, sorted subset of the selected columns.

    Notes
    -----
    Selected columns are coerced to numeric; the returned frame is sorted by time.
    """
    data = _parse_time(data, time_col=time_col)

    if cols is None:
        cols = [col for col in data.columns if col != time_col]

    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

        if lower_limit is not None:
            data.loc[data[col] < lower_limit, col] = lower_limit

    return data.set_index(time_col)[cols].sort_index()


def resample_series(data, time_col="UTC", cols=None, freq="1h", agg="mean", lower_limit=None, label_offset=None):
    """
    Resample selected time-series columns to a fixed frequency.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table.
    time_col : str, default 'UTC'
        Datetime column.
    cols : sequence of str, optional
        Columns to keep; defaults to every non-time column.
    freq : str, default '1h'
        Pandas offset alias for the target frequency (default `"1h"`).
    agg : str, default 'mean'
        Aggregation: `"mean"` (default), `"sum"`, or `"median"`.
    lower_limit : float, optional
        Clipping bound passed through to `select_series`.
    label_offset : str, optional
        Pandas `Timedelta` added to the bin labels after resampling.

    Returns
    -------
    pandas.DataFrame — series resampled to `freq`.
    """
    data = select_series(data, time_col=time_col, cols=cols, lower_limit=lower_limit)

    if agg == "sum":
        data = data.resample(freq).sum()
    elif agg == "median":
        data = data.resample(freq).median()
    else:
        data = data.resample(freq).mean()

    if label_offset is not None:
        data.index = data.index + pd.Timedelta(label_offset)

    return data


def merge_series(data_list, time_col="UTC", dropna=True):
    """
    Merge multiple time-series DataFrames by datetime index.

    Parameters
    ----------
    data_list : sequence of pandas.DataFrame
        Frames sharing a datetime index; concatenated column-wise.
    time_col : str, default 'UTC'
        Name given to the restored time column.
    dropna : bool, default True
        When `True` (default), rows with any NaN after concatenation are dropped.

    Returns
    -------
    pandas.DataFrame — concatenated frame with a restored `time_col` column.
    """
    data = pd.concat(data_list, axis=1)

    if dropna:
        data = data.dropna()

    data = data.reset_index()
    data = data.rename(columns={data.columns[0]: time_col})

    return data


def prepare_clearsky(ground_data, merra2_data, latitude, longitude, begin, end, freq="1min", source_prefix="merra2"):
    """
    Prepare clear-sky model inputs from ground and MERRA-2 data.

    Parameters
    ----------
    ground_data : pandas.DataFrame
        Ground observations with `ghi`/`dni`/`dhi` columns.
    merra2_data : pandas.DataFrame
        MERRA-2 (or compatible) rows with meteorological inputs and `ghi_clear`.
    latitude : float
        Site latitude [degrees].
    longitude : float
        Site longitude [degrees].
    begin : str
        Start date, `YYYY-MM-DD`.
    end : str
        End date, `YYYY-MM-DD`.
    freq : str, default '1min'
        Target index frequency (default `"1min"`).
    source_prefix : str, default 'merra2'
        Suffix for the renamed `ghi`/`ghi_clear` columns (default `"merra2"`).

    Returns
    -------
    pandas.DataFrame — aligned analysis table (`UTC`, `ghi`, `dni`, `dhi`, `sza`, atmospheric inputs).

    Notes
    -----
    Pipeline: detect time columns → build a complete time index at `freq` → align ground and MERRA-2 rows → interpolate meteorology → convert atmospheric quantities (`pressure`→`press` /100, `water_vapour`→`wv` /10, `ozone`→`ozone_input` /1000, `angstrom_exponent`→`ang_alpha`, `aod550`→`ang_beta`/`aod700`) → compute `sza` in **radians**. Typical output columns: `UTC`, `ghi`, `dni`, `dhi`, `sza`, `press`, `albedo`, `wv`, `aod700`, `ang_alpha`, `ang_beta`, `ozone_input`.
    """
    start = pd.to_datetime(begin).strftime("%Y-%m-%d") + " 00:00:00"
    end = pd.to_datetime(end).strftime("%Y-%m-%d") + " 23:59:00"
    utc_time = _build_index(start, end, freq=freq)

    ground_time_col = _get_time_col(ground_data)
    merra2_time_col = _get_time_col(merra2_data)

    ground_data = _clean_negative(ground_data, cols=["ghi", "dni", "dhi"])
    ground_cols = [ground_time_col] + [col for col in ["ghi", "dni", "dhi"] if col in ground_data.columns]
    ground_data = _reindex_time(ground_data[ground_cols], time_col=ground_time_col, target_index=utc_time)

    merra2_data = merra2_data.rename(columns={
        merra2_time_col: "UTC",
        "ghi": f"ghi_{source_prefix}",
        "ghi_clear": f"ghi_clear_{source_prefix}"
    })

    merra2_data = _interpolate_time(merra2_data, time_col="UTC", target_index=utc_time)

    data = ground_data.merge(merra2_data, on="UTC", how="left")
    data = _convert_clearsky(data)
    data = _cal_sza(data, latitude=latitude, longitude=longitude, time_col="UTC", sza_col="sza")

    return data


def filter_sep_data(data, latitude, longitude, time_col="Timestamp", zen_col="zen", ghi_col="Global", dni_col="Direct", dhi_col="Diffuse", pred_cols=None, max_zen=85, min_ghi=50, require_positive=True):
    """
    Filter data for DHI and DNI separation evaluation.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with observed components and optional predicted columns.
    latitude : float
        Site latitude [degrees]; used to compute `zen` when absent.
    longitude : float
        Site longitude [degrees].
    time_col : str, default 'Timestamp'
        Datetime column (default `"Timestamp"`).
    zen_col : str, default 'zen'
        Zenith angle column [degrees]; computed when missing.
    ghi_col : str, default 'Global'
        Observed GHI column.
    dni_col : str, default 'Direct'
        Observed DNI column.
    dhi_col : str, default 'Diffuse'
        Observed DHI column.
    pred_cols : sequence of str, optional
        Predicted columns that must be finite for a row to be kept.
    max_zen : float, default 85
        Rows kept only where `zen < max_zen` (default 85).
    min_ghi : float, default 50
        Rows kept only where `ghi > min_ghi` (default 50).
    require_positive : bool, default True
        When `True` (default), also require `dni > 0` and `dhi > 0`.

    Returns
    -------
    pandas.DataFrame — filtered copy of `data`.
    """
    data = data.copy()

    if zen_col not in data.columns:
        data = cal_zen(data, latitude=latitude, longitude=longitude, time_col=time_col, zen_col=zen_col)

    if pred_cols is None:
        pred_cols = []

    required_cols = [ghi_col, dni_col, dhi_col, zen_col] + pred_cols

    for col in required_cols:
        if col in data.columns and col != time_col:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    mask = data[required_cols].notna().all(axis=1)
    mask = mask & np.isfinite(data[zen_col])
    mask = mask & (data[zen_col] < max_zen)
    mask = mask & (data[ghi_col] > min_ghi)

    if require_positive:
        mask = mask & (data[dni_col] > 0) & (data[dhi_col] > 0)

    for col in pred_cols:
        mask = mask & np.isfinite(data[col])

    return data.loc[mask].copy()


def plot_columns(data, y_cols, labels=None, x_col=None, title="Data comparison", ylabel="Value", marker=True):
    """
    Plot selected columns in one figure.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to plot.
    y_cols : sequence of str
        Columns drawn as y series.
    labels : sequence of str, optional
        Legend labels; defaults to `y_cols`.
    x_col : str, optional
        Datetime x column; falls back to the sample index when absent.
    title : str, default 'Data comparison'
        Figure title.
    ylabel : str, default 'Value'
        Y-axis label.
    marker : bool, default True
        Draw point markers on each series (default `True`).

    Returns
    -------
    None — the figure is displayed and closed; it is not returned.

    Notes
    -----
    All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.
    """
    if labels is None:
        labels = y_cols

    if x_col is not None and x_col in data.columns:
        x = pd.to_datetime(data[x_col], errors="coerce")
        xlabel = x_col
    else:
        x = np.arange(len(data))
        xlabel = "Sample index"

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, col in enumerate(y_cols):
        if col in data.columns:
            ax.plot(x, data[col], label=labels[i], linewidth=2.2, linestyle="-", marker="o" if marker else None, markersize=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if x_col is not None and x_col in data.columns:
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_clearsky(data, model_cols, x_col=None, components=("ghi", "dni", "dhi")):
    """
    Plot multiple clear-sky models.

    Parameters
    ----------
    data : pandas.DataFrame
        Table holding clear-sky model outputs.
    model_cols : dict
        Maps a model name to `[ghi_col, dni_col, dhi_col]`.
    x_col : str, optional
        Datetime x column; falls back to the sample index when absent.
    components : tuple of str, default ('ghi', 'dni', 'dhi')
        Which components to draw, any of `"ghi"`, `"dni"`, `"dhi"`.

    Returns
    -------
    None — the figure is displayed and closed; it is not returned.

    Notes
    -----
    All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.
    """
    if x_col is not None and x_col in data.columns:
        x = pd.to_datetime(data[x_col], errors="coerce")
        xlabel = x_col
    else:
        x = np.arange(len(data))
        xlabel = "Sample Index"

    comp_index = {"ghi": 0, "dni": 1, "dhi": 2}
    comp_style = {"ghi": "-", "dni": "--", "dhi": "-."}
    comp_label = {"ghi": "GHI", "dni": "DNI", "dhi": "DHI"}
    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    for i, (model_name, cols) in enumerate(model_cols.items()):
        color = colors[i % len(colors)]

        for comp in components:
            if comp not in comp_index:
                continue

            idx = comp_index[comp]

            if idx < len(cols) and cols[idx] in data.columns:
                ax.plot(x, data[cols[idx]], color=color, linestyle=comp_style[comp], linewidth=1.5, label=f"{comp_label[comp]} {model_name}")

    ax.set_title("Clear-Sky Model Comparison", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"Irradiance [W m$^{-2}$]", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if x_col is not None and x_col in data.columns:
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_csd(data, ghi_col="ghi", clear_col="ghi_clear", csd_col="csd", max_points=3000):
    """
    Plot clear-sky detection results.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with observed, clear-sky and detection columns.
    ghi_col : str, default 'ghi'
        Observed GHI column.
    clear_col : str, default 'ghi_clear'
        Clear-sky GHI column.
    csd_col : str, default 'csd'
        Detection flag column (`0` = clear sky).
    max_points : int, default 3000
        Downsampling cap for long series (default 3000).

    Returns
    -------
    None — the figure is displayed and closed; it is not returned.

    Notes
    -----
    All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.
    """
    plot_data = data[[ghi_col, clear_col, csd_col]].dropna(subset=[ghi_col, clear_col]).copy()

    if len(plot_data) > max_points:
        step = max(1, len(plot_data) // max_points)
        plot_data = plot_data.iloc[::step].copy()

    x = np.arange(len(plot_data))
    csd_ghi = np.full(len(plot_data), np.nan)
    clear_mask = plot_data[csd_col] == 0
    csd_ghi[clear_mask.values] = plot_data.loc[clear_mask, ghi_col].values

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="w")

    ax.plot(x, plot_data[ghi_col].values, color="#1f77b4", linewidth=1.0, label="Observed GHI")
    ax.plot(x, plot_data[clear_col].values, color="black", linestyle=":", linewidth=1.0, label="Clear-Sky GHI")
    ax.plot(x, csd_ghi, color="black", linewidth=2.2, label="Detected Clear-Sky GHI")

    ax.set_title("Clear-Sky Detection", fontsize=18, fontweight="bold", family="serif")
    ax.set_xlabel("Sample Index", fontsize=14, fontweight="bold", family="serif")
    ax.set_ylabel(r"GHI [W m$^{-2}$]", fontsize=14, fontweight="bold", family="serif")
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    ax.grid(True, alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_series(data, obs_col, pred_col, obs_label="Observed", pred_label="Predicted", title="Series Comparison", ylabel="Value", threshold=0, max_value=None):
    """
    Plot two series for comparison.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with the two series to compare.
    obs_col : str
        Observed series column.
    pred_col : str
        Predicted series column.
    obs_label : str, default 'Observed'
        Legend label for the observed series.
    pred_label : str, default 'Predicted'
        Legend label for the predicted series.
    title : str, default 'Series Comparison'
        Figure title.
    ylabel : str, default 'Value'
        Y-axis label.
    threshold : float, default 0
        Only rows where either series exceeds `threshold` are plotted (default 0).
    max_value : float, optional
        When given, values above it are hidden (set to NaN).

    Returns
    -------
    None — the figure is displayed and closed; it is not returned.

    Notes
    -----
    Negative values in both series are hidden. Raises `RuntimeError` when no row exceeds `threshold`. All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.
    """
    plot_data = data[[obs_col, pred_col]].copy()
    plot_data.loc[plot_data[obs_col] < 0, obs_col] = np.nan
    plot_data.loc[plot_data[pred_col] < 0, pred_col] = np.nan

    if max_value is not None:
        plot_data.loc[plot_data[obs_col] > max_value, obs_col] = np.nan
        plot_data.loc[plot_data[pred_col] > max_value, pred_col] = np.nan

    active_mask = (plot_data[obs_col].fillna(0) > threshold) | (plot_data[pred_col].fillna(0) > threshold)

    if not active_mask.any():
        raise RuntimeError("No valid segment found for plotting.")

    plot_data = plot_data.loc[active_mask].reset_index(drop=True)
    x = np.arange(len(plot_data))

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.8, 3.2), facecolor="w")

    ax.plot(x, plot_data[obs_col], color="red", linewidth=1.8, linestyle="--", label=obs_label)
    ax.plot(x, plot_data[pred_col], color="blue", linewidth=1.8, linestyle="--", label=pred_label)

    ax.set_title(title, fontsize=18, fontweight="bold", family="serif")
    ax.set_xlabel("Sample Index", fontsize=14, fontweight="bold", family="serif")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", family="serif")
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.close()