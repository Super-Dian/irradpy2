# `irradpy2.tool` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (20): `cal_doy_year`, `cal_exth`, `cal_mae`, `cal_mbe`, `cal_metrics`, `cal_mse`, `cal_r2`, `cal_rmse`, `cal_zen`, `filter_sep_data`, `merge_columns`, `merge_series`, `plot_clearsky`, `plot_columns`, `plot_csd`, `plot_series`, `prepare_clearsky`, `read_csv`, `resample_series`, `select_series`

---

### `cal_doy_year(data, time_col='UTC', doy_col='doy', year_col='year')`

Calculate day of year and year.

**Parameters**

- `data` (*pandas.DataFrame*) — Input table.
- `time_col` (*str, default 'UTC'*) — Datetime column.
- `doy_col` (*str, default 'doy'*) — Output column name for day-of-year.
- `year_col` (*str, default 'year'*) — Output column name for the year.

**Returns**

pandas.DataFrame — copy of `data` with `doy_col` and `year_col` added.

### `cal_exth(data, time_col='UTC', zen_col='zen', exth_col='exth')`

Calculate horizontal extraterrestrial irradiance [W/m²].

**Parameters**

- `data` (*pandas.DataFrame*) — Input table containing a zenith-angle column.
- `time_col` (*str, default 'UTC'*) — Datetime column (day-of-year is derived from it).
- `zen_col` (*str, default 'zen'*) — Zenith angle column [degrees].
- `exth_col` (*str, default 'exth'*) — Output column name for horizontal extraterrestrial irradiance.

**Returns**

pandas.DataFrame — copy of `data` with `exth_col` added [W m-2].

**Notes**

Computes the extraterrestrial DNI (1366.1 W m-2 with the annual equation) times `cos(zen)`; set to 0 where `zen > 90°`.

### `cal_mae(data, obs_col='ghi', pred_col='ghi_pred')`

Calculate mean absolute error.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding observed and predicted columns.
- `obs_col` (*str, default 'ghi'*) — Observed-values column.
- `pred_col` (*str, default 'ghi_pred'*) — Predicted-values column.

**Returns**

float — mean absolute error.

### `cal_mbe(data, obs_col='ghi', pred_col='ghi_pred')`

Calculate mean bias error.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding observed and predicted columns.
- `obs_col` (*str, default 'ghi'*) — Observed-values column.
- `pred_col` (*str, default 'ghi_pred'*) — Predicted-values column.

**Returns**

float — mean bias error (`predicted - observed` convention).

**Notes**

Sign convention: `MBE = mean(predicted - observed)`. Positive MBE means overestimation, negative MBE underestimation. Some literature uses the opposite sign — check before comparing published values.

### `cal_metrics(data, obs_col='ghi', pred_col='ghi_pred')`

Calculate common evaluation metrics.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding observed and predicted columns.
- `obs_col` (*str, default 'ghi'*) — Observed-values column.
- `pred_col` (*str, default 'ghi_pred'*) — Predicted-values column.

**Returns**

dict — keys `MSE`, `RMSE`, `MAE`, `MBE`, `R2`.

### `cal_mse(data, obs_col='ghi', pred_col='ghi_pred')`

Calculate mean squared error.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding observed and predicted columns.
- `obs_col` (*str, default 'ghi'*) — Observed-values column.
- `pred_col` (*str, default 'ghi_pred'*) — Predicted-values column.

**Returns**

float — mean squared error.

**Notes**

Rows where either column is NaN are dropped before the mean.

### `cal_r2(data, obs_col='ghi', pred_col='ghi_pred')`

Calculate coefficient of determination.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding observed and predicted columns.
- `obs_col` (*str, default 'ghi'*) — Observed-values column.
- `pred_col` (*str, default 'ghi_pred'*) — Predicted-values column.

**Returns**

float — coefficient of determination, or `NaN` when observed variance is zero.

**Notes**

Returns `NaN` when the observed variance is zero.

### `cal_rmse(data, obs_col='ghi', pred_col='ghi_pred')`

Calculate root mean squared error.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding observed and predicted columns.
- `obs_col` (*str, default 'ghi'*) — Observed-values column.
- `pred_col` (*str, default 'ghi_pred'*) — Predicted-values column.

**Returns**

float — root mean squared error.

### `cal_zen(data, latitude, longitude, time_col=None, zen_col='zen')`

Calculate solar zenith angle in degrees.

**Parameters**

- `data` (*pandas.DataFrame*) — Input table; copied before the new column is added.
- `latitude` (*float*) — Site latitude [degrees].
- `longitude` (*float*) — Site longitude [degrees].
- `time_col` (*str, optional*) — Datetime column. When `None`, auto-detects among `Timestamp`, `UTC`, `UTC_time`, `time`.
- `zen_col` (*str, default 'zen'*) — Output column name for the zenith angle.

**Returns**

pandas.DataFrame — copy of `data` with `zen_col` added (zenith angle in degrees).

**Notes**

Returns a copy of `data` with `zen_col` filled in **degrees**. The clear-sky models take radians — convert with `np.radians(zen)`. Raises `ValueError` when no time column can be found.

### `filter_sep_data(data, latitude, longitude, time_col='Timestamp', zen_col='zen', ghi_col='Global', dni_col='Direct', dhi_col='Diffuse', pred_cols=None, max_zen=85, min_ghi=50, require_positive=True)`

Filter data for DHI and DNI separation evaluation.

**Parameters**

- `data` (*pandas.DataFrame*) — Table with observed components and optional predicted columns.
- `latitude` (*float*) — Site latitude [degrees]; used to compute `zen` when absent.
- `longitude` (*float*) — Site longitude [degrees].
- `time_col` (*str, default 'Timestamp'*) — Datetime column (default `"Timestamp"`).
- `zen_col` (*str, default 'zen'*) — Zenith angle column [degrees]; computed when missing.
- `ghi_col` (*str, default 'Global'*) — Observed GHI column.
- `dni_col` (*str, default 'Direct'*) — Observed DNI column.
- `dhi_col` (*str, default 'Diffuse'*) — Observed DHI column.
- `pred_cols` (*sequence of str, optional*) — Predicted columns that must be finite for a row to be kept.
- `max_zen` (*float, default 85*) — Rows kept only where `zen < max_zen` (default 85).
- `min_ghi` (*float, default 50*) — Rows kept only where `ghi > min_ghi` (default 50).
- `require_positive` (*bool, default True*) — When `True` (default), also require `dni > 0` and `dhi > 0`.

**Returns**

pandas.DataFrame — filtered copy of `data`.

### `merge_columns(data, values, columns, mask=None, fill_value=np.nan)`

Merge arrays into data columns.

**Parameters**

- `data` (*pandas.DataFrame*) — Destination table; copied before columns are added.
- `values` (*sequence of array-like*) — Arrays merged in order; matched positionally with `columns`.
- `columns` (*sequence of str*) — Destination column names.
- `mask` (*numpy.ndarray of bool, optional*) — Rows that receive `values`; rows outside the mask get `fill_value`. Defaults to all rows.
- `fill_value` (*float, default np.nan*) — Value written to rows outside `mask` (default `NaN`).

**Returns**

pandas.DataFrame — copy of `data` with the merged columns.

### `merge_series(data_list, time_col='UTC', dropna=True)`

Merge multiple time-series DataFrames by datetime index.

**Parameters**

- `data_list` (*sequence of pandas.DataFrame*) — Frames sharing a datetime index; concatenated column-wise.
- `time_col` (*str, default 'UTC'*) — Name given to the restored time column.
- `dropna` (*bool, default True*) — When `True` (default), rows with any NaN after concatenation are dropped.

**Returns**

pandas.DataFrame — concatenated frame with a restored `time_col` column.

### `plot_clearsky(data, model_cols, x_col=None, components=('ghi', 'dni', 'dhi'))`

Plot multiple clear-sky models.

**Parameters**

- `data` (*pandas.DataFrame*) — Table holding clear-sky model outputs.
- `model_cols` (*dict*) — Maps a model name to `[ghi_col, dni_col, dhi_col]`.
- `x_col` (*str, optional*) — Datetime x column; falls back to the sample index when absent.
- `components` (*tuple of str, default ('ghi', 'dni', 'dhi')*) — Which components to draw, any of `"ghi"`, `"dni"`, `"dhi"`.

**Returns**

None — the figure is displayed and closed; it is not returned.

**Notes**

All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.

### `plot_columns(data, y_cols, labels=None, x_col=None, title='Data comparison', ylabel='Value', marker=True)`

Plot selected columns in one figure.

**Parameters**

- `data` (*pandas.DataFrame*) — Table to plot.
- `y_cols` (*sequence of str*) — Columns drawn as y series.
- `labels` (*sequence of str, optional*) — Legend labels; defaults to `y_cols`.
- `x_col` (*str, optional*) — Datetime x column; falls back to the sample index when absent.
- `title` (*str, default 'Data comparison'*) — Figure title.
- `ylabel` (*str, default 'Value'*) — Y-axis label.
- `marker` (*bool, default True*) — Draw point markers on each series (default `True`).

**Returns**

None — the figure is displayed and closed; it is not returned.

**Notes**

All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.

### `plot_csd(data, ghi_col='ghi', clear_col='ghi_clear', csd_col='csd', max_points=3000)`

Plot clear-sky detection results.

**Parameters**

- `data` (*pandas.DataFrame*) — Table with observed, clear-sky and detection columns.
- `ghi_col` (*str, default 'ghi'*) — Observed GHI column.
- `clear_col` (*str, default 'ghi_clear'*) — Clear-sky GHI column.
- `csd_col` (*str, default 'csd'*) — Detection flag column (`0` = clear sky).
- `max_points` (*int, default 3000*) — Downsampling cap for long series (default 3000).

**Returns**

None — the figure is displayed and closed; it is not returned.

**Notes**

All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.

### `plot_series(data, obs_col, pred_col, obs_label='Observed', pred_label='Predicted', title='Series Comparison', ylabel='Value', threshold=0, max_value=None)`

Plot two series for comparison.

**Parameters**

- `data` (*pandas.DataFrame*) — Table with the two series to compare.
- `obs_col` (*str*) — Observed series column.
- `pred_col` (*str*) — Predicted series column.
- `obs_label` (*str, default 'Observed'*) — Legend label for the observed series.
- `pred_label` (*str, default 'Predicted'*) — Legend label for the predicted series.
- `title` (*str, default 'Series Comparison'*) — Figure title.
- `ylabel` (*str, default 'Value'*) — Y-axis label.
- `threshold` (*float, default 0*) — Only rows where either series exceeds `threshold` are plotted (default 0).
- `max_value` (*float, optional*) — When given, values above it are hidden (set to NaN).

**Returns**

None — the figure is displayed and closed; it is not returned.

**Notes**

Negative values in both series are hidden. Raises `RuntimeError` when no row exceeds `threshold`. All plotting functions use `matplotlib`. Each one builds a figure, calls `plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are displayed immediately and not returned; use an `Agg` backend or a monkey-patched `plt.show` to capture files instead of windows.

### `prepare_clearsky(ground_data, merra2_data, latitude, longitude, begin, end, freq='1min', source_prefix='merra2')`

Prepare clear-sky model inputs from ground and MERRA-2 data.

**Parameters**

- `ground_data` (*pandas.DataFrame*) — Ground observations with `ghi`/`dni`/`dhi` columns.
- `merra2_data` (*pandas.DataFrame*) — MERRA-2 (or compatible) rows with meteorological inputs and `ghi_clear`.
- `latitude` (*float*) — Site latitude [degrees].
- `longitude` (*float*) — Site longitude [degrees].
- `begin` (*str*) — Start date, `YYYY-MM-DD`.
- `end` (*str*) — End date, `YYYY-MM-DD`.
- `freq` (*str, default '1min'*) — Target index frequency (default `"1min"`).
- `source_prefix` (*str, default 'merra2'*) — Suffix for the renamed `ghi`/`ghi_clear` columns (default `"merra2"`).

**Returns**

pandas.DataFrame — aligned analysis table (`UTC`, `ghi`, `dni`, `dhi`, `sza`, atmospheric inputs).

**Notes**

Pipeline: detect time columns → build a complete time index at `freq` → align ground and MERRA-2 rows → interpolate meteorology → convert atmospheric quantities (`pressure`→`press` /100, `water_vapour`→`wv` /10, `ozone`→`ozone_input` /1000, `angstrom_exponent`→`ang_alpha`, `aod550`→`ang_beta`/`aod700`) → compute `sza` in **radians**. Typical output columns: `UTC`, `ghi`, `dni`, `dhi`, `sza`, `press`, `albedo`, `wv`, `aod700`, `ang_alpha`, `ang_beta`, `ozone_input`.

### `read_csv(file_path)`

Read a CSV file.

**Parameters**

- `file_path` (*str*) — Path to the CSV file to read.

**Returns**

pandas.DataFrame — the parsed CSV table.

**Notes**

Thin wrapper around `pandas.read_csv`. Downstream helpers such as `prepare_clearsky` auto-detect the time column among `UTC`, `UTC_time`, `time`.

### `resample_series(data, time_col='UTC', cols=None, freq='1h', agg='mean', lower_limit=None, label_offset=None)`

Resample selected time-series columns to a fixed frequency.

**Parameters**

- `data` (*pandas.DataFrame*) — Input table.
- `time_col` (*str, default 'UTC'*) — Datetime column.
- `cols` (*sequence of str, optional*) — Columns to keep; defaults to every non-time column.
- `freq` (*str, default '1h'*) — Pandas offset alias for the target frequency (default `"1h"`).
- `agg` (*str, default 'mean'*) — Aggregation: `"mean"` (default), `"sum"`, or `"median"`.
- `lower_limit` (*float, optional*) — Clipping bound passed through to `select_series`.
- `label_offset` (*str, optional*) — Pandas `Timedelta` added to the bin labels after resampling.

**Returns**

pandas.DataFrame — series resampled to `freq`.

### `select_series(data, time_col='UTC', cols=None, lower_limit=None)`

Select numeric time-series columns and set datetime index.

**Parameters**

- `data` (*pandas.DataFrame*) — Input table.
- `time_col` (*str, default 'UTC'*) — Datetime column; parsed to timezone-naive UTC and used as index.
- `cols` (*sequence of str, optional*) — Columns to keep; defaults to every non-time column.
- `lower_limit` (*float, optional*) — When given, values below it are raised to it (clipped).

**Returns**

pandas.DataFrame — time-indexed, sorted subset of the selected columns.

**Notes**

Selected columns are coerced to numeric; the returned frame is sorted by time.

