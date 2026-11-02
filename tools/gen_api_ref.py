#!/usr/bin/env python3
"""Generate per-function API reference pages from irradpy2 docstrings.

Read-only against the source tree; emits Markdown per module with one
entry per public function (signature, summary, parameters, returns,
references, example) in numpydoc-derived form.

Run as ``python tools/gen_api_ref.py`` from anywhere — paths resolve
relative to this file (``<irradpy2_project>/tools/``).
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # <irradpy2_project>/
SRC = ROOT / "src" / "irradpy2"
OUT = ROOT / "docs" / "reference"

MODULES = {
    "clear_sky": "clear-sky.md",
    "csd": "csd.md",
    "separation": "separation.md",
    "forecast": "forecast.md",
    "getirrad": "getirrad.md",
    "getgridded": "getgridded.md",
    "tool": "tool.md",
}

# Public entry points that are not listed in the package __all__ but are
# documented as public API (getgridded downloaders / registry helper).
EXTRA_PUBLIC = {
    "getgridded": {
        "download_era5_point", "download_era5_area",
        "download_cams_point", "download_cams_area",
        "list_merra2_collections",
    },
}

# Hand-written summary/parameter/returns/notes supplements for functions
# whose source docstrings are one-liners (tool.py, forecast.py) or empty
# (getgridded downloaders). Semantics verified against the source bodies
# and merged from docs/user-guide/data-tools.md, so a regeneration never
# loses them. Supplements fill only what the docstring lacks: a non-empty
# docstring summary or section always wins.
_SUP_PLOT = (
    "All plotting functions use `matplotlib`. Each one builds a figure, calls "
    "`plt.tight_layout()`, then `plt.show()` and `plt.close()`. Figures are "
    "displayed immediately and not returned; use an `Agg` backend or a "
    "monkey-patched `plt.show` to capture files instead of windows."
)

# ---- forecast: shared fragments (verified against forecast/forecast.py) ----

_FC_INPUTS = [
    ("datetime", "numpy.ndarray",
     "Datetime sequence, shape `(n,)` or `(n, 1)`; parsed with `pandas.to_datetime` "
     "(unparseable rows are dropped)."),
    ("ghi", "numpy.ndarray",
     "Global horizontal irradiance, shape `(n,)` or `(n, 1)`, same length as "
     "`datetime` [W m-2]."),
]

_FC_RETURNS = (
    "tuple of pandas.DataFrame — `(ori_data, pre_data)`. `ori_data` holds the "
    "observed evaluation window (`Timestamp`, `Global`); `pre_data` holds the "
    "predictions on the same timestamps (`Timestamp`, `pre_Global`)."
)

_FC_NOTES = (
    "Shared preprocessing: both arguments must be `numpy.ndarray` (TypeError "
    "otherwise) with matching lengths and shapes `(n,)` or `(n, 1)`; invalid "
    "timestamps are dropped, sentinel values `-9999/-999/-99` are gap-filled "
    "forward then backward, negative GHI is clipped to 0, and the series is "
    "sorted by time. `ValueError` is raised when GHI is entirely missing or the "
    "series is too short."
)

_FC_DL_PARAMS = [
    _FC_INPUTS[0], _FC_INPUTS[1],
    ("seq_len", "int, default 72", "Input window length in samples fed to the model."),
    ("horizon", "int, default 6",
     "Gap in samples between the end of the input window and the target sample."),
    ("epochs", "int, default 20",
     "Training epochs; the best validation-loss state is kept."),
    ("batch_size", "int, default 64", "Mini-batch size."),
    ("lr", "float, default 0.001", "Adam learning rate."),
]

_FC_DL_NOTES = (
    "Requires `torch`. The shared pipeline makes a chronological 70/15/15 "
    "train/validation/test split (no shuffling), scales inputs with `MinMaxScaler`, "
    "divides targets by the training-set maximum, trains with Adam and keeps the "
    "best validation-loss epoch (per-epoch progress is printed to stdout). The "
    "returned frames cover the test window; predictions are clipped at 0. "
    + _FC_NOTES
)

# ---- getgridded: shared fragments (verified against era5.py/cams.py/merra2.py) ----

_GR_CREDS = (
    "Uses the CDS `cdsapi` client: passing only `api_key` targets the public "
    "endpoint (CDS for ERA5, ADS for CAMS); otherwise settings come from "
    "`~/.cdsapirc`, with `api_url`/`api_key` overriding individual values."
)

_GR_START = ("start", "Any",
             "Start of the period; anything `pandas.Timestamp` accepts, converted "
             "to UTC (tz-aware values are converted, naive values are taken as-is).")
_GR_END = ("end", "Any",
           "End of the period (same parsing as `start`); must not precede `start` "
           "(`ValueError`).")

SUPPLEMENTS = {
    "tool": {
        "read_csv": {
            "returns": "pandas.DataFrame — the parsed CSV table.",
            "parameters": [("file_path", "str", "Path to the CSV file to read.")],
            "notes": (
                "Thin wrapper around `pandas.read_csv`. Downstream helpers such as "
                "`prepare_clearsky` auto-detect the time column among "
                "`UTC`, `UTC_time`, `time`."
            ),
        },
        "cal_zen": {
            "returns": "pandas.DataFrame — copy of `data` with `zen_col` added (zenith angle in degrees).",
            "parameters": [
                ("data", "pandas.DataFrame", "Input table; copied before the new column is added."),
                ("latitude", "float", "Site latitude [degrees]."),
                ("longitude", "float", "Site longitude [degrees]."),
                ("time_col", "str, optional", "Datetime column. When `None`, auto-detects among `Timestamp`, `UTC`, `UTC_time`, `time`."),
                ("zen_col", "str", "Output column name for the zenith angle."),
            ],
            "notes": (
                "Returns a copy of `data` with `zen_col` filled in **degrees**. "
                "The clear-sky models take radians — convert with `np.radians(zen)`. "
                "Raises `ValueError` when no time column can be found."
            ),
        },
        "cal_exth": {
            "returns": "pandas.DataFrame — copy of `data` with `exth_col` added [W m-2].",
            "parameters": [
                ("data", "pandas.DataFrame", "Input table containing a zenith-angle column."),
                ("time_col", "str", "Datetime column (day-of-year is derived from it)."),
                ("zen_col", "str", "Zenith angle column [degrees]."),
                ("exth_col", "str", "Output column name for horizontal extraterrestrial irradiance."),
            ],
            "notes": (
                "Computes the extraterrestrial DNI (1366.1 W m-2 with the annual "
                "equation) times `cos(zen)`; set to 0 where `zen > 90°`."
            ),
        },
        "cal_doy_year": {
            "returns": "pandas.DataFrame — copy of `data` with `doy_col` and `year_col` added.",
            "parameters": [
                ("data", "pandas.DataFrame", "Input table."),
                ("time_col", "str", "Datetime column."),
                ("doy_col", "str", "Output column name for day-of-year."),
                ("year_col", "str", "Output column name for the year."),
            ],
        },
        "cal_mse": {
            "returns": "float — mean squared error.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding observed and predicted columns."),
                ("obs_col", "str", "Observed-values column."),
                ("pred_col", "str", "Predicted-values column."),
            ],
            "notes": "Rows where either column is NaN are dropped before the mean.",
        },
        "cal_rmse": {
            "returns": "float — root mean squared error.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding observed and predicted columns."),
                ("obs_col", "str", "Observed-values column."),
                ("pred_col", "str", "Predicted-values column."),
            ],
        },
        "cal_mae": {
            "returns": "float — mean absolute error.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding observed and predicted columns."),
                ("obs_col", "str", "Observed-values column."),
                ("pred_col", "str", "Predicted-values column."),
            ],
        },
        "cal_mbe": {
            "returns": "float — mean bias error (`predicted - observed` convention).",
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding observed and predicted columns."),
                ("obs_col", "str", "Observed-values column."),
                ("pred_col", "str", "Predicted-values column."),
            ],
            "notes": (
                "Sign convention: `MBE = mean(predicted - observed)`. Positive MBE "
                "means overestimation, negative MBE underestimation. Some literature "
                "uses the opposite sign — check before comparing published values."
            ),
        },
        "cal_r2": {
            "returns": "float — coefficient of determination, or `NaN` when observed variance is zero.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding observed and predicted columns."),
                ("obs_col", "str", "Observed-values column."),
                ("pred_col", "str", "Predicted-values column."),
            ],
            "notes": "Returns `NaN` when the observed variance is zero.",
        },
        "cal_metrics": {
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding observed and predicted columns."),
                ("obs_col", "str", "Observed-values column."),
                ("pred_col", "str", "Predicted-values column."),
            ],
            "returns": "dict — keys `MSE`, `RMSE`, `MAE`, `MBE`, `R2`.",
        },
        "merge_columns": {
            "returns": "pandas.DataFrame — copy of `data` with the merged columns.",
            "parameters": [
                ("data", "pandas.DataFrame", "Destination table; copied before columns are added."),
                ("values", "sequence of array-like", "Arrays merged in order; matched positionally with `columns`."),
                ("columns", "sequence of str", "Destination column names."),
                ("mask", "numpy.ndarray of bool, optional", "Rows that receive `values`; rows outside the mask get `fill_value`. Defaults to all rows."),
                ("fill_value", "float", "Value written to rows outside `mask` (default `NaN`)."),
            ],
        },
        "select_series": {
            "returns": "pandas.DataFrame — time-indexed, sorted subset of the selected columns.",
            "parameters": [
                ("data", "pandas.DataFrame", "Input table."),
                ("time_col", "str", "Datetime column; parsed to timezone-naive UTC and used as index."),
                ("cols", "sequence of str, optional", "Columns to keep; defaults to every non-time column."),
                ("lower_limit", "float, optional", "When given, values below it are raised to it (clipped)."),
            ],
            "notes": "Selected columns are coerced to numeric; the returned frame is sorted by time.",
        },
        "resample_series": {
            "returns": "pandas.DataFrame — series resampled to `freq`.",
            "parameters": [
                ("data", "pandas.DataFrame", "Input table."),
                ("time_col", "str", "Datetime column."),
                ("cols", "sequence of str, optional", "Columns to keep; defaults to every non-time column."),
                ("freq", "str", "Pandas offset alias for the target frequency (default `\"1h\"`)."),
                ("agg", "str", "Aggregation: `\"mean\"` (default), `\"sum\"`, or `\"median\"`."),
                ("lower_limit", "float, optional", "Clipping bound passed through to `select_series`."),
                ("label_offset", "str, optional", "Pandas `Timedelta` added to the bin labels after resampling."),
            ],
        },
        "merge_series": {
            "returns": "pandas.DataFrame — concatenated frame with a restored `time_col` column.",
            "parameters": [
                ("data_list", "sequence of pandas.DataFrame", "Frames sharing a datetime index; concatenated column-wise."),
                ("time_col", "str", "Name given to the restored time column."),
                ("dropna", "bool", "When `True` (default), rows with any NaN after concatenation are dropped."),
            ],
        },
        "prepare_clearsky": {
            "returns": "pandas.DataFrame — aligned analysis table (`UTC`, `ghi`, `dni`, `dhi`, `sza`, atmospheric inputs).",
            "parameters": [
                ("ground_data", "pandas.DataFrame", "Ground observations with `ghi`/`dni`/`dhi` columns."),
                ("merra2_data", "pandas.DataFrame", "MERRA-2 (or compatible) rows with meteorological inputs and `ghi_clear`."),
                ("latitude", "float", "Site latitude [degrees]."),
                ("longitude", "float", "Site longitude [degrees]."),
                ("begin", "str", "Start date, `YYYY-MM-DD`."),
                ("end", "str", "End date, `YYYY-MM-DD`."),
                ("freq", "str", "Target index frequency (default `\"1min\"`)."),
                ("source_prefix", "str", "Suffix for the renamed `ghi`/`ghi_clear` columns (default `\"merra2\"`)."),
            ],
            "notes": (
                "Pipeline: detect time columns → build a complete time index at "
                "`freq` → align ground and MERRA-2 rows → interpolate meteorology → "
                "convert atmospheric quantities (`pressure`→`press` /100, "
                "`water_vapour`→`wv` /10, `ozone`→`ozone_input` /1000, "
                "`angstrom_exponent`→`ang_alpha`, `aod550`→`ang_beta`/`aod700`) → "
                "compute `sza` in **radians**. Typical output columns: `UTC`, `ghi`, "
                "`dni`, `dhi`, `sza`, `press`, `albedo`, `wv`, `aod700`, `ang_alpha`, "
                "`ang_beta`, `ozone_input`."
            ),
        },
        "filter_sep_data": {
            "returns": "pandas.DataFrame — filtered copy of `data`.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table with observed components and optional predicted columns."),
                ("latitude", "float", "Site latitude [degrees]; used to compute `zen` when absent."),
                ("longitude", "float", "Site longitude [degrees]."),
                ("time_col", "str", "Datetime column (default `\"Timestamp\"`)."),
                ("zen_col", "str", "Zenith angle column [degrees]; computed when missing."),
                ("ghi_col", "str", "Observed GHI column."),
                ("dni_col", "str", "Observed DNI column."),
                ("dhi_col", "str", "Observed DHI column."),
                ("pred_cols", "sequence of str, optional", "Predicted columns that must be finite for a row to be kept."),
                ("max_zen", "float", "Rows kept only where `zen < max_zen` (default 85)."),
                ("min_ghi", "float", "Rows kept only where `ghi > min_ghi` (default 50)."),
                ("require_positive", "bool", "When `True` (default), also require `dni > 0` and `dhi > 0`."),
            ],
        },
        "plot_columns": {
            "returns": "None — the figure is displayed and closed; it is not returned.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table to plot."),
                ("y_cols", "sequence of str", "Columns drawn as y series."),
                ("labels", "sequence of str, optional", "Legend labels; defaults to `y_cols`."),
                ("x_col", "str, optional", "Datetime x column; falls back to the sample index when absent."),
                ("title", "str", "Figure title."),
                ("ylabel", "str", "Y-axis label."),
                ("marker", "bool", "Draw point markers on each series (default `True`)."),
            ],
            "notes": _SUP_PLOT,
        },
        "plot_clearsky": {
            "returns": "None — the figure is displayed and closed; it is not returned.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table holding clear-sky model outputs."),
                ("model_cols", "dict", "Maps a model name to `[ghi_col, dni_col, dhi_col]`."),
                ("x_col", "str, optional", "Datetime x column; falls back to the sample index when absent."),
                ("components", "tuple of str", "Which components to draw, any of `\"ghi\"`, `\"dni\"`, `\"dhi\"`."),
            ],
            "notes": _SUP_PLOT,
        },
        "plot_csd": {
            "returns": "None — the figure is displayed and closed; it is not returned.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table with observed, clear-sky and detection columns."),
                ("ghi_col", "str", "Observed GHI column."),
                ("clear_col", "str", "Clear-sky GHI column."),
                ("csd_col", "str", "Detection flag column (`0` = clear sky)."),
                ("max_points", "int", "Downsampling cap for long series (default 3000)."),
            ],
            "notes": _SUP_PLOT,
        },
        "plot_series": {
            "returns": "None — the figure is displayed and closed; it is not returned.",
            "parameters": [
                ("data", "pandas.DataFrame", "Table with the two series to compare."),
                ("obs_col", "str", "Observed series column."),
                ("pred_col", "str", "Predicted series column."),
                ("obs_label", "str", "Legend label for the observed series."),
                ("pred_label", "str", "Legend label for the predicted series."),
                ("title", "str", "Figure title."),
                ("ylabel", "str", "Y-axis label."),
                ("threshold", "float", "Only rows where either series exceeds `threshold` are plotted (default 0)."),
                ("max_value", "float, optional", "When given, values above it are hidden (set to NaN)."),
            ],
            "notes": (
                "Negative values in both series are hidden. Raises `RuntimeError` "
                "when no row exceeds `threshold`. " + _SUP_PLOT
            ),
        },
    },
    "forecast": {
        "forecast_persistence": {
            "parameters": _FC_INPUTS + [
                ("horizon", "int, default 6",
                 "Forecast lead in samples: the prediction at time `t` is the "
                 "observation at `t - horizon`."),
                ("test_ratio", "float, default 0.15",
                 "Trailing fraction of the time-sorted series used as the "
                 "evaluation window."),
            ],
            "returns": _FC_RETURNS,
            "notes": (
                "The evaluation window starts at "
                "`max(int(n * (1 - test_ratio)), horizon)`; predictions are "
                "clipped at 0. " + _FC_NOTES
            ),
        },
        "forecast_arima": {
            "parameters": _FC_INPUTS + [
                ("order", "tuple of int, default (2, 1, 2)",
                 "ARIMA (p, d, q) order passed to "
                 "`statsmodels.tsa.arima.model.ARIMA`."),
                ("test_ratio", "float, default 0.15",
                 "Trailing fraction of the time-sorted series used as the test "
                 "window."),
            ],
            "returns": _FC_RETURNS,
            "notes": (
                "Fits ARIMA on the first `(1 - test_ratio)` fraction (requires "
                "`statsmodels` and at least 20 training samples, `ValueError` "
                "otherwise) and forecasts exactly `len(test)` steps; predictions "
                "are clipped at 0. " + _FC_NOTES
            ),
        },
        "forecast_rnn": {
            "parameters": _FC_DL_PARAMS + [
                ("hidden", "int, default 128", "Hidden units of the RNN layer."),
            ],
            "returns": _FC_RETURNS,
            "notes": _FC_DL_NOTES,
        },
        "forecast_lstm": {
            "parameters": _FC_DL_PARAMS + [
                ("hidden", "int, default 128", "Hidden units of the LSTM layer."),
            ],
            "returns": _FC_RETURNS,
            "notes": _FC_DL_NOTES,
        },
        "forecast_informer": {
            "parameters": _FC_DL_PARAMS + [
                ("d_model", "int, default 128",
                 "Transformer embedding dimension; must be divisible by `nhead` "
                 "(`ValueError` otherwise)."),
                ("nhead", "int, default 4", "Number of attention heads."),
                ("num_layers", "int, default 2",
                 "Number of transformer encoder layers."),
            ],
            "returns": _FC_RETURNS,
            "notes": (
                "A lightweight Informer-style Transformer, not the full original "
                "Informer implementation. " + _FC_DL_NOTES
            ),
        },
    },
    "getgridded": {
        "download_era5_point": {
            "summary": (
                "Download ERA5 reanalysis for one point from the CDS API and "
                "write it to CSV."
            ),
            "parameters": [
                ("lat", "float", "Latitude [degrees]."),
                ("lon", "float", "Longitude [degrees]."),
                _GR_START, _GR_END,
                ("variables", "sequence of str, optional",
                 "CDS variable names or local aliases (`ghi`, `temp_air`, ... "
                 "resolved via `INPUT_VARIABLE_MAP`); defaults to "
                 "`[\"ssrd\", \"strd\", \"t2m\", \"sp\", \"tcc\", \"tp\"]`."),
                ("save_path", "str or pathlib.Path, optional",
                 "Output CSV path; defaults to "
                 "`era5_<lat>_<lon>_<start>_<end>.csv`. The `.csv` suffix is "
                 "forced."),
                ("api_url", "str, optional", "CDS API endpoint override."),
                ("api_key", "str, optional", "CDS API key override."),
                ("quiet", "bool, default True",
                 "Suppress `cdsapi` client progress output."),
                ("map_variables", "bool, default True",
                 "Rename columns to standard names (`ghi`, `longwave_down`, "
                 "`temp_air`, `pressure`, `cloud_fraction`, `precipitation`, ...)."),
                ("convert_units", "bool, default True",
                 "Apply unit conversions (e.g. K -> degC, m -> mm)."),
            ],
            "returns": (
                "PointResult dataclass — `data` (pandas.DataFrame with a "
                "`UTC_time` column plus the requested variables), `metadata` "
                "(dict: request, columns, coverage), `csv_path`, `metadata_path` "
                "(pathlib.Path). The CSV and a `<stem>_metadata.json` sidecar are "
                "written to disk."
            ),
            "notes": (
                "Issues one CDS retrieve covering the whole range, then parses "
                "and rewrites the CSV. " + _GR_CREDS
            ),
        },
        "download_era5_area": {
            "summary": (
                "Download ERA5 reanalysis for a bounding box from the CDS API as "
                "monthly GRIB files."
            ),
            "parameters": [
                ("lat_1", "float", "First corner latitude [degrees]."),
                ("lon_1", "float", "First corner longitude [degrees]."),
                ("lat_2", "float",
                 "Opposite corner latitude [degrees]; must differ from `lat_1` "
                 "(`ValueError`), order versus `lat_1` is irrelevant (sorted "
                 "internally)."),
                ("lon_2", "float",
                 "Opposite corner longitude [degrees]; must differ from `lon_1` "
                 "(`ValueError`)."),
                _GR_START, _GR_END,
                ("variables", "sequence of str, optional",
                 "CDS variable names or local aliases; defaults to "
                 "`[\"ssrd\", \"strd\", \"t2m\", \"sp\", \"tcc\"]`."),
                ("output_dir", "str or pathlib.Path, default \"ERA5_data\"",
                 "Directory for GRIB parts and metadata (created if missing)."),
                ("file_prefix", "str, optional",
                 "Output file prefix; defaults to "
                 "`era5_area_<lat_1>_<lon_1>_<lat_2>_<lon_2>`."),
                ("api_url", "str, optional", "CDS API endpoint override."),
                ("api_key", "str, optional", "CDS API key override."),
                ("quiet", "bool, default True",
                 "Suppress `cdsapi` client progress output."),
            ],
            "returns": (
                "AreaResult dataclass — `grib_paths` (list of pathlib.Path, one "
                "GRIB part per month), `metadata` (dict), `metadata_path`. A "
                "`<prefix>_metadata.json` sidecar is written."
            ),
            "notes": (
                "The range is split into month chunks, each retrieved as its own "
                "GRIB file. " + _GR_CREDS
            ),
        },
        "download_era5": {
            "summary": (
                "Unified ERA5 entry point; dispatches to the point or area "
                "downloader by `mode`."
            ),
            "parameters": [
                ("mode", "str, default \"point\"",
                 "`\"point\"` or `\"area\"` (case-insensitive; `ValueError` "
                 "otherwise)."),
                ("start", "Any", "Start of the period; required (`ValueError` when `None`)."),
                ("end", "Any", "End of the period; required."),
                ("variables", "sequence of str, optional",
                 "Forwarded to the chosen downloader."),
                ("api_url", "str, optional", "CDS API endpoint override."),
                ("api_key", "str, optional", "CDS API key override."),
                ("quiet", "bool, default True",
                 "Suppress `cdsapi` client progress output."),
                ("**kwargs", "keyword arguments",
                 "`mode=\"point\"` requires `lat`, `lon` (optional `save_path`, "
                 "`map_variables`, `convert_units`); `mode=\"area\"` requires "
                 "`lat_1`, `lon_1`, `lat_2`, `lon_2` (optional `output_dir`, "
                 "`file_prefix`). Missing required keys raise `ValueError`."),
            ],
            "returns": (
                "PointResult or AreaResult — whichever dataclass the selected "
                "downloader returns."
            ),
            "notes": _GR_CREDS,
        },
        "download_cams_point": {
            "summary": (
                "Download CAMS satellite-derived radiation for one point from the "
                "ADS API."
            ),
            "parameters": [
                ("lat", "float", "Latitude [degrees]."),
                ("lon", "float", "Longitude [degrees]."),
                _GR_START, _GR_END,
                ("save_path", "str or pathlib.Path, optional",
                 "Output path; defaults to "
                 "`cams_<lat>_<lon>_<start>_<end>.csv`. The suffix follows "
                 "`file_format` (`.csv` or `.nc`)."),
                ("file_format", "str, default \"csv\"",
                 "`\"csv\"` or `\"netcdf\"` (`ValueError` otherwise)."),
                ("sky_type", "str, default \"observed_cloud\"",
                 "`\"clear\"` or `\"observed_cloud\"` (`ValueError` otherwise)."),
                ("time_step", "str, default \"1min\"",
                 "One of `\"1min\"`, `\"15min\"`, `\"1h\"`, `\"1d\"`, `\"1M\"` "
                 "(`ValueError` otherwise)."),
                ("time_ref", "str, default \"UT\"",
                 "`\"UT\"` (universal time) or `\"TST\"` (true solar time) "
                 "(`ValueError` otherwise)."),
                ("altitude", "float, optional",
                 "Site altitude [m]; when `None` the service default (-999) is "
                 "sent."),
                ("quiet", "bool, default True",
                 "Suppress `cdsapi` client progress output."),
                ("parse_csv", "bool, default True",
                 "CSV only: parse the download, standardize it and rewrite it "
                 "with a leading `UTC_time` column. Ignored for NetCDF (`data` "
                 "stays `None`)."),
                ("map_variables", "bool, default True",
                 "Rename parsed columns to standard names (`ghi`, `dni`, `dhi`, "
                 "`ghi_clear`, ...) via `VARIABLE_MAP`."),
                ("api_url", "str, optional", "ADS API endpoint override."),
                ("api_key", "str, optional", "ADS API key override."),
            ],
            "returns": (
                "CamsPointResult dataclass — `data` (pandas.DataFrame when "
                "parsed, else `None`), `metadata` (dict), `data_path`, "
                "`metadata_path` (pathlib.Path). A `<stem>_metadata.json` "
                "sidecar is written."
            ),
            "notes": (
                "A parsed CSV is rewritten in place with columns ordered "
                "`UTC_time`, then GHI/DNI/DHI (observed and clear-sky), then the "
                "rest. " + _GR_CREDS
            ),
        },
        "download_cams_area": {
            "summary": (
                "Download CAMS gridded satellite radiation for a period as "
                "NetCDF files (one per variable and month)."
            ),
            "parameters": [
                ("lat_1", "float, optional",
                 "Corner latitude [degrees]. When any of the four corners is "
                 "`None` the whole-domain request is used (no area restriction); "
                 "when all four are given they are sorted internally to "
                 "[north, west, south, east]."),
                ("lon_1", "float, optional", "Corner longitude [degrees]; see `lat_1`."),
                ("lat_2", "float, optional", "Corner latitude [degrees]; see `lat_1`."),
                ("lon_2", "float, optional", "Corner longitude [degrees]; see `lat_1`."),
                ("start", "Any", "Start of the period; required (`ValueError` when `None`)."),
                ("end", "Any",
                 "End of the period; required; must not precede `start` "
                 "(`ValueError`)."),
                ("variables", "sequence of str, optional",
                 "Local aliases (`ghi`, `dni`, `dhi`, `bhi`) resolved via "
                 "`AREA_VARIABLE_MAP`; defaults to `[\"ghi\", \"dni\", \"dhi\"]`."),
                ("sky_type", "str, default \"observed_cloud\"",
                 "`\"clear\"` or `\"observed_cloud\"` (`ValueError` otherwise)."),
                ("output_dir", "str or pathlib.Path, default \"CAMS_data\"",
                 "Directory for NetCDF parts and metadata (created if missing)."),
                ("file_prefix", "str, optional",
                 "Output file prefix; defaults to `cams_area_<corners>` when an "
                 "area is given, else `cams_area`."),
                ("quiet", "bool, default True",
                 "Suppress `cdsapi` client progress output."),
                ("api_url", "str, optional", "ADS API endpoint override."),
                ("api_key", "str, optional", "ADS API key override."),
            ],
            "returns": (
                "CamsAreaResult dataclass — `netcdf_paths` (list of "
                "pathlib.Path), `metadata` (dict), `metadata_path`. A "
                "`<prefix>_metadata.json` sidecar is written."
            ),
            "notes": (
                "One retrieve is issued per variable and calendar month. "
                + _GR_CREDS
            ),
        },
        "download_cams": {
            "summary": (
                "Unified CAMS entry point; dispatches to the point or area "
                "downloader by `mode`."
            ),
            "parameters": [
                ("mode", "str, default \"point\"",
                 "`\"point\"` or `\"area\"` (case-insensitive; `ValueError` "
                 "otherwise)."),
                ("quiet", "bool, default True",
                 "Suppress `cdsapi` client progress output."),
                ("api_url", "str, optional", "ADS API endpoint override."),
                ("api_key", "str, optional", "ADS API key override."),
                ("**kwargs", "keyword arguments",
                 "`mode=\"point\"` requires `lat`, `lon`, `start`, `end` "
                 "(optional `save_path`, `file_format`, `sky_type`, `time_step`, "
                 "`time_ref`, `altitude`, `parse_csv`, `map_variables`); "
                 "`mode=\"area\"` requires `start`, `end` (optional `lat_1`, "
                 "`lon_1`, `lat_2`, `lon_2`, `variables`, `sky_type`, "
                 "`output_dir`, `file_prefix`). Missing required keys raise "
                 "`ValueError`."),
            ],
            "returns": (
                "CamsPointResult or CamsAreaResult — whichever dataclass the "
                "selected downloader returns."
            ),
            "notes": _GR_CREDS,
        },
        "download_merra2": {
            "parameters": [
                ("collection_names", "str or list of str, optional",
                 "MERRA-2 collection(s) from the built-in registry (`rad`, "
                 "`slv`, `aer`, `asm`, `dewpt`, `evspsbl`, `hur`, `huss`, "
                 "`phis`, `pr`, `prc`, `prbc`, `prmax`, `prsn`, `rls`, "
                 "`sftgif`, `sftkf`, `sftlf`, `sftof`, `shg`, `sic`, `tas`, "
                 "`tasmax`, `tasmin`, `uas`, `vas`). Point mode defaults to "
                 "`[\"rad\", \"slv\", \"aer\"]`; area mode requires it "
                 "explicitly (`ValueError`) to avoid huge default downloads."),
                ("initial_year", "int, optional",
                 "First year. Required in area mode (`ValueError`); in point "
                 "mode used only when `start`/`end` are not given."),
                ("final_year", "int, optional",
                 "Last year; same rules as `initial_year`."),
                ("initial_month", "int, default 1",
                 "First month of the year/month/day range."),
                ("final_month", "int, default 12",
                 "Last month of the year/month/day range."),
                ("initial_day", "int, default 1",
                 "First day of the year/month/day range."),
                ("final_day", "int, optional",
                 "Last day. Required in area mode and whenever the "
                 "year/month/day range form is used (`ValueError`)."),
                ("lat_1", "float, default -90",
                 "Area corner latitude [degrees] (area mode); must satisfy "
                 "`lat_1 <= lat_2` (`RuntimeError` otherwise)."),
                ("lon_1", "float, default -180",
                 "Area corner longitude [degrees] (area mode); must satisfy "
                 "`lon_1 <= lon_2` (`RuntimeError` otherwise)."),
                ("lat_2", "float, default 90",
                 "Opposite area corner latitude [degrees] (area mode)."),
                ("lon_2", "float, default 180",
                 "Opposite area corner longitude [degrees] (area mode)."),
                ("lat", "float, optional",
                 "Point latitude [degrees]. Required in point mode "
                 "(`ValueError`)."),
                ("lon", "float, optional",
                 "Point longitude [degrees]. Required in point mode "
                 "(`ValueError`)."),
                ("mode", "str, default \"area\"",
                 "`\"area\"` (NetCDF subset by four corners) or `\"point\"` "
                 "(single-point CSV time series); `ValueError` otherwise."),
                ("start", "Any, optional",
                 "Point-mode period start (e.g. `\"YYYY-MM-DD\"`); `start` and "
                 "`end` must be given together. Date-only values start at "
                 "00:00:00 UTC."),
                ("end", "Any, optional",
                 "Point-mode period end; date-only values extend to 23:59:59."),
                ("variables", "sequence or dict, optional",
                 "Variable subset. A string or list is allowed for a single "
                 "collection; with multiple collections pass a dict "
                 "`{collection: [\"VAR\", ...]}` (`ValueError` otherwise). Names "
                 "must exist in the collection's registry (`ValueError`)."),
                ("output_dir", "str or pathlib.Path, default \"MERRA2_data\"",
                 "Area-mode output directory (created if missing); downloads "
                 "land in a dated subdirectory."),
                ("save_path", "str or pathlib.Path, optional",
                 "Point mode: also write the resulting table to this CSV. "
                 "Ignored in area mode."),
                ("auth", "dict, optional",
                 "Earthdata credentials `{\"uid\": ..., \"password\": ...}`; "
                 "alternative to `username`/`password`. Required in both modes "
                 "(`ValueError` when absent)."),
                ("username", "str, optional",
                 "Earthdata Login username (combined with `password` into "
                 "`auth`)."),
                ("password", "str, optional", "Earthdata Login password."),
                ("merge_timelapse", "str, default \"none\"",
                 "Area mode: `\"none\"`, `\"daily\"`, `\"monthly\"` or "
                 "`\"yearly\"` — merge downloaded files afterwards "
                 "(`RuntimeError` otherwise)."),
                ("thread_num", "int, default 1",
                 "Area mode: parallel download threads."),
                ("map_variables", "bool, default True",
                 "Rename columns to standard names (`VARIABLE_MAP`)."),
                ("timeout", "tuple of float, default (30, 300)",
                 "Point mode: `(connect, read)` timeouts in seconds for OPeNDAP "
                 "requests."),
            ],
            "returns": (
                "Depends on `mode`: area mode returns `str` — the directory "
                "holding the downloaded (and merged) NetCDF files; point mode "
                "returns `(pandas.DataFrame, dict)` — the time-series table "
                "(with `time` as the first column) and metadata including "
                "per-variable units."
            ),
            "notes": (
                "Requires NASA Earthdata Login credentials. Area mode validates "
                "`collection_names`, `initial_year`, `final_year` and `final_day` "
                "up front (`ValueError`) and raises `RuntimeError` when the "
                "bounding box or `merge_timelapse` is invalid, or after three "
                "failed download retries."
            ),
        },
        "list_merra2_collections": {
            "returns": (
                "dict — a copy of the built-in registry "
                "`{collection_name: {...}}`; mutating it does not affect the "
                "module-level table."
            ),
        },
    },
}

SECTION_HEADERS = {
    "parameters", "inputs", "returns", "outputs",
    "references", "examples", "example", "notes", "raises",
}


def signature_of(node: ast.FunctionDef) -> str:
    a = node.args
    parts = []
    pos = a.posonlyargs + a.args
    defaults = list(a.defaults)
    n = len(pos)
    nd = len(defaults)
    for i, arg in enumerate(pos):
        j = i - (n - nd)
        d = f"={ast.unparse(defaults[j])}" if j >= 0 else ""
        parts.append(f"{arg.arg}{d}")
    if a.vararg:
        parts.append(f"*{a.vararg.arg}")
    for i, arg in enumerate(a.kwonlyargs):
        d = f"={ast.unparse(a.kw_defaults[i])}" if a.kw_defaults[i] is not None else ""
        parts.append(f"{arg.arg}{d}")
    if a.kwarg:
        parts.append(f"**{a.kwarg.arg}")
    return ", ".join(parts)


def split_sections(doc: str):
    """Return (summary, {section: [paragraphs]}) from a numpydoc-ish docstring."""
    lines = doc.splitlines()
    summary_lines = []
    i = 0
    # summary = first paragraph (until blank line that precedes a section header)
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.fullmatch(r"-{3,}\s*", nxt) and line.strip().lower() in SECTION_HEADERS:
            break
        if not line.strip() and any(
            re.fullmatch(r"-{3,}\s*", lines[j + 1] or "") and lines[j].strip().lower() in SECTION_HEADERS
            for j in range(i + 1, min(i + 3, len(lines) - 1))
        ):
            break
        summary_lines.append(line)
        i += 1
    summary = "\n".join(summary_lines).strip()

    sections = {}
    cur = None
    buf = []
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.fullmatch(r"-{3,}\s*", nxt) and line.strip().lower() in SECTION_HEADERS:
            if cur:
                sections[cur] = buf
            cur = line.strip().lower()
            buf = []
            i += 2
            continue
        buf.append(line)
        i += 1
    if cur:
        sections[cur] = buf
    return summary, sections


def parse_params(lines):
    """Parse 'name : type' + indented description blocks."""
    entries = []
    cur = None
    for line in lines:
        m = re.match(r"^(\*{0,2}[\w\[\],\. ]+?)\s*:\s*(.+)$", line)
        if m and not line.startswith(" "):
            cur = {"name": m.group(1).strip(), "type": m.group(2).strip(), "desc": []}
            entries.append(cur)
        elif cur is not None:
            if line.strip():
                cur["desc"].append(line.strip())
    for e in entries:
        e["desc"] = " ".join(e["desc"]).strip()
    return entries


def parse_named_outputs(lines):
    """Parse '[EghX, EbnX, EdhX]' + indented 'name : description' blocks."""
    shape = ""
    entries = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^\[([\w\s,]+)\]$", s)
        if m and not entries:
            shape = s
            continue
        m = re.match(r"^(\*{0,2}\w+)\s*:\s*(.+)$", s)
        if m:
            entries.append({"name": m.group(1).strip(), "type": "", "desc": m.group(2).strip()})
    return shape, entries


def render_entry(name: str, node: ast.FunctionDef, doc: str, supp: dict | None = None) -> str:
    supp = supp or {}
    summary, sections = split_sections(doc)
    if not summary and supp.get("summary"):
        summary = supp["summary"]
    out = [f"### `{name}({signature_of(node)})`", ""]
    if summary:
        out += [summary.replace("\n", " "), ""]

    has_params = False
    for key, label in (("parameters", "Parameters"), ("inputs", "Inputs")):
        if key in sections:
            params = parse_params(sections[key])
            if params:
                has_params = True
                out.append(f"**{label}**")
                out.append("")
                for p in params:
                    t = f" (*{p['type']}*)" if p["type"] and p["type"].lower() != "optional" else (" (*optional*)" if p["type"].lower() == "optional" else "")
                    d = f" — {p['desc']}" if p["desc"] else ""
                    out.append(f"- `{p['name']}`{t}{d}")
                out.append("")
    if not has_params and "parameters" in supp:
        out.append("**Parameters**")
        out.append("")
        for name_, type_, desc in supp["parameters"]:
            out.append(f"- `{name_}` (*{type_}*) — {desc}")
        out.append("")

    for key, label in (("returns", "Returns"), ("outputs", "Outputs")):
        if key in sections:
            body = "\n".join(sections[key]).strip()
            if body:
                out.append(f"**{label}**")
                out.append("")
                params = parse_params(sections[key])
                shape = ""
                if not params:
                    shape, params = parse_named_outputs(sections[key])
                if params:
                    if shape:
                        out.append(f"Returns {shape}:")
                        out.append("")
                    for p in params:
                        t = f" (*{p['type']}*)" if p["type"] else ""
                        d = f" — {p['desc']}" if p["desc"] else ""
                        out.append(f"- `{p['name']}`{t}{d}")
                else:
                    out.append(body.replace("\n", " "))
                out.append("")

    if not ({"returns", "outputs"} & sections.keys()) and "returns" in supp:
        out.append("**Returns**")
        out.append("")
        out.append(supp["returns"])
        out.append("")

    if "references" in sections:
        body = "\n".join(sections["references"]).strip()
        if body:
            out.append("**References**")
            out.append("")
            out.append(body)
            out.append("")

    if "notes" in sections:
        body = "\n".join(sections["notes"]).strip()
        if body:
            out.append("**Notes**")
            out.append("")
            out.append(body)
            out.append("")
    elif "notes" in supp:
        out.append("**Notes**")
        out.append("")
        out.append(supp["notes"])
        out.append("")

    for key in ("example", "examples"):
        if key in sections:
            body = "\n".join(sections[key]).strip()
            if body:
                out.append("**Example**")
                out.append("")
                out.append("```python")
                out.extend(body.splitlines())
                out.append("```")
                out.append("")
    return "\n".join(out)


def collect_functions(path: Path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node) or ""
            yield node, doc


def load_all(pkg: str):
    """Return the public name set for a package, or None to keep every
    non-underscore function (tool.py has no __all__)."""
    base = SRC / pkg
    if not base.is_dir():
        return None
    init = base / "__init__.py"
    tree = ast.parse(init.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "__all__":
                    return set(ast.literal_eval(node.value)) | EXTRA_PUBLIC.get(pkg, set())
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "__all__":
            return set(ast.literal_eval(node.value)) | EXTRA_PUBLIC.get(pkg, set())
    return EXTRA_PUBLIC.get(pkg, set()) or None


def module_files(pkg: str):
    base = SRC / pkg
    if base.is_dir():
        for f in sorted(base.glob("*.py")):
            if f.name != "__init__.py":
                yield f
    else:
        yield SRC / f"{pkg}.py"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pkg, fname in MODULES.items():
        allowed = load_all(pkg)
        seen = {}
        for path in module_files(pkg):
            for node, doc in collect_functions(path):
                if allowed is not None and node.name not in allowed:
                    continue
                if node.name in seen:
                    continue
                supp = SUPPLEMENTS.get(pkg, {}).get(node.name)
                seen[node.name] = render_entry(node.name, node, doc, supp)
        missing = sorted(allowed - seen.keys()) if allowed is not None else []
        if missing:
            print(f"WARNING {pkg}: not found in sources: {missing}")
        entries = sorted(seen.items(), key=lambda x: x[0].lower())
        page = [
            f"# `irradpy2.{pkg}` API reference",
            "",
            "One entry per public function. Function names are listed in the module",
            "overview; parameter semantics follow the source docstrings.",
            "",
            "Reference lists are reproduced verbatim from the source docstrings.",
            "They attribute the published formulation each model implements;",
            "they are implementation provenance, not verified bibliographic records.",
            "",
        ]
        names = ", ".join(f"`{n}`" for n, _ in entries)
        page += [f"Functions ({len(entries)}): {names}", "", "---", ""]
        page += [e for _, e in entries]
        (OUT / fname).write_text("\n".join(page) + "\n")
        print(f"{fname}: {len(entries)} entries")


if __name__ == "__main__":
    main()
