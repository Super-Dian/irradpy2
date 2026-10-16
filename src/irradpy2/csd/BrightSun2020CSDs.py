import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.optimize import fmin


def datetime_convert(datetime: np.ndarray):
    """Convert a datetime sequence to (day-of-year, year) arrays.

    Parameters
    ----------
    datetime : np.ndarray
        Datetime sequence parsed with ``pandas.to_datetime``; invalid
        entries become NaT. The original array shape is preserved.

    Returns
    -------
    Dayth : numpy.ndarray of float
        Day of year (1-366), same shape as ``datetime``.
    Year : numpy.ndarray of float
        Calendar year, same shape as ``datetime``.
    """
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def BrightSun2020CSDs(zen: np.ndarray, ghi: np.ndarray, ghics: np.ndarray, dhi: np.ndarray, difcs: np.ndarray, datetime: np.ndarray, plot_figure=None):
    """
    Bright-Sun 2020 clear-sky detection methodology for cloudless sky CSDs.

    References
    ----------
    Alia-Martinez, M., Antonanzas, J., Urraca, R., Martinez-De-Pison, F. J., and Antonanzas-Torres, F.
    (2016). Benchmark of algorithms for solar clear-sky detection. Journal of Renewable and Sustainable Energy, 8(3).

    Ellis, 2018, PVLIB detect_ghics at commit f4e4ad3bbe335fa49bf300ce53784d69d719ca98.

    Reno, M. J., and Hansen, C. W. (2016). Identification of periods of clear sky irradiance in time series of GHI measurements.
    Renewable Energy, 90, 520-531.

    Shen, Y., Wei, H., Zhu, T., Zhao, X., and Zhang, K. (2018). A Data-driven Clear Sky Model for Direct Normal Irradiance.
    IOP Conference Series: Journal of Physics: Conference Series, 1072.

    Inman, R. H., Edson, J. G., and Coimbra, C. F. M. (2015). Impact of local broadband turbidity estimation on forecasting of
    clear sky direct normal irradiance. Solar Energy, 117, 125-138.

    Gueymard, C. A., Bright, J. M., Lingfors, D., Habte, A., and Sengupta, M. (2019). A posteriori clear-sky identification
    methods in solar irradiance time series: Review and preliminary validation using sky imagers.
    Renewable and Sustainable Energy Reviews.

    Larraneta, M., Reno, M. J., Lillo-Bravo, I., and Silva-Perez, M. A. (2017). Identifying periods of clear sky direct normal
    irradiance. Renewable Energy, 113, 756-763.

    Quesada-Ruiz, S., Linares-Rodriguez, A., Ruiz-Arias, J. A., and Pozo-Vazquez, D. (2015). An advanced ANN-based method to
    estimate hourly solar radiation from multi-spectral MSG imagery. Solar Energy, 115, 494-504.

    Zhang, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018). Modeling and Simulation of High Frequency Solar Irradiance.

    Zhang, W., Kleiber, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018). A stochastic downscaling approach for generating
    high-frequency solar irradiance scenarios. Solar Energy, 176, 370-379.

    Inputs
    ------
    zen : np.ndarray
        Zenith angle [deg], column vector.
    ghi : np.ndarray
        Global horizontal irradiance [W m-2], column vector.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance [W m-2], column vector.
    dhi : np.ndarray
        Diffuse horizontal irradiance [W m-2], column vector.
    difcs : np.ndarray
        Clear-sky diffuse horizontal irradiance [W m-2], column vector.
    datetime : np.ndarray
        Datetime input.
    plot_figure : optional
        If provided, plot the intermediate and final clear-sky detection results.

    Outputs
    -------
    csd : np.ndarray
        Final clear-sky detection flag for cloudless sky.
        1 = clouds suspected
        0 = clear
    csd_tricomponent : np.ndarray
        Tri-component clear-sky detection before duration filters.
    ghics_optimised : np.ndarray
        Optimised clear-sky GHI curve.
    dnics_optimised : np.ndarray
        Optimised clear-sky DNI curve.
    difcs_optimised : np.ndarray
        Optimised clear-sky DIF curve.

    Notes
    -----
    This function is a strict Python translation of the MATLAB version.
    The following parts are intentionally preserved:
    - parameter names
    - parameter order
    - output order
    - threshold values
    - decision order
    - optimisation logic
    - duration filtering logic
    """

    def _to_column_vector(arr: np.ndarray, var_name: str) -> np.ndarray:
        """Convert input to a column vector following MATLAB-style checks."""
        arr = np.array(arr, dtype=float, copy=True)

        if arr.ndim == 1:
            return arr.reshape(-1, 1)

        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr
            if arr.shape[0] == 1:
                return arr.T
            raise ValueError(f"var {var_name} must be a column vector")

        raise ValueError(f"var {var_name} must be a column vector")

    def _hankel_window(x: np.ndarray, window_length: int) -> np.ndarray:
        """Build MATLAB-style hankel(x, [x(end), NaN(1, window_length-1)])."""
        x = np.asarray(x, dtype=float).reshape(-1)
        return hankel(x, np.concatenate(([x[-1]], np.full(window_length - 1, np.nan))))

    def _nanmean_axis1(x: np.ndarray) -> np.ndarray:
        """Row-wise nanmean."""
        return np.nanmean(x, axis=1).reshape(-1, 1)

    def _nanmax_axis1(x: np.ndarray) -> np.ndarray:
        """Row-wise nanmax."""
        return np.nanmax(x, axis=1).reshape(-1, 1)

    def _nansum_axis1(x: np.ndarray) -> np.ndarray:
        """Row-wise nansum."""
        return np.nansum(x, axis=1).reshape(-1, 1)

    def _nearest_index_from_sorted_grid(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Approximate MATLAB knnsearch for a sorted 1D grid."""
        grid = np.asarray(grid, dtype=float).reshape(-1)
        values = np.asarray(values, dtype=float).reshape(-1)

        idx = np.searchsorted(grid, values)
        idx = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1]
        right = grid[idx]
        choose_right = np.abs(values - right) < np.abs(values - left)
        out = idx - 1
        out[choose_right] = idx[choose_right]
        return out

    def _interp_linear(x: np.ndarray, y: np.ndarray, xx: np.ndarray) -> np.ndarray:
        """MATLAB-like linear interpolation with NaN outside range."""
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        xx = np.asarray(xx, dtype=float).reshape(-1)
        yy = np.interp(xx, x, y, left=np.nan, right=np.nan)
        return yy.reshape(-1, 1)

    def _compute_reno_stats(meas: np.ndarray, clear: np.ndarray, window_length: int):
        """Compute Reno-style statistics."""
        meas_window = _hankel_window(meas, window_length)
        clear_window = _hankel_window(clear, window_length)

        meas_mean = np.nanmean(meas_window, axis=1).reshape(-1, 1)
        meas_max = np.nanmax(meas_window, axis=1).reshape(-1, 1)
        meas_diff = np.vstack([np.diff(meas_window, axis=0), np.full((1, window_length), np.nan)])
        meas_slope = np.vstack([np.diff(meas_window, axis=0), np.full((1, window_length), np.nan)])
        with np.errstate(invalid="ignore", divide="ignore"):
            meas_slope_nstd = (np.nanstd(meas_slope, axis=1, ddof=1) / meas_mean.reshape(-1)).reshape(-1, 1)
        meas_line_length = np.nansum(np.sqrt(meas_diff * meas_diff + 1.0 ** 2), axis=1).reshape(-1, 1)

        clear_mean = np.nanmean(clear_window, axis=1).reshape(-1, 1)
        clear_max = np.nanmax(clear_window, axis=1).reshape(-1, 1)
        clear_diff = np.vstack([np.diff(clear_window, axis=0), np.full((1, window_length), np.nan)])
        clear_slope = np.vstack([np.diff(clear_window, axis=0), np.full((1, window_length), np.nan)])
        clear_line_length = np.nansum(np.sqrt(1.0 ** 2 + clear_diff ** 2), axis=1).reshape(-1, 1)

        return {
            "meas_window": meas_window,
            "clear_window": clear_window,
            "meas_mean": meas_mean,
            "meas_max": meas_max,
            "meas_diff": meas_diff,
            "meas_slope": meas_slope,
            "meas_slope_nstd": meas_slope_nstd,
            "meas_line_length": meas_line_length,
            "clear_mean": clear_mean,
            "clear_max": clear_max,
            "clear_diff": clear_diff,
            "clear_slope": clear_slope,
            "clear_line_length": clear_line_length,
        }

    def _scale_limit_with_zenith(zen_values: np.ndarray, big_val: float, mid_val: float, small_val: float, zenith_turn_around: float, z_start: float = 20.0, z_end: float = 90.0, keep_constant_until_turnaround: bool = False):
        """Build zenith-dependent limit matching MATLAB interpolation logic."""
        z = np.round(np.arange(z_start, z_end + 0.0001, 0.01), 2)
        turn_idx = int(np.argmin(np.abs(z - zenith_turn_around)))

        if keep_constant_until_turnaround:
            first_part = np.linspace(big_val, big_val, turn_idx + 1)
        else:
            first_part = np.linspace(big_val, mid_val, turn_idx + 1)

        second_part = np.linspace(mid_val if not keep_constant_until_turnaround else big_val, small_val, len(z) - (turn_idx + 1) + 1)
        second_part = second_part[1:]
        curve = np.flip(np.concatenate([first_part, second_part]))
        inds = _nearest_index_from_sorted_grid(z, zen_values.reshape(-1))
        return curve[inds].reshape(-1, 1)

    # ---------------------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------------------
    zen = _to_column_vector(np.asarray(zen, dtype=float), "zen")
    ghi = _to_column_vector(np.asarray(ghi, dtype=float), "ghi")
    ghics = _to_column_vector(np.asarray(ghics, dtype=float), "ghics")
    dhi = _to_column_vector(np.asarray(dhi, dtype=float), "dhi")
    difcs = _to_column_vector(np.asarray(difcs, dtype=float), "difcs")

    Dayth, Year = datetime_convert(datetime)
    datetime = np.asarray(datetime).reshape(-1)
    Dayth = Dayth.reshape(-1)
    Year = Year.reshape(-1)

    if len(set([len(ghi), len(ghics), len(zen), len(dhi), len(difcs), len(datetime)])) != 1:
        raise ValueError("vars must be equal in length")

    # ---------------------------------------------------------------------
    # Calculate DNI
    # ---------------------------------------------------------------------
    dhi[dhi > ghi] = ghi[dhi > ghi]
    difcs[difcs > ghics] = ghics[difcs > ghics]

    with np.errstate(divide="ignore", invalid="ignore"):
        dnics = (ghics - difcs) / np.cos(np.deg2rad(zen))
        dni = (ghi - dhi) / np.cos(np.deg2rad(zen))

    # ---------------------------------------------------------------------
    # Clean NaN values
    # ---------------------------------------------------------------------
    idxs_nan = (np.isnan(ghi) | np.isnan(dni) | np.isnan(dhi)).reshape(-1)
    not_nan_indxs = (~np.isnan(ghi) & ~np.isnan(dni) & ~np.isnan(dhi)).reshape(-1)
    n_output = len(ghi)

    ghi = ghi[~idxs_nan]
    dni = dni[~idxs_nan]
    dhi = dhi[~idxs_nan]
    ghics = ghics[~idxs_nan]
    dnics = dnics[~idxs_nan]
    difcs = difcs[~idxs_nan]
    zen = zen[~idxs_nan]
    datetime = datetime[~idxs_nan]
    Dayth = Dayth[~idxs_nan]
    Year = Year[~idxs_nan]

    # ---------------------------------------------------------------------
    # Parameterisation
    # ---------------------------------------------------------------------
    reno_ghi_mean_diff = 75
    reno_ghi_max_diff = 75
    reno_ghi_lower_line_length = -5
    reno_ghi_upper_line_length = 10
    reno_ghi_var_diff = 0.1
    reno_ghi_slope_dev = 8

    opt_thres = 30

    BrightSunCriteria = {}
    BrightSunCriteria["ghi_c1_big_zenith"] = 0.5
    BrightSunCriteria["ghi_c1_mean_diff"] = 0.125
    BrightSunCriteria["ghi_c1_small_zenith"] = 0.25

    BrightSunCriteria["ghi_c2_big_zenith"] = 0.5
    BrightSunCriteria["ghi_c2_max_diff"] = 0.125
    BrightSunCriteria["ghi_c2_small_zenith"] = 0.25

    BrightSunCriteria["ghi_c3_lower_line_length"] = -0.5
    BrightSunCriteria["ghi_c3_small_zenith"] = -7

    BrightSunCriteria["ghi_c4_var_diff"] = 0.4

    BrightSunCriteria["ghi_c5_slope_dev"] = 15
    BrightSunCriteria["ghi_c5_small_zenith"] = BrightSunCriteria["ghi_c5_slope_dev"] * 3

    BrightSunCriteria["dif_c1_big_zenith"] = 0.5
    BrightSunCriteria["dif_c1_mean_diff"] = 0.5
    BrightSunCriteria["dif_c1_small_zenith"] = 0.25

    BrightSunCriteria["dif_c2_big_zenith"] = 0.5
    BrightSunCriteria["dif_c2_max_diff"] = 0.5
    BrightSunCriteria["dif_c2_small_zenith"] = 0.25

    BrightSunCriteria["dif_c3_lower_line_length"] = -1.7
    BrightSunCriteria["dif_c3_small_zenith"] = -6

    BrightSunCriteria["dif_c4_var_diff"] = 0.2

    BrightSunCriteria["dif_c5_slope_dev"] = 8
    BrightSunCriteria["dif_c5_small_zenith"] = BrightSunCriteria["dif_c5_slope_dev"] * 3

    zenith_turn_around = 30
    window_length = 10
    max_iterations = 20

    upper_alpha_limit = 1.5
    lower_alpha_limit = 0.7
    lower_alpha_limit_DIF = 0.4

    window_1st_filter = 1
    tolerance_1st_filter = 1

    window_2nd_filter = 1
    tolerance_2nd_filter = 1

    # ---------------------------------------------------------------------
    # Initial CSD guess using Reno for optimisation only
    # ---------------------------------------------------------------------
    stats = _compute_reno_stats(ghi.reshape(-1), ghics.reshape(-1), window_length)

    c1 = np.ones_like(ghi)
    c2 = np.ones_like(ghi)
    c3 = np.ones_like(ghi)
    c4 = np.ones_like(ghi)
    c5 = np.ones_like(ghi)
    c6 = np.ones_like(ghi)

    c1[np.abs(stats["meas_mean"] - stats["clear_mean"]) < reno_ghi_mean_diff] = 0
    c2[np.abs(stats["meas_max"] - stats["clear_max"]) < reno_ghi_max_diff] = 0
    c3[(stats["meas_line_length"] - stats["clear_line_length"] > reno_ghi_lower_line_length) & (stats["meas_line_length"] - stats["clear_line_length"] < reno_ghi_upper_line_length)] = 0
    c4[stats["meas_slope_nstd"] < reno_ghi_var_diff] = 0
    c5[np.nanmax(np.abs(stats["meas_slope"] - stats["clear_slope"]), axis=1).reshape(-1, 1) < reno_ghi_slope_dev] = 0
    c6[(stats["clear_mean"] != 0) & ~np.isnan(stats["clear_mean"])] = 0

    csd_initial_sum = c1 + c2 + c3 + c4 + c5 + c6
    csd_ghiz = np.zeros_like(ghi)
    csd_ghiz[csd_initial_sum > 0] = 1
    csd_initial = csd_ghiz.copy()

    if plot_figure is not None:
        datetime_plot = datetime
        plt.figure(figsize=(12, 5))
        plt.plot(datetime_plot, ghi.reshape(-1), label="GHI")
        plt.plot(datetime_plot, dni.reshape(-1), label="DNI")
        plt.plot(datetime_plot, dhi.reshape(-1), label="DIF")
        ghiCSD = ghi.astype(float).copy()
        ghiCSD[csd_initial == 1] = np.nan
        plt.plot(datetime_plot, ghiCSD.reshape(-1), linewidth=2, color="k", label="CSD")
        plt.plot(datetime_plot, ghics.reshape(-1), "k:", label="Clear-sky irradiance")
        dniCSD = dni.astype(float).copy()
        difCSD = dhi.astype(float).copy()
        dniCSD[csd_initial == 1] = np.nan
        difCSD[csd_initial == 1] = np.nan
        plt.plot(datetime_plot, dniCSD.reshape(-1), linewidth=2, color="k")
        plt.plot(datetime_plot, difCSD.reshape(-1), linewidth=2, color="k")
        plt.plot(datetime_plot, dnics.reshape(-1), "k:")
        plt.plot(datetime_plot, difcs.reshape(-1), "k:")
        plt.ylim([1, 1300])
        plt.legend()
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.title("Initial CSD from Reno 2016 example")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Find unique days
    # ---------------------------------------------------------------------
    dtn = Year * 1000 + Dayth
    unique_days = np.unique(dtn)

    alpha_ghi = np.ones(len(unique_days), dtype=float)
    alpha_dif = np.ones(len(unique_days), dtype=float)
    alpha_dni = np.ones(len(unique_days), dtype=float)

    if plot_figure is not None:
        old_ghics = ghics.copy()
        old_dnics = dnics.copy()
        old_difcs = difcs.copy()

    # ---------------------------------------------------------------------
    # Optimisation of clear-sky irradiance
    # ---------------------------------------------------------------------
    for d in range(len(unique_days)):
        idxs = np.where(dtn == unique_days[d])[0]

        ghid = ghi[idxs]
        ghicsd = ghics[idxs]
        difd = dhi[idxs]
        difcsd = difcs[idxs]
        dnid = dni[idxs]
        dnicsd = dnics[idxs]
        csdd = csd_initial[idxs].copy()

        csdd[(difd < opt_thres) | (ghid < opt_thres)] = 1

        test_ghi = ghid[csdd == 0].reshape(-1)
        test_ghics = ghicsd[csdd == 0].reshape(-1)
        test_dif = difd[csdd == 0].reshape(-1)
        test_difcs = difcsd[csdd == 0].reshape(-1)
        test_dni = dnid[csdd == 0].reshape(-1)
        test_dnics = dnicsd[csdd == 0].reshape(-1)

        rmse_ghi = lambda a: np.sqrt(np.nanmean((test_ghi - a * test_ghics) ** 2))
        rmse_dif = lambda a: np.sqrt(np.nanmean((test_dif - a * test_difcs) ** 2))
        rmse_dni = lambda a: np.sqrt(np.nanmean((test_dni - a * test_dnics) ** 2))

        if np.sum(csdd == 0) > 60:
            current_alpha = float(alpha_ghi[d])
            iter_count = 0
            while iter_count < max_iterations:
                previous_alpha = current_alpha
                current_alpha = float(fmin(rmse_ghi, current_alpha, disp=False)[0])
                iter_count += 1
                if round(current_alpha * 10000) == round(previous_alpha * 10000):
                    break
            alpha_ghi[d] = current_alpha

            current_alpha = float(alpha_dif[d])
            iter_count = 0
            while iter_count < max_iterations:
                previous_alpha = current_alpha
                current_alpha = float(fmin(rmse_dif, current_alpha, disp=False)[0])
                iter_count += 1
                if round(current_alpha * 10000) == round(previous_alpha * 10000):
                    break
            alpha_dif[d] = current_alpha

            current_alpha = float(alpha_dni[d])
            iter_count = 0
            while iter_count < max_iterations:
                previous_alpha = current_alpha
                current_alpha = float(fmin(rmse_dni, current_alpha, disp=False)[0])
                iter_count += 1
                if round(current_alpha * 10000) == round(previous_alpha * 10000):
                    break
            alpha_dni[d] = current_alpha

        temp = alpha_ghi[d]
        temp = min(temp, upper_alpha_limit)
        temp = max(temp, lower_alpha_limit)
        alpha_ghi[d] = temp

        temp = alpha_dni[d]
        temp = min(temp, upper_alpha_limit)
        temp = max(temp, lower_alpha_limit)
        alpha_dni[d] = temp

        temp = alpha_dif[d]
        temp = min(temp, upper_alpha_limit)
        temp = max(temp, lower_alpha_limit_DIF)
        alpha_dif[d] = temp

        ghics[idxs] = ghics[idxs] * alpha_ghi[d]
        difcs[idxs] = difcs[idxs] * alpha_dif[d]
        dnics[idxs] = dnics[idxs] * alpha_dni[d]

    if plot_figure is not None:
        datetime_plot = datetime
        plt.figure(figsize=(12, 5))
        plt.plot(datetime_plot, ghi.reshape(-1), label="GHI")
        plt.plot(datetime_plot, dni.reshape(-1), label="DNI")
        plt.plot(datetime_plot, dhi.reshape(-1), label="DIF")
        plt.plot(datetime_plot, ghiCSD.reshape(-1), linewidth=2, color="k", label="Reno CSD")
        plt.plot(datetime_plot, old_ghics.reshape(-1), "b:", linewidth=2, label="Clear-sky original")
        plt.plot(datetime_plot, ghics.reshape(-1), "r:", linewidth=2, label="Clear-sky optimised")
        plt.plot(datetime_plot, dniCSD.reshape(-1), linewidth=2, color="k")
        plt.plot(datetime_plot, difCSD.reshape(-1), linewidth=2, color="k")
        plt.plot(datetime_plot, old_dnics.reshape(-1), "b:", linewidth=2)
        plt.plot(datetime_plot, old_difcs.reshape(-1), "b:", linewidth=2)
        plt.plot(datetime_plot, dnics.reshape(-1), "r:", linewidth=2)
        plt.plot(datetime_plot, difcs.reshape(-1), "r:", linewidth=2)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.title("Example of clear-sky optimisation")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Tri-component analysis: GHI
    # ---------------------------------------------------------------------
    stats = _compute_reno_stats(ghi.reshape(-1), ghics.reshape(-1), window_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        line_diff = (stats["meas_line_length"] - stats["clear_line_length"]) / stats["clear_line_length"]

    c1 = np.ones_like(ghi)
    c2 = np.ones_like(ghi)
    c3 = np.ones_like(ghi)
    c4 = np.ones_like(ghi)
    c5 = np.ones_like(ghi)
    c6 = np.ones_like(ghi)

    c1_lim = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["ghi_c1_big_zenith"], BrightSunCriteria["ghi_c1_mean_diff"], BrightSunCriteria["ghi_c1_small_zenith"], zenith_turn_around)
    c2_lim = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["ghi_c2_big_zenith"], BrightSunCriteria["ghi_c2_max_diff"], BrightSunCriteria["ghi_c2_small_zenith"], zenith_turn_around)
    c3_lim_lower = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["ghi_c3_lower_line_length"], BrightSunCriteria["ghi_c3_lower_line_length"], BrightSunCriteria["ghi_c3_small_zenith"], zenith_turn_around, keep_constant_until_turnaround=True)
    c3_lim_upper = np.abs(c3_lim_lower)
    c5_lim_lower = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["ghi_c5_slope_dev"], BrightSunCriteria["ghi_c5_slope_dev"], BrightSunCriteria["ghi_c5_small_zenith"], zenith_turn_around, keep_constant_until_turnaround=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        c1[np.abs(stats["meas_mean"] - stats["clear_mean"]) / stats["clear_mean"] < c1_lim] = 0
        c2[np.abs(stats["meas_max"] - stats["clear_max"]) / stats["clear_max"] < c2_lim] = 0
    c3[(line_diff > c3_lim_lower) & (line_diff < c3_lim_upper)] = 0
    c4[stats["meas_slope_nstd"] < BrightSunCriteria["ghi_c4_var_diff"]] = 0
    c5[np.nanmax(np.abs(stats["meas_slope"] - stats["clear_slope"]), axis=1).reshape(-1, 1) < c5_lim_lower] = 0
    c6[(stats["clear_mean"] != 0) & ~np.isnan(stats["clear_mean"])] = 0

    csd_1 = c1 + c2 + c3 + c4 + c5 + c6
    csd_ghi = np.zeros_like(ghi)
    csd_ghi[csd_1 > 0] = 1
    criteria_ghi = np.hstack([c1, c2, c3, c4, c5, c6])

    # ---------------------------------------------------------------------
    # Tri-component analysis: DIF
    # ---------------------------------------------------------------------
    stats = _compute_reno_stats(dhi.reshape(-1), difcs.reshape(-1), window_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        line_diff = (stats["meas_line_length"] - stats["clear_line_length"]) / stats["clear_line_length"]

    c1 = np.ones_like(dhi)
    c2 = np.ones_like(dhi)
    c3 = np.ones_like(dhi)
    c4 = np.ones_like(dhi)
    c5 = np.ones_like(dhi)
    c6 = np.ones_like(dhi)

    c1_lim = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["dif_c1_big_zenith"], BrightSunCriteria["dif_c1_mean_diff"], BrightSunCriteria["dif_c1_small_zenith"], zenith_turn_around)
    c2_lim = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["dif_c2_big_zenith"], BrightSunCriteria["dif_c2_max_diff"], BrightSunCriteria["dif_c2_small_zenith"], zenith_turn_around)
    c3_lim_lower = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["dif_c3_lower_line_length"], BrightSunCriteria["dif_c3_lower_line_length"], BrightSunCriteria["dif_c3_small_zenith"], zenith_turn_around, keep_constant_until_turnaround=True)
    c3_lim_upper = np.abs(c3_lim_lower)
    c5_lim_lower = _scale_limit_with_zenith(zen.reshape(-1), BrightSunCriteria["dif_c5_slope_dev"], BrightSunCriteria["dif_c5_slope_dev"], BrightSunCriteria["dif_c5_small_zenith"], zenith_turn_around, keep_constant_until_turnaround=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        c1[np.abs(stats["meas_mean"] - stats["clear_mean"]) / stats["clear_mean"] < c1_lim] = 0
        c2[np.abs(stats["meas_max"] - stats["clear_max"]) / stats["clear_max"] < c2_lim] = 0
    c3[(line_diff > c3_lim_lower) & (line_diff < c3_lim_upper)] = 0
    c4[stats["meas_slope_nstd"] < BrightSunCriteria["dif_c4_var_diff"]] = 0
    c5[np.nanmax(np.abs(stats["meas_slope"] - stats["clear_slope"]), axis=1).reshape(-1, 1) < c5_lim_lower] = 0
    c6[(stats["clear_mean"] != 0) & ~np.isnan(stats["clear_mean"])] = 0

    csd_2 = c1 + c2 + c3 + c4 + c5 + c6
    csd_dif = np.zeros_like(ghi)
    csd_dif[csd_2 > 0] = 1
    criteria_dif = np.hstack([c1, c2, c3, c4, c5, c6])

    # ---------------------------------------------------------------------
    # Tri-component analysis: DNI
    # ---------------------------------------------------------------------
    z = np.round(np.arange(30.0, 90.0 + 0.0001, 0.01), 2)
    kc_lims = np.flip(np.linspace(0.5, 0.9, len(z))).reshape(-1, 1)
    inds = _nearest_index_from_sorted_grid(z, zen.reshape(-1))
    kc_lim = kc_lims[inds].reshape(-1, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        kcb = dni / dnics

    csd_dni = np.ones_like(dni)
    csd_dni[kcb > kc_lim] = 0
    criteria_dni = csd_dni.copy()

    # ---------------------------------------------------------------------
    # Overall criteria
    # ---------------------------------------------------------------------
    csd_overall = np.ones_like(ghi)
    csd_overall[csd_ghi + csd_dif + csd_dni == 0] = 0

    if plot_figure is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(ghi.reshape(-1), label="GHI")
        plt.plot(dni.reshape(-1), label="DNI")
        plt.plot(dhi.reshape(-1), label="DIF")
        ghiCSD = ghi.astype(float).copy()
        dniCSD = dni.astype(float).copy()
        difCSD = dhi.astype(float).copy()
        allCSD = ghi.astype(float).copy()
        ghiCSD[csd_ghi == 1] = np.nan
        dniCSD[csd_dni == 1] = np.nan
        difCSD[csd_dif == 1] = np.nan
        allCSD[csd_overall == 1] = np.nan
        plt.plot(ghiCSD.reshape(-1), linewidth=4, color="k")
        plt.plot(ghics.reshape(-1), "k:")
        plt.plot(allCSD.reshape(-1), linewidth=2, color="g")
        plt.plot(dnics.reshape(-1), "k:")
        plt.plot(difcs.reshape(-1), "k:")
        plt.plot(dniCSD.reshape(-1), linewidth=2, color="k")
        plt.plot(difCSD.reshape(-1), linewidth=2, color="k")
        plt.ylim([0, 1300])
        plt.legend(["GHI", "DNI", "DIF", "CSD", "CS", "combined CSD"])
        plt.xlabel("Time")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.title("Demonstrate that the tri-component is working")
        plt.tight_layout()
        plt.show()

    if plot_figure is not None:
        major_line_width = 20
        line_sep = (major_line_width - 3) / 5
        cols = np.array([[199, 233, 180], [127, 205, 187], [65, 182, 196], [44, 127, 184], [37, 52, 148]], dtype=float) / 255.0

        plt.figure(figsize=(12, 4))
        plt.plot(ghi.reshape(-1), label="GHI")
        plt.plot(dni.reshape(-1), label="DNI")
        plt.plot(dhi.reshape(-1), label="DIF")
        plt.plot(ghics.reshape(-1), "k:", label="GHIcs")
        plt.plot(dnics.reshape(-1), "k:", label="DNIcs")
        plt.plot(difcs.reshape(-1), "k:", label="DIFcs")

        for c in range(5):
            cc = ghi.astype(float).copy()
            crit = criteria_ghi[:, [c]]
            cc[crit == 1] = np.nan
            plt.plot(cc.reshape(-1), linewidth=major_line_width - line_sep * (c + 1), color=cols[c, :])

        csdghi = ghi.astype(float).copy()
        csdghi[csd_ghi == 1] = np.nan
        plt.plot(csdghi.reshape(-1), "m")

        csdghi = ghi.astype(float).copy()
        csdghi[csd_overall == 1] = np.nan
        plt.plot(csdghi.reshape(-1), "r", linewidth=2)

        csddni = dni.astype(float).copy()
        csddni[csd_dni == 1] = np.nan
        plt.plot(csddni.reshape(-1), "m")

        csddni = dni.astype(float).copy()
        csddni[csd_overall == 1] = np.nan
        plt.plot(csddni.reshape(-1), "r", linewidth=2)

        for c in range(5):
            cc = dhi.astype(float).copy()
            crit = criteria_dif[:, [c]]
            cc[crit == 1] = np.nan
            plt.plot(cc.reshape(-1), linewidth=major_line_width - line_sep * (c + 1), color=cols[c, :])

        csddif = dhi.astype(float).copy()
        csddif[csd_dif == 1] = np.nan
        plt.plot(csddif.reshape(-1), "m")

        csddif = dhi.astype(float).copy()
        csddif[csd_overall == 1] = np.nan
        plt.plot(csddif.reshape(-1), "r", linewidth=2)

        plt.ylim([0, 1300])
        plt.xlabel("Time")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.title("Explore the nature of Reno criteria")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Duration criteria
    # ---------------------------------------------------------------------
    if window_1st_filter <= 1:
        if tolerance_1st_filter == window_1st_filter:
            tolerance_1st_filter = 2
        window_1st_filter = 2

    if window_2nd_filter <= 1:
        if tolerance_2nd_filter == window_2nd_filter:
            tolerance_2nd_filter = 2
        window_2nd_filter = 2

    csdh_1st_duration = np.nansum(_hankel_window(csd_overall.reshape(-1), window_1st_filter), axis=1).reshape(-1, 1)
    csdh_1st_duration = np.vstack([np.full((window_1st_filter // 2, 1), np.nan), csdh_1st_duration[:len(csdh_1st_duration) - window_1st_filter // 2]])
    csd_1st_duration = np.zeros_like(ghi)
    csd_1st_duration[csdh_1st_duration > tolerance_1st_filter] = 1

    A = np.arange(1, len(zen) + 1).reshape(-1, 1)
    B = np.where(np.round(zen.reshape(-1)) == 85)[0] + 1

    if len(B) == 0:
        distance_to_nearest_sunrise_set = np.full((len(zen), 1), np.inf)
    else:
        distance_to_nearest_sunrise_set = np.min(np.abs(A - B.reshape(1, -1)), axis=1).reshape(-1, 1)

    csd_1st_duration[distance_to_nearest_sunrise_set < window_1st_filter] = 0

    csdh_2nd_duration = np.nansum(_hankel_window(csd_overall.reshape(-1), window_2nd_filter), axis=1).reshape(-1, 1)
    csdh_2nd_duration = np.vstack([np.full((window_2nd_filter // 2, 1), np.nan), csdh_2nd_duration[:len(csdh_2nd_duration) - window_2nd_filter // 2]])
    csd_2nd_duration = np.zeros_like(ghi)
    csd_2nd_duration[csdh_2nd_duration > tolerance_2nd_filter] = 1

    window_3rd_duration = 10
    tolerance_3rd_duration = 2
    csdh_3rd_duration = np.nansum(_hankel_window(csd_overall.reshape(-1), window_3rd_duration), axis=1).reshape(-1, 1)
    csdh_3rd_duration = np.vstack([np.full((window_3rd_duration // 2, 1), np.nan), csdh_3rd_duration[:len(csdh_3rd_duration) - window_3rd_duration // 2]])
    csd_3rd_duration = np.zeros_like(ghi)
    csd_3rd_duration[csdh_3rd_duration > tolerance_3rd_duration] = 1

    csd_2nd_duration[(csd_3rd_duration == 0) & (distance_to_nearest_sunrise_set < window_1st_filter)] = 0

    csd_nan_indexed = np.ones_like(ghi)
    csd_nan_indexed[csd_overall + csd_1st_duration + csd_2nd_duration == 0] = 0

    # ---------------------------------------------------------------------
    # Outputs indexed back to original length
    # ---------------------------------------------------------------------
    csd = np.ones((n_output, 1), dtype=float)
    csd[not_nan_indxs] = csd_nan_indexed

    ghics_optimised = np.full((n_output, 1), np.nan, dtype=float)
    dnics_optimised = np.full((n_output, 1), np.nan, dtype=float)
    difcs_optimised = np.full((n_output, 1), np.nan, dtype=float)

    ghics_optimised[not_nan_indxs] = ghics
    dnics_optimised[not_nan_indxs] = dnics
    difcs_optimised[not_nan_indxs] = difcs

    csd_tricomponent = np.full((n_output, 1), np.nan, dtype=float)
    csd_tricomponent[not_nan_indxs] = csd_overall

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(figsize=(12, 4))
        csdtri = ghi.astype(float).copy()
        csd60w = ghi.astype(float).copy()
        csd10w = ghi.astype(float).copy()
        csdall = ghi.astype(float).copy()

        csdtri[csd_overall == 1] = np.nan
        csd60w[csd_1st_duration == 1] = np.nan
        csd10w[csd_2nd_duration == 1] = np.nan
        csdall[csd_nan_indexed == 1] = np.nan

        plt.plot(ghi.reshape(-1), label="GHI")
        plt.plot(csdtri.reshape(-1), color=np.array([215, 48, 31]) / 255.0, linewidth=10, label="Tri-Component CSD")
        plt.plot(csd60w.reshape(-1), color=np.array([254, 196, 79]) / 255.0, linewidth=7.5, label="Passed 90-min Filter")
        plt.plot(csd10w.reshape(-1), color=np.array([120, 198, 121]) / 255.0, linewidth=5, label="Passed 30-min Filter")
        plt.plot(csdall.reshape(-1), color="k", linewidth=2.5, label="Bright-Sun CSD")
        plt.ylim([0, 1300])
        plt.xlabel("Time")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.legend()
        plt.title("Durational window example")
        plt.tight_layout()
        plt.show()

    if plot_figure is not None:
        colours = np.array([[215, 48, 39], [254, 224, 139], [145, 207, 96], [120, 198, 121]], dtype=float) / 255.0

        plt.figure(figsize=(12, 5))
        plt.plot(ghi.reshape(-1), label="GHI")
        plt.plot(dni.reshape(-1), label="DNI")
        plt.plot(dhi.reshape(-1), label="DIF")

        ghiCSD = ghi.astype(float).copy()
        dniCSD = dni.astype(float).copy()
        difCSD = dhi.astype(float).copy()

        ghiCSD[csd_ghi == 1] = np.nan
        dniCSD[csd_dni == 1] = np.nan
        difCSD[csd_dif == 1] = np.nan

        plt.plot(ghics.reshape(-1), "k:")
        plt.plot(ghiCSD.reshape(-1), linewidth=12.5, color="k")

        ghiCSD2 = ghi.astype(float).copy()
        ghiCSD2[csd_overall == 1] = np.nan
        plt.plot(ghiCSD2.reshape(-1), linewidth=10, color=colours[0, :])

        ghiCSD3 = ghi.astype(float).copy()
        ghiCSD3[csd_1st_duration == 1] = np.nan
        plt.plot(ghiCSD3.reshape(-1), linewidth=7.5, color=colours[1, :])

        ghiCSD4 = ghi.astype(float).copy()
        ghiCSD4[csd_2nd_duration == 1] = np.nan
        plt.plot(ghiCSD4.reshape(-1), linewidth=5, color=colours[2, :])

        ghiCSD5 = ghi.astype(float).copy()
        ghiCSD5[csd_nan_indexed == 1] = np.nan
        plt.plot(ghiCSD5.reshape(-1), linewidth=2, color="m")

        plt.plot(dnics.reshape(-1), "k:")
        plt.plot(difcs.reshape(-1), "k:")
        plt.plot(dniCSD.reshape(-1), linewidth=2, color="k")
        plt.plot(difCSD.reshape(-1), linewidth=2, color="k")

        if len(ghi) > 20000:
            plt.xlim([1, 20000])

        plt.legend(["GHI", "DNI", "DIF", "Clear-sky irrad.", "Tri-component CSD", "Corroborating CSD", "90min window", "30min window CSD", "Final CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.title("Bright-Sun CSD 2019 example")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(ghi.reshape(-1), label="GHI")
        plt.plot(ghics.reshape(-1), "k:", label="GHIcs")
        ghiCSD_final = ghi.astype(float).copy()
        ghiCSD_final[csd.reshape(-1)[not_nan_indxs] == 1] = np.nan
        plt.plot(ghiCSD_final.reshape(-1), linewidth=4, color=colours[2, :], label="CSD")
        plt.legend()
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.title("Bright-Sun final CSD")
        plt.tight_layout()
        plt.show()

    return csd, csd_tricomponent, ghics_optimised, dnics_optimised, difcs_optimised