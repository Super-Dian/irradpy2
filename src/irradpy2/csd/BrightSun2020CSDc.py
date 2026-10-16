import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.optimize import fmin


def datetime_convert(datetime: np.ndarray):
    """Convert a datetime sequence to (datetime, day-of-year, year) arrays.

    Parameters
    ----------
    datetime : np.ndarray
        Datetime sequence parsed with ``pandas.to_datetime``; invalid
        entries become NaT.

    Returns
    -------
    datetime_dt : numpy.ndarray of object
        Parsed datetime values as a flat 1-D object array.
    Dayth : numpy.ndarray of float
        Day of year (1-366), flat 1-D.
    Year : numpy.ndarray of float
        Calendar year, flat 1-D.
    """
    datetime_array = np.asarray(datetime)
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    datetime_dt = np.asarray(time.to_pydatetime(), dtype=object)
    Dayth = np.asarray(time.dayofyear, dtype=float)
    Year = np.asarray(time.year, dtype=float)

    return datetime_dt, Dayth, Year


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Bright-Sun 2020 CSDc
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

### References:
# M. Alia-Martinez, J. Antonanzas, R. Urraca, F. J. Martinez-De-Pison, and F. Antonanzas-Torres, 2016.
# Benchmark of algorithms for solar clear-sky detection.
# Journal of Renewable and Sustainable Energy, 8(3).
#
# Ellis, 2018, PVLIB detect_ghics at commit f4e4ad3bbe335fa49bf300ce53784d69d719ca98.
#
# Reno, M. J. and Hansen, C. W. 2016.
# Identification of periods of clear sky irradiance in time series of GHI measurements.
# Renewable Energy, 90, 520-531.
#
# Shen, Yu; Wei, Haikun; Zhu, Tingting; Zhao, Xin; Zhang, Kanjian. 2018.
# A Data-driven Clear Sky Model for Direct Normal Irradiance.
# IOP Conference Series: Journal of Physics: Conference Series, 1072.
#
# Inman, Rich H; Edson, James G and Coimbra, Carlos F M. 2015.
# Impact of local broadband turbidity estimation on forecasting of clear sky direct normal irradiance.
# Solar Energy, 117, 125-138.
#
# Gueymard, C.A; Bright, J.M.; Lingfors, D.; Habte, A. and Sengupta, M. 2019.
# A posteriori clear-sky identification methods in solar irradiance time series:
# Review and preliminary validation using sky imagers.
# Renewable and Sustainable Energy Reviews.
#
# Larraneta, M; Reno, M J; Lillo-bravo, I; Silva-p, M A. 2017.
# Identifying periods of clear sky direct normal irradiance.
# Renewable Energy, 113, 756-763.
#
# Quesada-Ruiz, Samuel; Linares-Rodriguez, Alvaro; Ruiz-Arias, José A; Pozo-Vázquez, David. 2015.
# An advanced ANN-based method to estimate hourly solar radiation from multi-spectral MSG imagery.
# Solar Energy, 115, 494-504.
#
# Zhang, Wenqi; Florita, Anthony R; Hodge, Bri-Mathias; Mather, Barry. 2018.
# Modeling and Simulation of High Frequency Solar Irradiance.
#
# Zhang, Wenqi; Kleiber, William; Florita, Anthony R; Hodge, Bri-Mathias; Mather, Barry. 2018.
# A stochastic downscaling approach for generating high-frequency solar irradiance scenarios.
# Solar Energy, 176, 370-379.

### Inputs:
# zen         [degrees]   Zenith angle, column vector.
# ghi         [W m-2]     Global horizontal irradiance, column vector.
# ghics       [W m-2]     Clear-sky global horizontal irradiance, column vector.
# dhi         [W m-2]     Diffuse horizontal irradiance, column vector.
# difcs       [W m-2]     Clear-sky diffuse horizontal irradiance, column vector.
# datetime                Datetime input.
# plot_figure             If defined, diagnostic figures are plotted.

### Outputs:
# csd                     Clear-sky detection for cloudless sky.
#                         1 = clouds suspected, 0 = clear.
# csd_tricomponent        Clear-sky detection from the reno tri-component optimisation.
# ghics_optimised         Optimised clear-sky GHI curve.
# dnics_optimised         Optimised clear-sky DNI curve.
# difcs_optimised         Optimised clear-sky DIF curve.

### Notes:
# Assumption: all data is in 1-min resolution.
# This Python implementation is a strict structural refactor of the MATLAB code.
# Parameters, thresholds, criteria order, optimisation order, and duration-filter logic are preserved.
# Closure is not guaranteed after the optimisation, exactly as noted in the original MATLAB code.


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Helper functions
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

def _ensure_column_vector(x: np.ndarray, var_name: str, error_message: str = None):
    """Ensure input is a column vector, following the original MATLAB safety-check order."""
    x = np.array(x, dtype=float, copy=True)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 1:
        x = x.T
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(error_message if error_message is not None else f"var {var_name} must be a column vector")
    return x


def _hankel_window(x: np.ndarray, window_length: int):
    """Replicate MATLAB hankel(x,[x(end),NaN(1,window_length-1)]) behavior."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return hankel(x, np.concatenate(([x[-1]], np.full(window_length - 1, np.nan))))


def _nanmax_axis1(x: np.ndarray):
    """Row-wise nanmax."""
    return np.nanmax(x, axis=1)


def _nanmean_axis1(x: np.ndarray):
    """Row-wise nanmean."""
    return np.nanmean(x, axis=1)


def _nansum_axis1(x: np.ndarray):
    """Row-wise nansum."""
    return np.nansum(x, axis=1)


def _nearest_indices(reference_values: np.ndarray, query_values: np.ndarray):
    """Equivalent to knnsearch for a monotonic 1D reference grid."""
    reference_values = np.asarray(reference_values, dtype=float).reshape(-1)
    query_values = np.asarray(query_values, dtype=float).reshape(-1)
    idx = np.searchsorted(reference_values, query_values)
    idx = np.clip(idx, 1, len(reference_values) - 1)
    left = reference_values[idx - 1]
    right = reference_values[idx]
    choose_left = np.abs(query_values - left) <= np.abs(query_values - right)
    idx = np.where(choose_left, idx - 1, idx)
    return idx


def _build_brightsun_limits(z: np.ndarray, zen: np.ndarray, val_big: float, val_mid: float, val_small: float, zenith_turn_around: float):
    """Build zenith-flexible limits exactly following the original piecewise-linear construction."""
    turn_idx_arr = np.where(np.isclose(z, zenith_turn_around))[0]
    if len(turn_idx_arr) == 0:
        raise ValueError("zenith_turn_around must exist exactly in z grid.")
    turn_idx = int(turn_idx_arr[0])
    upper_part = np.linspace(val_big, val_mid, turn_idx + 1)
    lower_part = np.linspace(val_mid, val_small, len(z) - turn_idx)
    lims = np.flip(np.concatenate((upper_part.reshape(-1, 1), lower_part.reshape(-1, 1)[1:, :]), axis=0).reshape(-1))
    inds = _nearest_indices(z, zen.reshape(-1))
    return lims[inds].reshape(-1, 1)


def _compute_reno_statistics(meas: np.ndarray, clear: np.ndarray, window_length: int):
    """Compute Reno/Ellis/Inman statistics in the same order as the MATLAB code."""
    meas_window = _hankel_window(meas, window_length)
    clear_window = _hankel_window(clear, window_length)

    meas_mean = _nanmean_axis1(meas_window).reshape(-1, 1)
    meas_max = _nanmax_axis1(meas_window).reshape(-1, 1)
    meas_diff = np.vstack((np.diff(meas_window, axis=0), np.full((1, window_length), np.nan)))
    meas_slope = np.vstack((np.diff(meas_window, axis=0), np.full((1, window_length), np.nan)))
    meas_slope_nstd = (np.nanstd(meas_slope, axis=1, ddof=1).reshape(-1, 1)) / meas_mean
    meas_line_length = _nansum_axis1(np.sqrt(meas_diff * meas_diff + 1 ** 2)).reshape(-1, 1)

    clear_mean = _nanmean_axis1(clear_window).reshape(-1, 1)
    clear_max = _nanmax_axis1(clear_window).reshape(-1, 1)
    clear_diff = np.vstack((np.diff(clear_window, axis=0), np.full((1, window_length), np.nan)))
    clear_slope = np.vstack((np.diff(clear_window, axis=0), np.full((1, window_length), np.nan)))
    clear_line_length = _nansum_axis1(np.sqrt(1 ** 2 + clear_diff ** 2)).reshape(-1, 1)

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


def _reno_initial_csd_ghi_only(ghi: np.ndarray, ghics: np.ndarray, window_length: int, reno_ghi_mean_diff: float, reno_ghi_max_diff: float, reno_ghi_lower_line_length: float, reno_ghi_upper_line_length: float, reno_ghi_var_diff: float, reno_ghi_slope_dev: float):
    """Initial Reno GHI-only CSD guess for optimisation only."""
    stats = _compute_reno_statistics(ghi, ghics, window_length)

    line_diff = stats["meas_line_length"] - stats["clear_line_length"]

    c1 = np.ones_like(ghi)
    c2 = np.ones_like(ghi)
    c3 = np.ones_like(ghi)
    c4 = np.ones_like(ghi)
    c5 = np.ones_like(ghi)
    c6 = np.ones_like(ghi)

    c1[np.abs(stats["meas_mean"] - stats["clear_mean"]) < reno_ghi_mean_diff] = 0
    c2[np.abs(stats["meas_max"] - stats["clear_max"]) < reno_ghi_max_diff] = 0
    c3[(line_diff > reno_ghi_lower_line_length) & (line_diff < reno_ghi_upper_line_length)] = 0
    c4[stats["meas_slope_nstd"] < reno_ghi_var_diff] = 0
    c5[np.nanmax(np.abs(stats["meas_slope"] - stats["clear_slope"]), axis=1).reshape(-1, 1) < reno_ghi_slope_dev] = 0
    c6[(stats["clear_mean"] != 0) & ~np.isnan(stats["clear_mean"])] = 0

    csd_initial = c1 + c2 + c3 + c4 + c5 + c6
    csd_ghiz = np.zeros_like(ghi)
    csd_ghiz[csd_initial > 0] = 1

    return csd_ghiz


def _brightsun_component_csd(meas: np.ndarray, clear: np.ndarray, zen: np.ndarray, window_length: int, c1_big: float, c1_mid: float, c1_small: float, c2_big: float, c2_mid: float, c2_small: float, c3_lower: float, c3_small: float, c4_var: float, c5_mid: float, c5_small: float, zenith_turn_around: float):
    """Tri-component Bright-Sun Reno-style detection for GHI or DIF."""
    stats = _compute_reno_statistics(meas, clear, window_length)
    line_diff = (stats["meas_line_length"] - stats["clear_line_length"]) / stats["clear_line_length"]

    c1 = np.ones_like(meas)
    c2 = np.ones_like(meas)
    c3 = np.ones_like(meas)
    c4 = np.ones_like(meas)
    c5 = np.ones_like(meas)
    c6 = np.ones_like(meas)

    z = np.arange(20, 90.0001, 0.01).reshape(-1, 1)
    zen_flat = zen.reshape(-1, 1)

    c1_lim = _build_brightsun_limits(z.reshape(-1), zen_flat, c1_big, c1_mid, c1_small, zenith_turn_around)
    c2_lim = _build_brightsun_limits(z.reshape(-1), zen_flat, c2_big, c2_mid, c2_small, zenith_turn_around)
    c3_lim_lower = _build_brightsun_limits(z.reshape(-1), zen_flat, c3_lower, c3_lower, c3_small, zenith_turn_around)
    c3_lim_upper = np.abs(c3_lim_lower)
    c5_lim_lower = _build_brightsun_limits(z.reshape(-1), zen_flat, c5_mid, c5_mid, c5_small, zenith_turn_around)

    c1[np.abs(stats["meas_mean"] - stats["clear_mean"]) / stats["clear_mean"] < c1_lim] = 0
    c2[np.abs(stats["meas_max"] - stats["clear_max"]) / stats["clear_max"] < c2_lim] = 0
    c3[(line_diff > c3_lim_lower) & (line_diff < c3_lim_upper)] = 0
    c4[stats["meas_slope_nstd"] < c4_var] = 0
    c5[np.nanmax(np.abs(stats["meas_slope"] - stats["clear_slope"]), axis=1).reshape(-1, 1) < c5_lim_lower] = 0
    c6[(stats["clear_mean"] != 0) & ~np.isnan(stats["clear_mean"])] = 0

    csd_sum = c1 + c2 + c3 + c4 + c5 + c6
    csd = np.zeros_like(meas)
    csd[csd_sum > 0] = 1

    return csd, np.hstack((c1, c2, c3, c4, c5, c6))


def _dni_component_csd(dni: np.ndarray, dnics: np.ndarray, zen: np.ndarray):
    """DNI component using the Quesada-Ruiz style kc thresholding as in the MATLAB code."""
    z = np.arange(30, 90.0001, 0.01).reshape(-1)
    kc_lims = np.flip(np.linspace(0.5, 0.9, len(z))).reshape(-1)
    inds = _nearest_indices(z, zen.reshape(-1))
    kc_lim = kc_lims[inds].reshape(-1, 1)

    kcb = dni / dnics
    csd_dni = np.ones_like(dni)
    csd_dni[kcb > kc_lim] = 0
    criteria_dni = csd_dni.copy()

    return csd_dni, criteria_dni


def _duration_filter(csd_overall: np.ndarray, zen: np.ndarray, window_length: int, tolerance: int):
    """Duration filter with the same Hankel-centering treatment as the MATLAB implementation."""
    csdh_duration = np.nansum(_hankel_window(csd_overall, window_length), axis=1).reshape(-1, 1)
    half_window = int(window_length / 2)
    csdh_duration = np.vstack((np.full((half_window, 1), np.nan), csdh_duration[:len(csdh_duration) - half_window]))
    csd_duration = np.zeros_like(csd_overall)
    csd_duration[csdh_duration > tolerance] = 1

    A = np.arange(1, len(zen) + 1).reshape(-1, 1)
    B = np.where(np.round(zen.reshape(-1)) == 85)[0] + 1
    if len(B) == 0:
        distance_to_nearest_sunrise_set = np.full((len(zen), 1), np.inf)
    else:
        distance_to_nearest_sunrise_set = np.min(np.abs(A - B.reshape(1, -1)), axis=1).reshape(-1, 1)

    return csd_duration, distance_to_nearest_sunrise_set


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Main function
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

def BrightSun2020CSDc(zen: np.ndarray, ghi: np.ndarray, ghics: np.ndarray, dhi: np.ndarray, difcs: np.ndarray, datetime: np.ndarray, plot_figure=None):
    """
    The Bright-Sun clear-sky detection methodology, 2020.
    Cloudless sky CSDc.

    This function is a structural Python refactor of the original MATLAB implementation.
    Parameters, thresholds, criteria order, optimisation flow, and duration-filter logic are preserved.

    Parameters
    ----------
    zen : array-like
        Solar zenith angle [degrees], column vector.
    ghi : array-like
        Global horizontal irradiance [W m-2], column vector.
    ghics : array-like
        Clear-sky global horizontal irradiance [W m-2], column vector.
    dhi : array-like
        Diffuse horizontal irradiance [W m-2], column vector.
    difcs : array-like
        Clear-sky diffuse horizontal irradiance [W m-2], column vector.
    datetime : array-like
        Datetime input.
    plot_figure : any, optional
        If provided, diagnostic figures are plotted.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection for cloudless sky.
        1 means that clouds were detected.
        0 means clear sky.
    csd_tricomponent : np.ndarray
        Clear-sky detection from the Reno tri-component optimisation.
    ghics_optimised : np.ndarray
        Optimised clear-sky GHI curve.
    dnics_optimised : np.ndarray
        Optimised clear-sky DNI curve.
    difcs_optimised : np.ndarray
        Optimised clear-sky diffuse (DIF) curve.
    """

    warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
    warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0 for slice.*")

    # safety checks
    zen = _ensure_column_vector(zen, "zen")
    ghi = _ensure_column_vector(ghi, "ghi")
    ghics = _ensure_column_vector(ghics, "ghics", "var ghi must be a column vector")
    dhi = _ensure_column_vector(dhi, "dhi")
    difcs = _ensure_column_vector(difcs, "difcs", "var difcscs must be a column vector")

    datetime_dt, Dayth, Year = datetime_convert(datetime)

    if len(np.unique([len(ghi), len(ghics), len(zen), len(dhi), len(difcs), len(Dayth)])) != 1:
        raise ValueError("vars must be equal in length")

    # calculate DNI
    dhi[dhi > ghi] = ghi[dhi > ghi]
    difcs[difcs > ghics] = ghics[difcs > ghics]

    dnics = (ghics - difcs) / np.cos(np.deg2rad(zen))
    dni = (ghi - dhi) / np.cos(np.deg2rad(zen))

    # Clean the data for NaN values
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
    datetime_dt = datetime_dt[~idxs_nan]
    Dayth = Dayth[~idxs_nan]
    Year = Year[~idxs_nan]

    # Parameterisation
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
    lower_alpha_limit_DIF = 0.3

    window_1st_filter = 90
    tolerance_1st_filter = 10
    window_2nd_filter = 30
    tolerance_2nd_filter = 0

    # Initial CSD guess using Reno for Optimisation Only.
    csd_initial = _reno_initial_csd_ghi_only(ghi, ghics, window_length, reno_ghi_mean_diff, reno_ghi_max_diff, reno_ghi_lower_line_length, reno_ghi_upper_line_length, reno_ghi_var_diff, reno_ghi_slope_dev)

    if plot_figure is not None:
        lst_plot = np.array(datetime_dt)
        plt.figure(num="Initial CSD from Reno 2016 example", facecolor="w")
        plt.plot(lst_plot, ghi)
        plt.plot(lst_plot, dni)
        plt.plot(lst_plot, dhi)
        ghiCSD = ghi.copy()
        ghiCSD[csd_initial == 1] = np.nan
        plt.plot(lst_plot, ghiCSD, linewidth=2, color="k")
        plt.plot(lst_plot, ghics, "k:")
        dniCSD = dni.copy()
        dniCSD[csd_initial == 1] = np.nan
        difCSD = dhi.copy()
        difCSD[csd_initial == 1] = np.nan
        plt.plot(lst_plot, dniCSD, linewidth=2, color="k")
        plt.plot(lst_plot, difCSD, linewidth=2, color="k")
        plt.plot(lst_plot, dnics, "k:")
        plt.plot(lst_plot, difcs, "k:")
        plt.ylim([1, 1300])
        plt.legend(["GHI", "DNI", "DIF", "CSD", "Clear-sky irradiance"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.show()

    # Find all the unique days
    dtn = Year * 1000 + Dayth
    unique_days = np.unique(dtn)

    alpha_ghi = np.ones((len(unique_days), 1))
    alpha_dif = np.ones((len(unique_days), 1))
    alpha_dni = np.ones((len(unique_days), 1))

    # Optimisation of clear-sky irradiance
    if plot_figure is not None:
        old_ghics = ghics.copy()
        old_dnics = dnics.copy()
        old_difcs = difcs.copy()

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

        test_ghi = ghid[csdd == 0]
        test_ghics = ghicsd[csdd == 0]
        test_dif = difd[csdd == 0]
        test_difcs = difcsd[csdd == 0]
        test_dni = dnid[csdd == 0]
        test_dnics = dnicsd[csdd == 0]

        rmse_ghi = lambda a: np.sqrt(np.nanmean((test_ghi - a * test_ghics) ** 2))
        rmse_dif = lambda a: np.sqrt(np.nanmean((test_dif - a * test_difcs) ** 2))
        rmse_dni = lambda a: np.sqrt(np.nanmean((test_dni - a * test_dnics) ** 2))

        if np.sum(csdd == 0) > 60:
            current_alpha = alpha_ghi[d, 0]
            previous_alpha = np.nan
            iter_count = 0
            while iter_count < max_iterations and np.round(current_alpha * 10000) != np.round(previous_alpha * 10000):
                previous_alpha = current_alpha
                current_alpha = float(fmin(rmse_ghi, current_alpha, disp=False)[0])
                iter_count += 1
            alpha_ghi[d, 0] = current_alpha

            current_alpha = alpha_dif[d, 0]
            previous_alpha = np.nan
            iter_count = 0
            while iter_count < max_iterations and np.round(current_alpha * 10000) != np.round(previous_alpha * 10000):
                previous_alpha = current_alpha
                current_alpha = float(fmin(rmse_dif, current_alpha, disp=False)[0])
                iter_count += 1
            alpha_dif[d, 0] = current_alpha

            current_alpha = alpha_dni[d, 0]
            previous_alpha = np.nan
            iter_count = 0
            while iter_count < max_iterations and np.round(current_alpha * 10000) != np.round(previous_alpha * 10000):
                previous_alpha = current_alpha
                current_alpha = float(fmin(rmse_dni, current_alpha, disp=False)[0])
                iter_count += 1
            alpha_dni[d, 0] = current_alpha

        temp = alpha_ghi[d, 0]
        if temp > upper_alpha_limit:
            temp = upper_alpha_limit
        if temp < lower_alpha_limit:
            temp = lower_alpha_limit
        alpha_ghi[d, 0] = temp

        temp = alpha_dni[d, 0]
        if temp > upper_alpha_limit:
            temp = upper_alpha_limit
        if temp < lower_alpha_limit:
            temp = lower_alpha_limit
        alpha_dni[d, 0] = temp

        temp = alpha_dif[d, 0]
        if temp > upper_alpha_limit:
            temp = upper_alpha_limit
        if temp < lower_alpha_limit_DIF:
            temp = lower_alpha_limit_DIF
        alpha_dif[d, 0] = temp

        ghics[idxs] = ghics[idxs] * alpha_ghi[d, 0]
        difcs[idxs] = difcs[idxs] * alpha_dif[d, 0]
        dnics[idxs] = dnics[idxs] * alpha_dni[d, 0]

    if plot_figure is not None:
        lst_plot = np.array(datetime_dt)
        plt.figure(num="Example of clear-sky optimisation", facecolor="w")
        plt.plot(lst_plot, ghi)
        plt.plot(lst_plot, dni)
        plt.plot(lst_plot, dhi)
        plt.plot(lst_plot, ghiCSD, linewidth=2, color="k")
        plt.plot(lst_plot, old_ghics, "b:", linewidth=2)
        plt.plot(lst_plot, ghics, "r:", linewidth=2)
        plt.plot(lst_plot, dniCSD, linewidth=2, color="k")
        plt.plot(lst_plot, difCSD, linewidth=2, color="k")
        plt.plot(lst_plot, old_dnics, "b:", linewidth=2)
        plt.plot(lst_plot, old_difcs, "b:", linewidth=2)
        plt.plot(lst_plot, dnics, "r:", linewidth=2)
        plt.plot(lst_plot, difcs, "r:", linewidth=2)
        plt.legend(["GHI", "DIF", "DNI", "Reno CSD", "Clear-sky original", "Clear-sky optimised"])
        plt.xlabel("Time")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.show()

    # perform tri-component analysis.
    csd_ghi, criteria_ghi = _brightsun_component_csd(
        ghi, ghics, zen, window_length,
        BrightSunCriteria["ghi_c1_big_zenith"], BrightSunCriteria["ghi_c1_mean_diff"], BrightSunCriteria["ghi_c1_small_zenith"],
        BrightSunCriteria["ghi_c2_big_zenith"], BrightSunCriteria["ghi_c2_max_diff"], BrightSunCriteria["ghi_c2_small_zenith"],
        BrightSunCriteria["ghi_c3_lower_line_length"], BrightSunCriteria["ghi_c3_small_zenith"],
        BrightSunCriteria["ghi_c4_var_diff"],
        BrightSunCriteria["ghi_c5_slope_dev"], BrightSunCriteria["ghi_c5_small_zenith"],
        zenith_turn_around
    )

    csd_dif, criteria_dif = _brightsun_component_csd(
        dhi, difcs, zen, window_length,
        BrightSunCriteria["dif_c1_big_zenith"], BrightSunCriteria["dif_c1_mean_diff"], BrightSunCriteria["dif_c1_small_zenith"],
        BrightSunCriteria["dif_c2_big_zenith"], BrightSunCriteria["dif_c2_max_diff"], BrightSunCriteria["dif_c2_small_zenith"],
        BrightSunCriteria["dif_c3_lower_line_length"], BrightSunCriteria["dif_c3_small_zenith"],
        BrightSunCriteria["dif_c4_var_diff"],
        BrightSunCriteria["dif_c5_slope_dev"], BrightSunCriteria["dif_c5_small_zenith"],
        zenith_turn_around
    )

    csd_dni, criteria_dni = _dni_component_csd(dni, dnics, zen)

    # Overall criteria
    csd_overall = np.ones_like(ghi)
    csd_overall[csd_ghi + csd_dif + csd_dni == 0] = 0

    if plot_figure is not None:
        f = plt.figure(num="Demonstrate that the tri-componenet is working", facecolor="w")
        f.set_size_inches(10, 2.5)
        plt.plot(ghi)
        plt.plot(dni)
        plt.plot(dhi)
        ghiCSD = ghi.copy()
        ghiCSD[csd_ghi == 1] = np.nan
        dniCSD = dni.copy()
        dniCSD[csd_dni == 1] = np.nan
        difCSD = dhi.copy()
        difCSD[csd_dif == 1] = np.nan
        allCSD = ghi.copy()
        allCSD[csd_overall == 1] = np.nan
        plt.plot(ghiCSD, linewidth=4, color="k")
        plt.plot(ghics, "k:")
        plt.plot(allCSD, linewidth=2, color="g")
        plt.plot(dnics, "k:")
        plt.plot(difcs, "k:")
        plt.plot(dniCSD, linewidth=2, color="k")
        plt.plot(difCSD, linewidth=2, color="k")
        plt.ylim([0, 1300])
        plt.legend(["GHI", "DNI", "DIF", "CSD", "CS", "combined CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.show()

    if plot_figure is not None:
        major_line_width = 20
        line_sep = (major_line_width - 3) / 5
        cols = np.array([[199, 233, 180], [127, 205, 187], [65, 182, 196], [44, 127, 184], [37, 52, 148]], dtype=float) / 255.0
        f = plt.figure(num="explore the nature of Reno criteria", facecolor="w")
        f.set_size_inches(10, 2.5)
        plt.plot(ghi)
        plt.plot(dni)
        plt.plot(dhi)
        plt.plot(ghics, "k:")
        plt.plot(dnics, "k:")
        plt.plot(difcs, "k:")
        for c in range(5):
            temp = ghi.copy()
            crit = criteria_ghi[:, c:c + 1]
            temp[crit == 1] = np.nan
            plt.plot(temp, linewidth=major_line_width - line_sep * (c + 1), color=cols[c, :])
        csdghi = ghi.copy()
        csdghi[csd_ghi == 1] = np.nan
        plt.plot(csdghi, "m")
        csdghi = ghi.copy()
        csdghi[csd_overall == 1] = np.nan
        plt.plot(csdghi, "r", linewidth=2)

        csddni = dni.copy()
        csddni[csd_dni == 1] = np.nan
        plt.plot(csddni, "m")
        csddni = dni.copy()
        csddni[csd_overall == 1] = np.nan
        plt.plot(csddni, "r", linewidth=2)

        for c in range(5):
            temp = dhi.copy()
            crit = criteria_dif[:, c:c + 1]
            temp[crit == 1] = np.nan
            plt.plot(temp, linewidth=major_line_width - line_sep * (c + 1), color=cols[c, :])
        csddif = dhi.copy()
        csddif[csd_dif == 1] = np.nan
        plt.plot(csddif, "m")
        csddif = dhi.copy()
        csddif[csd_overall == 1] = np.nan
        plt.plot(csddif, "r", linewidth=2)

        plt.ylim([0, 1300])
        plt.legend(["GHI", "DNI", "DIF", "GHIcs", "DNIcs", "DIFcs", "c1", "c2", "c3", "c4", "c5", "CSD_{g/b/d}", "tri-CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.show()

    # Duration criteria
    if window_1st_filter <= 1:
        if tolerance_1st_filter == window_1st_filter:
            tolerance_1st_filter = 2
        window_1st_filter = 2

    if window_2nd_filter <= 1:
        if tolerance_2nd_filter == window_2nd_filter:
            tolerance_2nd_filter = 2
        window_2nd_filter = 2

    csd_1st_duration, distance_to_nearest_sunrise_set = _duration_filter(csd_overall, zen, window_1st_filter, tolerance_1st_filter)
    csd_1st_duration[distance_to_nearest_sunrise_set < window_1st_filter] = 0

    csd_2nd_duration, _ = _duration_filter(csd_overall, zen, window_2nd_filter, tolerance_2nd_filter)

    window_3rd_duration = 10
    tolerance_3rd_duration = 2
    csd_3rd_duration, _ = _duration_filter(csd_overall, zen, window_3rd_duration, tolerance_3rd_duration)

    csd_2nd_duration[(csd_3rd_duration == 0) & (distance_to_nearest_sunrise_set < window_1st_filter)] = 0

    csd_nan_indexed = np.ones_like(ghi)
    csd_nan_indexed[csd_overall + csd_1st_duration + csd_2nd_duration == 0] = 0

    # outputs that are indexed.
    csd = np.ones((n_output, 1))
    csd[not_nan_indxs] = csd_nan_indexed

    ghics_optimised = np.full((n_output, 1), np.nan)
    dnics_optimised = np.full((n_output, 1), np.nan)
    difcs_optimised = np.full((n_output, 1), np.nan)

    ghics_optimised[not_nan_indxs] = ghics
    dnics_optimised[not_nan_indxs] = dnics
    difcs_optimised[not_nan_indxs] = difcs

    csd_tricomponent = np.full((n_output, 1), np.nan)
    csd_tricomponent[not_nan_indxs] = csd_overall

    # plots
    if plot_figure is not None:
        f = plt.figure(num="durational window example", facecolor="w")
        f.set_size_inches(10, 2.5)
        csdtri = ghi.copy()
        csdtri[csd_overall == 1] = np.nan
        csd60w = ghi.copy()
        csd60w[csd_1st_duration == 1] = np.nan
        csd10w = ghi.copy()
        csd10w[csd_2nd_duration == 1] = np.nan
        csdall = ghi.copy()
        csdall[csd_nan_indexed == 1] = np.nan
        plt.plot(ghi)
        plt.plot(csdtri, color=np.array([0, 0, 0]), linewidth=10)
        plt.plot(csd60w, color=np.array([215, 48, 31]) / 255.0, linewidth=7.5)
        plt.plot(csd10w, color=np.array([254, 196, 79]) / 255.0, linewidth=5)
        plt.plot(csdall, color=np.array([120, 198, 121]) / 255.0, linewidth=2.5)
        plt.legend(["GHI", "Tri-Component CSD", "Passed 90-min Filter", "Passed 30-min Filter", "Bright-Sun CSD"])
        plt.ylim([0, 1300])
        plt.xlabel("Time")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.show()

    if plot_figure is not None:
        colours = np.array([[215, 48, 39], [254, 224, 139], [145, 207, 96], [120, 198, 121]], dtype=float) / 255.0
        plt.figure(num="Bright-Sun CSD 2019 example", facecolor="w")
        plt.plot(ghi)
        plt.plot(dni)
        plt.plot(dhi)
        ghiCSD = ghi.copy()
        ghiCSD[csd_ghi == 1] = np.nan
        dniCSD = dni.copy()
        dniCSD[csd_dni == 1] = np.nan
        difCSD = dhi.copy()
        difCSD[csd_dif == 1] = np.nan
        plt.plot(ghics, "k:")
        plt.plot(ghiCSD, linewidth=12.5, color="k")
        ghiCSD = ghi.copy()
        ghiCSD[csd_overall == 1] = np.nan
        plt.plot(ghiCSD, linewidth=10, color=colours[0, :])
        ghiCSD = ghi.copy()
        ghiCSD[csd_1st_duration == 1] = np.nan
        plt.plot(ghiCSD, linewidth=7.5, color=colours[1, :])
        ghiCSD = ghi.copy()
        ghiCSD[csd_2nd_duration == 1] = np.nan
        plt.plot(ghiCSD, linewidth=5, color=colours[2, :])
        ghiCSD = ghi.copy()
        ghiCSD[csd_nan_indexed == 1] = np.nan
        plt.plot(ghiCSD, linewidth=2, color="m")
        plt.plot(dnics, "k:")
        plt.plot(difcs, "k:")
        plt.plot(dniCSD, linewidth=2, color="k")
        plt.plot(difCSD, linewidth=2, color="k")
        if len(ghi) > 20000:
            plt.xlim([1, 20000])
        plt.legend(["GHI", "DNI", "DIF", "Clear-sky irrad.", "Tri-component CSD", "Corroborating CSD", "90min window", "30min window CSD", "Final CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.show()

        colours = np.array([[215, 48, 39], [254, 224, 139], [145, 207, 96]], dtype=float) / 255.0
        plt.figure(num="Bright-Sun CSD 2019 example simple", facecolor="w")
        plt.plot(ghi)
        plt.plot(ghics, "k:")
        ghiCSD = ghi.copy()
        ghiCSD[csd[not_nan_indxs] == 1] = np.nan
        plt.plot(ghiCSD, linewidth=4, color=colours[2, :])
        plt.legend(["GHI", "GHIcs", "CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.show()

    return [csd, csd_tricomponent, ghics_optimised, dnics_optimised, difcs_optimised]