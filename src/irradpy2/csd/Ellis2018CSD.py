import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.optimize import fmin


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Ellis 2018 CSD
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

### References:
# PVLIB detect_ghics at commit f4e4ad3bbe335fa49bf300ce53784d69d719ca98
# https://github.com/pvlib/pvlib-python/pull/596/commits

### Inputs:
# ghi         [W m-2]     Global horizontal irradiance, column vector.
# ghics       [W m-2]     Clear-sky global horizontal irradiance, column vector.
# plot_figure             If defined, a diagnostic figure is plotted.

### Outputs:
# csd                     Clear-sky detection flag.
#                         1 = clouds suspected, 0 = clear.

### Notes:
# Assumption: all data is in 1-min resolution.
# This Python implementation is a direct structural translation of the MATLAB code.
# The original parameterisation, optimisation flow, criteria order, and plotting logic are preserved.


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Helper functions
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

def _ensure_column_vector(x: np.ndarray, var_name: str, error_message: str = None):
    """Ensure input is a column vector, following the original MATLAB safety-check order."""
    x = np.asarray(x, dtype=float)
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


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Main function
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

def Ellis2018CSD(ghi: np.ndarray, ghics: np.ndarray, plot_figure=None):
    """
    Ellis 2018 clear-sky detection method.

    This function is a direct structural Python translation of the MATLAB implementation.
    The original criteria thresholds, optimisation order, and output logic are preserved.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance [W m-2], column vector.
    ghics : array-like
        Clear-sky global horizontal irradiance [W m-2], column vector;
        equal in length to ``ghi``.
    plot_figure : any, optional
        If provided, a diagnostic figure illustrating the outcome is plotted.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        1 means that clouds were detected.
        0 means clear sky.
    """

    warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
    warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0 for slice.*")

    # safety checks
    ghi = _ensure_column_vector(ghi, "ghi")
    ghics = _ensure_column_vector(ghics, "ghics", "var ghics must be a column vector")
    if len(ghi) != len(ghics):
        raise ValueError("vars ghi and ghics must be equal")

    # Parameterisation for the Ellis method.
    mean_diff = 79.144
    max_diff = 59.152
    lower_line_length = -41.416
    upper_line_length = 77.789
    var_diff = 0.00745
    slope_dev = 68.579

    window_length = 10
    max_iterations = 20

    # produce Hankel matrices for GHI and GHIcs as defined by window_length.
    ghi_window = _hankel_window(ghi, window_length)
    ghics_window = _hankel_window(ghics, window_length)

    # calculate measurement statistics for ghi.
    meas_mean = np.nanmean(ghi_window, axis=1).reshape(-1, 1)
    meas_max = np.nanmax(ghi_window, axis=1).reshape(-1, 1)
    meas_diff = np.vstack((np.diff(ghi_window, axis=0), np.full((1, window_length), np.nan)))
    meas_slope = np.vstack((np.diff(ghi_window, axis=0), np.full((1, window_length), np.nan)))
    meas_slope_nstd = np.nanstd(meas_slope, axis=1, ddof=1).reshape(-1, 1) / meas_mean
    meas_line_length = np.nansum(np.sqrt(meas_diff * meas_diff), axis=1).reshape(-1, 1)

    # Repeat the above for GHI clear sky statistics.
    clear_mean = np.nanmean(ghics_window, axis=1).reshape(-1, 1)
    clear_max = np.nanmax(ghics_window, axis=1).reshape(-1, 1)
    clear_diff = np.vstack((np.diff(ghics_window, axis=0), np.full((1, window_length), np.nan)))
    clear_slope = np.vstack((np.diff(ghics_window, axis=0), np.full((1, window_length), np.nan)))

    # initialise the criteria matrices where 1 = cloud.
    c1 = np.ones_like(ghi)
    c2 = np.ones_like(ghi)
    c3 = np.ones_like(ghi)
    c4 = np.ones_like(ghi)
    c5 = np.ones_like(ghi)
    c6 = np.ones_like(ghi)

    # initial alpha and iteration values.
    alpha = 1.0
    previous_alpha = np.nan
    iteration = 0

    # begin optimisation with a while loop.
    while np.round(alpha * 10000) != np.round(previous_alpha * 10000) and iteration <= max_iterations:
        clear_line_length = np.nansum(np.sqrt(alpha ** 2 * clear_diff ** 2), axis=1).reshape(-1, 1)
        line_diff = meas_line_length - clear_line_length

        c1[np.abs(meas_mean - alpha * clear_mean) < mean_diff] = 0
        c2[np.abs(meas_max - alpha * clear_max) < max_diff] = 0
        c3[(line_diff > lower_line_length) & (line_diff < upper_line_length)] = 0
        c4[meas_slope_nstd < var_diff] = 0
        c5[np.nanmax(np.abs(meas_slope - alpha * clear_slope), axis=1).reshape(-1, 1) < slope_dev] = 0
        c6[(clear_mean != 0) & ~np.isnan(clear_mean)] = 0

        # should any criteria be 1(cloud), then the time-step is deemed cloudy.
        csd = c1 + c2 + c3 + c4 + c5 + c6
        csd[csd > 0] = 1

        # isolate clear measured ghi and corresponding clear-sky ghi.
        test_ghi = ghi[csd == 0]
        test_ghics = ghics[csd == 0]

        # store the current alpha.
        previous_alpha = alpha

        # run a Nelder-Mead search of the RMSE through alpha.
        rmse = lambda alpha_value: np.sqrt(np.nanmean((test_ghi - alpha_value * test_ghics) ** 2))
        alpha = float(fmin(rmse, alpha, disp=False)[0])

        # update iteration count.
        iteration += 1

    # plot an example csd from the input data.
    if plot_figure is not None:
        plt.figure(num="Ellis 2018 CSD example", facecolor="w")
        plt.plot(ghi.reshape(-1))
        CSD = ghi.reshape(-1).copy()
        CSD[csd.reshape(-1) == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd