import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import hankel


def datetime_convert(datetime: np.ndarray):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Garcia 2014 CSD
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

### References:
# García, R D; García, O E; Cuevas, E; Cachorro, V E; Ramos, R; Frutos, A M De. 2014.
# Solar radiation measurements compared to simulations at the BSRN Izaña station.
# Mineral dust radiative forcing and efficiency study.
# Journal of Geophysical Research: Atmospheres. 119, 179-194.

### Inputs:
# ghi         [W m-2]     Global horizontal irradiance, column vector.
# dhi         [W m-2]     Diffuse horizontal irradiance, column vector.
# zen         [degrees]   Zenith angle, column vector.
# aod550                  Total aerosol extinction at 550 nm, column vector.
# datetime                Datetime input.
# plot_figure             If defined, a diagnostic figure is plotted.

### Outputs:
# csd                     Clear-sky detection flag.
#                         1 = clouds suspected, 0 = clear.

### Notes:
# Assumption: all data is in 1-min resolution and all vectors perfectly match in time.
# This Python implementation is a direct structural translation of the MATLAB code.
# The original thresholds, criteria order, solar-noon handling, and plotting logic are preserved.


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

def Garcia2014CSD(ghi: np.ndarray, dhi: np.ndarray, zen: np.ndarray, aod550: np.ndarray, datetime: np.ndarray, plot_figure=None):
    """
    Garcia 2014 clear-sky detection method.

    This function is a direct structural Python translation of the MATLAB implementation.
    The original thresholds, solar-noon procedure, variability test, and plotting logic are preserved.
    """

    warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
    warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0 for slice.*")

    # safety checks
    ghi = _ensure_column_vector(ghi, "ghi")
    dhi = _ensure_column_vector(dhi, "dhi")
    zen = _ensure_column_vector(zen, "zen")
    aod550 = _ensure_column_vector(aod550, "aod550")
    Dayth, Year = datetime_convert(datetime)
    if len(np.unique([len(ghi), len(dhi), len(zen), len(aod550), len(Dayth)])) != 1:
        raise ValueError("vars must be equal in length")

    # pre allocation
    csd = np.zeros_like(ghi)

    # normalised total shortwave magnitude test
    b = 0.41 * aod550 + 1.09
    ghin = ghi / np.cos(np.deg2rad(zen)) ** b
    csd[ghin > 1250] = 1
    csd[(ghin < 1000) & (zen < 78.5)] = 1
    csd[(ghin < 900) & (zen > 78.5)] = 1

    # maximum diffuse shortwave test
    dmax = 150
    dlim = dmax * np.cos(np.deg2rad(zen)) ** 0.5
    csd[dhi > dlim] = 1

    # change in magnitude with time test
    s = 1365
    fdt = s * np.cos(np.deg2rad(zen))
    dfdt = fdt[:-1] - fdt[1:]
    dfdt = np.vstack((np.array([[np.nan]]), dfdt))
    dghi = ghi[:-1] - ghi[1:]
    dghi = np.vstack((np.array([[np.nan]]), dghi))
    C = 2
    ft_max = np.abs(dfdt) + C * np.cos(np.deg2rad(zen))

    # find the unique days in the time series
    days = Year * 1000 + Dayth
    unique_days = np.unique(days)

    # pre allocate solar noon
    solar_noon = np.zeros_like(ghi)

    # loop through each unique day
    for i in range(len(unique_days)):
        idx = np.where(days == unique_days[i])[0]
        solar_noon[idx] = np.min(zen[idx])

    ft_min = np.abs(dfdt) - (np.cos(np.deg2rad(solar_noon)) + 0.1) / np.cos(np.deg2rad(zen))

    # apply the filter
    csd[np.abs(dghi) > ft_max] = 1
    csd[np.abs(dghi) < ft_min] = 1

    # normalised diffuse ratio variability test
    difn = dhi / np.cos(np.deg2rad(zen)) ** b
    difn11 = _hankel_window(difn, 11)
    dif_std = np.nanstd(difn11, axis=1, ddof=1).reshape(-1, 1)

    # Cannot remotely get close to std dev of 0.0012 as suggested by the paper.
    # A value of 5 looks decent though.
    csd[dif_std > 5] = 1

    # figure
    if plot_figure is not None:
        plt.figure(num="Garcia CSD example", facecolor="w")
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