import numpy as np
import matplotlib.pyplot as plt


#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#
### Batlles 1999 CSD
#-----##-----##-----##-----##-----##-----##-----##-----##-----##-----##-----#

### References:
# Batlles, F. J., Olmo, F. J., Tovar, J., & Alados-Arboledas, L. (1999).
# Comparison of cloudless sky parametrizations of solar irradiance at various spanish midlatitude locations.
# Theoretical and Applied Climatology, 66, 81-93.
#
# Original MATLAB version coded by Jamie M. Bright, 11/2018.

### Inputs:
# ghi  [W m-2]       Global horizontal irradiance, column vector.
# dhi  [W m-2]       Diffuse horizontal irradiance, column vector.
# exth [W m-2]       Horizontal projection of extraterrestrial irradiance, column vector.
# zen  [degrees]     Zenith angle, column vector.
# plot_figure        If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method.

### Outputs:
# csd                Clear-sky detection flag.
#                    0 = clear
#                    1 = cloudy

### Notes:
# Assumption: all data is in 1-min resolution and all vectors perfectly match each other in terms of time stamps.
# This Python function is translated strictly from the original MATLAB code.
# The parameter names, threshold expressions, decision order, and intermediate logic are intentionally preserved.

### Codes:
def Batlles1999CSD(ghi: np.ndarray, dhi: np.ndarray, exth: np.ndarray, zen: np.ndarray, plot_figure=None):
    """
    Batlles 1999 clear-sky detection model.

    References
    ----------
    Batlles, F. J., Olmo, F. J., Tovar, J., & Alados-Arboledas, L. (1999).
    Comparison of cloudless sky parametrizations of solar irradiance at various spanish midlatitude locations.
    Theoretical and Applied Climatology, 66, 81-93.

    Original MATLAB version coded by Jamie M. Bright, 11/2018.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance [W m-2], column vector.
    dhi : np.ndarray
        Diffuse horizontal irradiance [W m-2], column vector.
    exth : np.ndarray
        Horizontal projection of extraterrestrial irradiance [W m-2], column vector.
    zen : np.ndarray
        Zenith angle [degrees], column vector.
    plot_figure : optional
        If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method.

    Outputs
    -------
    csd : np.ndarray
        Clear-sky detection flag.
        0 = clear
        1 = cloudy

    Notes
    -----
    Assumption: all data is in 1-min resolution and all vectors perfectly match each other in terms of time stamps.
    This function is a strict Python translation of the MATLAB version.
    The parameter names, threshold expressions, decision order, and intermediate logic are intentionally preserved.
    """

    # safety checks
    ghi = np.asarray(ghi, dtype=float)
    if ghi.ndim == 1:
        ghi = ghi.reshape(-1, 1)
    if ghi.shape[1] != 1:
        ghi = ghi.T
        if ghi.ndim != 2 or ghi.shape[1] != 1:
            raise ValueError("var ghi must be a column vector")

    dhi = np.asarray(dhi, dtype=float)
    if dhi.ndim == 1:
        dhi = dhi.reshape(-1, 1)
    if dhi.shape[1] != 1:
        dhi = dhi.T
        if dhi.ndim != 2 or dhi.shape[1] != 1:
            raise ValueError("var dhi must be a column vector")

    zen = np.asarray(zen, dtype=float)
    if zen.ndim == 1:
        zen = zen.reshape(-1, 1)
    if zen.shape[1] != 1:
        zen = zen.T
        if zen.ndim != 2 or zen.shape[1] != 1:
            raise ValueError("var zen must be a column vector")

    exth = np.asarray(exth, dtype=float)
    if exth.ndim == 1:
        exth = exth.reshape(-1, 1)
    if exth.shape[1] != 1:
        exth = exth.T
        if exth.ndim != 2 or exth.shape[1] != 1:
            raise ValueError("var exth must be a column vector")

    if len(np.unique([len(ghi), len(dhi), len(zen), len(exth)])) != 1:
        raise ValueError("vars must be equal in length")

    # clearness index
    kt = ghi / exth

    # diffuse fraction
    k = dhi / ghi

    # criteria "high values of kt and low values of k corrrespond to cloudless conditions, while cloudy conditions present a combination of higher k and lower kt... the expressions defining the minimum kt, ktt, and the maximum k, kk allowed for cloudless conditions are":
    h = 90 - zen
    ktt = -0.3262 - 0.0032 * h + 0.6842 * np.log10(h)
    kk = 1.0827 - 0.3893 * np.log10(h)

    # clear-sky detection time series. 0=clear, 1=cloudy.
    csd = np.zeros(ghi.shape)
    csd[(kt < ktt) & (k > kk)] = 1

    ## figure example
    if plot_figure is not None:
        plt.figure(num="Batlles CSD example", facecolor="w")
        plt.plot(ghi)
        CSD = ghi.copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd