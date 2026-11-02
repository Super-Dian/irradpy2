import numpy as np
import matplotlib.pyplot as plt


def Reno2016CSD(ghi: np.ndarray, ghics: np.ndarray, plot_figure=None):
    """
    Reno 2016 clear-sky detection model.

    References
    ----------
    Reno, M. J. and Hansen, C. W. (2016).
    Identification of periods of clear sky irradiance in time series of GHI measurements.
    Renewable Energy, 90, 520-531.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance, column vector.
        Not necessary to be continuous so long as the ghics perfectly corresponds.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance, column vector.
    plot_figure : optional
        If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method, e.g. 1.

    Outputs
    -------
    csd : np.ndarray
        Clear-sky detection.
        A flag of 1 means that clouds were detected whereas a flag of 0 means that the hour is clear.

    Example
    -------
    csd = Reno2016CSD(ghi, ghics, 1)
    """

    ghi = np.array(ghi)
    ghics = np.array(ghics)

    if not (ghi.ndim == 1 or (ghi.ndim == 2 and ghi.shape[1] == 1)):
        ghi = ghi.T
        if not (ghi.ndim == 1 or (ghi.ndim == 2 and ghi.shape[1] == 1)):
            raise ValueError("var ghi must be a column vector")

    if not (ghics.ndim == 1 or (ghics.ndim == 2 and ghics.shape[1] == 1)):
        ghics = ghics.T
        if not (ghics.ndim == 1 or (ghics.ndim == 2 and ghics.shape[1] == 1)):
            raise ValueError("var ghics must be a column vector")

    ghi = np.array(ghi, dtype=float).reshape(-1)
    ghics = np.array(ghics, dtype=float).reshape(-1)

    if len(np.unique(np.array([len(ghi), len(ghics)]))) != 1:
        raise ValueError("vars must be equal in length")

    mean_lim = 75
    max_lim = 75
    lower_L_lim = -5
    upper_L_lim = 10
    sigma_lim = 0.005
    X_lim = 8
    temporal_resolution = 1
    interval_size = 10

    ghi_window = np.full((len(ghi), interval_size), np.nan, dtype=float)
    ghics_window = np.full((len(ghics), interval_size), np.nan, dtype=float)

    for i in range(interval_size):
        ghi_window[:len(ghi) - i, i] = ghi[i:]
        ghics_window[:len(ghics) - i, i] = ghics[i:]

    c1 = np.zeros(ghi.shape, dtype=float)
    c2 = np.zeros(ghi.shape, dtype=float)
    c3 = np.zeros(ghi.shape, dtype=float)
    c4 = np.zeros(ghi.shape, dtype=float)
    c5 = np.zeros(ghi.shape, dtype=float)

    csi_mean = np.nanmean(ghics_window, axis=1)
    ghi_mean = np.nanmean(ghi_window, axis=1)
    c1[np.abs(csi_mean - ghi_mean) > mean_lim] = 1

    ghi_max = np.nanmax(ghi_window, axis=1)
    csi_max = np.nanmax(ghics_window, axis=1)
    c2[np.abs(csi_max - ghi_max) > max_lim] = 1

    ghi_l = np.nansum(np.sqrt((ghi_window[:, 1:interval_size] - ghi_window[:, 0:interval_size - 1]) ** 2 + temporal_resolution ** 2), axis=1)
    ghics_l = np.nansum(np.sqrt((ghics_window[:, 1:interval_size] - ghics_window[:, 0:interval_size - 1]) ** 2 + temporal_resolution ** 2), axis=1)
    L_upper = ghics_l + upper_L_lim
    L_lower = ghics_l + lower_L_lim
    c3[(ghi_l < L_lower) & (ghi_l > L_upper)] = 1

    s = (ghi_window[:, 1:interval_size] - ghi_window[:, 0:interval_size - 1]) / temporal_resolution
    sigma = (1.0 / ghi_mean) * np.sqrt(np.nanmean((s - np.nanmean(s, axis=1, keepdims=True)) ** 2, axis=1))
    c4[sigma > sigma_lim] = 1

    ghi_x = ghi_window[:, 1:interval_size] - ghi_window[:, 0:interval_size - 1]
    ghics_x = ghics_window[:, 1:interval_size] - ghics_window[:, 0:interval_size - 1]
    X = np.nanmax(np.abs(ghi_x - ghics_x), axis=1)
    c5[X > X_lim] = 1

    csd = c1 + c2 + c3 + c4 + c5
    csd[csd > 0] = 1
    csd[np.isnan(ghi)] = np.nan

    if plot_figure is not None:
        plt.figure(num="Reno CSD example", facecolor="w")
        plt.plot(ghi)
        CSD = np.array(ghi, dtype=float)
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return np.asarray(csd, dtype=float).reshape(-1, 1)