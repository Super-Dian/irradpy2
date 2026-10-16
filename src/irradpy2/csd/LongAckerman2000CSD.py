import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _ensure_column_vector(x, var_name):
    """
    Ensure input is a numeric column vector.

    Parameters
    ----------
    x : array-like
        Input vector.
    var_name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Column vector with shape (n, 1).

    Raises
    ------
    ValueError
        If the input cannot be converted into a column vector.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.ndim != 2 or x.shape[1] != 1:
        x = x.T
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"var {var_name} must be a column vector")

    return x


def _rolling_std_centered_11(x):
    """
    Replicate the MATLAB 11-point centered standard deviation construction:

    difn11 = [difn(1:end-10), difn(2:end-9), ..., difn(11:end)]
    dif_std = [NaN(5,1); std(difn11,0,2,'omitnan'); NaN(5,1)]'

    Parameters
    ----------
    x : np.ndarray
        1D input array.

    Returns
    -------
    np.ndarray
        1D array of same length as x, with NaN at both ends.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)

    if n < 11:
        return np.full(n, np.nan, dtype=float)

    windows = np.column_stack([x[i:n - 10 + i] for i in range(11)])
    std_mid = np.nanstd(windows, axis=1, ddof=1)

    out = np.full(n, np.nan, dtype=float)
    out[5:n - 5] = std_mid
    return out


def LongAckerman2000CSD(ghi, dhi, zen, datetime: np.ndarray, plot_figure=None):
    """
    Long and Ackerman (2000) clear-sky detection model.

    This is a structural Python translation of the original MATLAB code.
    Parameters, thresholds, decision logic, and output meaning are preserved.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance, column vector.
    dhi : array-like
        Diffuse horizontal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    datetime : np.ndarray
        Datetime input.
    plot_figure : any, optional
        If provided, a figure illustrating the result is plotted.

    Returns
    -------
    np.ndarray
        Clear-sky detection flag as a column vector.
        1 means clouds were detected.
        0 means clear sky.
    """
    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    ghi = _ensure_column_vector(ghi, "ghi")
    dhi = _ensure_column_vector(dhi, "dhi")
    zen = _ensure_column_vector(zen, "zen")
    Dayth, Year = datetime_convert(datetime)
    Dayth = Dayth.reshape(-1)
    Year = Year.reshape(-1)

    if len(np.unique([len(ghi), len(dhi), len(zen), len(Dayth)])) != 1:
        raise ValueError("vars must be equal in length")

    ghi_1d = ghi.reshape(-1)
    dif_1d = dhi.reshape(-1)
    zen_1d = zen.reshape(-1)

    # ------------------------------------------------------------------
    # Begin
    # ------------------------------------------------------------------
    csd = np.zeros_like(ghi_1d, dtype=float)

    # ------------------------------------------------------------------
    # Normalised total shortwave magnitude test
    # ------------------------------------------------------------------
    b = 1.2
    with np.errstate(divide="ignore", invalid="ignore"):
        ghin = ghi_1d / (np.cos(np.deg2rad(zen_1d)) ** b)

    csd[ghin > 1250] = 1
    csd[(ghin < 1000) & (zen_1d < 78.5)] = 1
    csd[(ghin < 900) & (zen_1d > 78.5)] = 1

    # ------------------------------------------------------------------
    # Maximum diffuse shortwave test
    # ------------------------------------------------------------------
    dmax = 150
    dlim = dmax * (np.cos(np.deg2rad(zen_1d)) ** 0.5)
    csd[dif_1d > dlim] = 1

    # ------------------------------------------------------------------
    # Change in magnitude with time test
    # ------------------------------------------------------------------
    s = 1365.0
    fdt = s * np.cos(np.deg2rad(zen_1d))

    dfdt = fdt[:-1] - fdt[1:]
    dfdt = np.concatenate(([np.nan], dfdt))

    dghi = ghi_1d[:-1] - ghi_1d[1:]
    dghi = np.concatenate(([np.nan], dghi))

    C = 2.0
    ft_max = np.abs(dfdt) + C * np.cos(np.deg2rad(zen_1d))

    days = Year * 1000 + Dayth
    unique_days = np.unique(days)
    solar_noon = np.zeros_like(ghi_1d, dtype=float)

    for ud in unique_days:
        idx = np.where(days == ud)[0]
        solar_noon[idx] = np.min(zen_1d[idx])

    with np.errstate(divide="ignore", invalid="ignore"):
        ft_min = np.abs(dfdt) - (np.cos(np.deg2rad(solar_noon)) + 0.1) / np.cos(np.deg2rad(zen_1d))

    csd[np.abs(dghi) > ft_max] = 1
    csd[np.abs(dghi) < ft_min] = 1

    # ------------------------------------------------------------------
    # Normalised diffuse ratio variability test
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        difn = dif_1d / (np.cos(np.deg2rad(zen_1d)) ** b)

    dif_std = _rolling_std_centered_11(difn)

    # Same threshold as the MATLAB implementation
    csd[dif_std > 5] = 1

    csd = csd.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Long and Ackerman CSD example", facecolor="w")
        plt.plot(ghi_1d)
        CSD = ghi_1d.copy()
        CSD[csd.reshape(-1) == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd