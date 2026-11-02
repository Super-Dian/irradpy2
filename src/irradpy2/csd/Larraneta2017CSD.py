import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def datetime_convert(datetime: np.ndarray):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def _ensure_column_vector(x, var_name):
    """
    Ensure input is a column vector.

    Parameters
    ----------
    x : array-like
        Input array.
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


def _nanmean_hourly(x):
    """
    Compute hourly mean using 60-sample blocks, matching MATLAB reshape logic.

    Parameters
    ----------
    x : array-like
        1D array.

    Returns
    -------
    np.ndarray
        Hourly mean values.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    usable_len = len(x) - (len(x) % 60)

    if usable_len == 0:
        return np.array([], dtype=float)

    reshaped = x[:usable_len].reshape(60, usable_len // 60, order="F")
    return np.nanmean(reshaped, axis=0)


def Larraneta2017CSD(dni, dnics, zen, datetime: np.ndarray, plot_figure=None):
    """
    Larraneta 2017 Clear Sky Detection model.

    Parameters
    ----------
    dni : array-like
        Direct normal irradiance, column vector.
    dnics : array-like
        Clear-sky direct normal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    datetime : np.ndarray
        Datetime input.
    plot_figure : any, optional
        If defined, a figure illustrating the CSD output is plotted.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        1 means clouds were detected.
        0 means clear sky.
    """
    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    dni = _ensure_column_vector(dni, "dni")
    dnics = _ensure_column_vector(dnics, "dnics")
    zen = _ensure_column_vector(zen, "zen")

    Dayth, Year = datetime_convert(datetime)

    if len(np.unique([len(dni), len(zen), len(dnics), len(Dayth)])) != 1:
        raise ValueError("vars must be equal in length")

    dni_1d = dni.reshape(-1)
    dnics_1d = dnics.reshape(-1)
    zen_1d = zen.reshape(-1)

    # ------------------------------------------------------------------
    # Preamble
    # ------------------------------------------------------------------
    days = Year * 1000 + Dayth
    unique_days = np.unique(days)

    solar_noon = np.zeros_like(dni_1d, dtype=float)

    for day in unique_days:
        idx = np.where(days == day)[0]
        solar_noon[idx] = np.min(zen_1d[idx])

    int100pc = solar_noon
    int66pc = solar_noon * (2 / 3)
    int33pc = solar_noon * (1 / 3)
    int0pc = np.ones_like(solar_noon) * 90

    hint100pc = _nanmean_hourly(int100pc)
    hint66pc = _nanmean_hourly(int66pc)
    hint33pc = _nanmean_hourly(int33pc)
    hint0pc = _nanmean_hourly(int0pc)

    interval1 = np.vstack([hint0pc, hint33pc])
    interval2 = np.vstack([hint33pc, hint66pc])
    interval3 = np.vstack([hint66pc, hint100pc])

    # ------------------------------------------------------------------
    # Hourly test values
    # ------------------------------------------------------------------
    hdni = _nanmean_hourly(dni_1d)
    hdnics = _nanmean_hourly(dnics_1d)
    hzen = _nanmean_hourly(zen_1d)

    c1 = np.ones_like(hdni, dtype=float)
    c2 = np.ones_like(hdni, dtype=float)
    c3 = np.ones_like(hdni, dtype=float)

    kb = hdni / hdnics

    # ------------------------------------------------------------------
    # Test 1: Hourly mean criterion
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_pct = np.abs(100.0 * ((hdnics - hdni) / hdni))

    mask1 = (hzen < interval1[0, :]) & (hzen > interval1[1, :]) & (diff_pct < 35)
    mask2 = (hzen < interval2[0, :]) & (hzen > interval2[1, :]) & (diff_pct < 15)
    mask3 = (hzen < interval3[0, :]) & (hzen > interval3[1, :]) & (diff_pct < 10)

    c1[mask1] = 0
    c1[mask2] = 0
    c1[mask3] = 0
    c1[diff_pct < 2.5] = 0

    # ------------------------------------------------------------------
    # Test 2: Slope criterion
    # ------------------------------------------------------------------
    scs = np.concatenate(([np.nan], hdnics[1:] - hdnics[:-1]))
    sm = np.concatenate(([np.nan], hdni[1:] - hdni[:-1]))

    c2[np.sign(scs) == np.sign(sm)] = 0

    # ------------------------------------------------------------------
    # Test 3: Line length criterion
    # ------------------------------------------------------------------
    lcs = np.concatenate(([np.nan], np.sqrt((hdnics[1:] - hdnics[:-1]) ** 2)))
    lm = np.concatenate(([np.nan], np.sqrt((hdni[1:] - hdni[:-1]) ** 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        ldcs = np.abs(100.0 * ((lcs - lm) / lm))

    mask1 = (hzen < interval1[0, :]) & (hzen > interval1[1, :]) & (ldcs < 220)
    mask2 = (hzen < interval2[0, :]) & (hzen > interval2[1, :]) & (ldcs < 110)
    mask3 = (hzen < interval3[0, :]) & (hzen > interval3[1, :]) & (ldcs < 30)

    c3[mask1] = 0
    c3[mask2] = 0
    c3[mask3] = 0

    # ------------------------------------------------------------------
    # Apply criteria
    # ------------------------------------------------------------------
    criteria = c1 + c2 + c3

    x = np.arange(1, len(criteria) + 1, dtype=float)
    xx = np.linspace(1, len(hdni), len(hdni) * 60) if len(hdni) > 0 else np.array([], dtype=float)

    if len(criteria) > 1:
        criteria_1min = np.interp(xx, x, criteria)
    elif len(criteria) == 1:
        criteria_1min = np.full(len(hdni) * 60, criteria[0], dtype=float)
    else:
        criteria_1min = np.array([], dtype=float)

    criteria_1min = np.concatenate(
        [criteria_1min, np.full(len(dni_1d) - len(criteria_1min), np.nan)]
    )

    csd = np.zeros_like(dni_1d, dtype=float)
    csd[criteria_1min != 0] = 1
    csd = csd.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Figure example
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Larraneta CSD example", facecolor="w")
        plt.plot(dni_1d)
        CSD = dni_1d.copy()
        CSD[csd.reshape(-1) == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["DNI", "CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time [min]")
        plt.show()

    return csd