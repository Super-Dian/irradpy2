import numpy as np
import pandas as pd


def datetime_convert(datetime):
    """Convert a datetime sequence to (day-of-year, year) arrays.

    Parameters
    ----------
    datetime : array-like
        Datetime sequence parsed with ``pandas.to_datetime``; invalid entries
        become NaT. The original array shape is preserved.

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


def ideriah(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray):
    """
    Ideriah 1981 clear-sky model.

    References
    ----------
    Ideriah, F. J. K. (1981).
    A model for calculating direct and diffuse solar radiation.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using
    routine hourly global irradiance measurements in Romania.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    pressure : np.ndarray
        Local barometric pressure [mb]
    wv : np.ndarray
        Total precipitable water vapour [cm]
    ang_alpha : np.ndarray
        Angstrom exponent
    ang_beta : np.ndarray
        Angstrom turbidity coefficient

    Returns
    -------
    [EghIDERIAH, EbnIDERIAH, EdhIDERIAH] : list of np.ndarray
        EghIDERIAH : Global horizontal irradiance [W m-2]
        EbnIDERIAH : Direct normal irradiance [W m-2]
        EdhIDERIAH : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    sza_deg = sza / np.pi * 180
    amk = 1 / (np.cos(sza) + 0.15 * (93.885 - sza_deg) ** (-1.253))
    am = amk * pressure / 1013.25
    amb = amk * ang_beta

    Tscat = (5.228 / am) * (np.exp(-0.0002254 * am) - np.exp(-0.140875 * am)) * np.exp(-amb / 0.6489 ** ang_alpha) + 0.00165 * am + 0.2022 * (1 - amb / 2.875 ** ang_alpha)

    Tabs = (0.1055 + 0.07053 * np.log10(amk * wv + 0.07854)) * np.exp(-amb / 1.519 ** ang_alpha)

    # direct normal irradiance
    EbnIDERIAH = Eext * (Tscat - Tabs)

    # diffuse horizontal irradiance
    EdhIDERIAH = 0.5 * Tscat * Eext * np.cos(sza) * np.cos(sza) ** (1 / 3)

    # global horizontal irradiance
    EghIDERIAH = EbnIDERIAH * np.cos(sza) + EdhIDERIAH

    # Quality control
    lower = 0
    EbnIDERIAH[EbnIDERIAH < lower] = lower
    EdhIDERIAH[EdhIDERIAH < lower] = lower
    EghIDERIAH[EghIDERIAH < lower] = lower

    return [EghIDERIAH, EbnIDERIAH, EdhIDERIAH]