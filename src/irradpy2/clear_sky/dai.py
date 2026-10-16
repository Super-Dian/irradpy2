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


def dai(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, ang_beta: np.ndarray, ang_alpha: np.ndarray):
    """
    Dai 2014 clear-sky model.

    References
    ----------
    Dai, Q., & Fang, X. (2014).
    A simple model to predict solar radiation under clear sky conditions.

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
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ang_alpha : np.ndarray
        Angstrom exponent

    Returns
    -------
    [EghDAI, EbnDAI, EdhDAI] : list of np.ndarray
        EghDAI : Global horizontal irradiance [W m-2]
        EbnDAI : Direct normal irradiance [W m-2]
        EdhDAI : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    sza_deg = sza / np.pi * 180
    mr = 1 / (np.cos(sza) + 0.15 * (93.885 - sza_deg) ** (-1.253))
    ma = mr * pressure / 1013

    # aerosol optical depth
    aod500 = ang_beta * 0.5 ** (-ang_alpha)
    tau = 0.744 * aod500

    # transmittance of Rayleigh scattering
    TR = np.exp(-0.103 * ma ** 0.571)

    # transmittance of water vapour
    TW = np.exp(-0.081 * (wv * mr) ** 0.213)

    # transmittance of aerosol
    TA = np.exp(-(tau ** 0.91) * mr ** 0.87)

    # direct normal irradiance
    EbnDAI = Eext * TA * TW * TR

    # diffuse horizontal irradiance
    EdhDAI = (0.143 + 0.113 * np.cos(sza) - 0.0485 * wv + tau) * (Eext - EbnDAI) * np.cos(sza)

    # global horizontal irradiance
    EghDAI = EbnDAI * np.cos(sza) + EdhDAI

    # Quality control
    lower = 0
    EbnDAI[EbnDAI < lower] = lower
    EdhDAI[EdhDAI < lower] = lower
    EghDAI[EghDAI < lower] = lower

    return [EghDAI, EbnDAI, EdhDAI]