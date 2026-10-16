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


def janjai(sza: np.ndarray, datetime: np.ndarray, altitude: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original Janjai clear-sky model.

    References
    ----------
    Janjai, S., Sricharoen, K., & Pattarapanitchai, S. (2011). Semi-empirical models for the estimation of clear sky solar global and direct normal irradiances in the tropics.
    Applied Energy, 88(12), 4749-4755.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    altitude : np.ndarray
        Location altitude [m]
    ang_alpha : np.ndarray
        Angstrom exponent
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [Eghjanjai, Ebnjanjai, Edhjanjai] : list of np.ndarray
        Eghjanjai : Global horizontal irradiance [W m-2]
        Ebnjanjai : Direct normal irradiance [W m-2]
        Edhjanjai : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    altitude = np.asarray(altitude, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    ama = am * np.exp(-0.0001184 * altitude)

    # coefficients
    a1 = 0.778227
    a2 = 0.71640
    b1 = 1.198932
    b2 = 0.35320
    c1 = -0.106634
    c2 = 0.10126
    d1 = 0.337373
    d2 = 0.841372
    e1 = 0.009181
    e2 = 0.017649
    f1 = -0.009852
    f2 = 0.004851
    g1 = 0.482012
    g2 = -0.48286

    # direct normal irradiance
    A2 = a2 * Eext * (np.cos(sza)) ** b2
    B2 = c2 + d2 * ang_beta + e2 * ang_alpha + f2 * wv + g2 * ozone
    Ebnjanjai = A2 * np.exp(-B2 * ama)

    # global horizontal irradiance
    A1 = a1 * Eext * (np.cos(sza)) ** b1
    B1 = c1 + d1 * ang_beta + e1 * ang_alpha + f1 * wv + g1 * ozone
    Eghjanjai = A1 * np.exp(-B1 * ama)

    # diffuse horizontal irradiance
    Edhjanjai = Eghjanjai - Ebnjanjai * np.cos(sza)

    # Quality control
    lower = 0
    Ebnjanjai[Ebnjanjai < lower] = 0
    Edhjanjai[Edhjanjai < lower] = 0
    Eghjanjai[Eghjanjai < lower] = 0

    return [Eghjanjai, Ebnjanjai, Edhjanjai]