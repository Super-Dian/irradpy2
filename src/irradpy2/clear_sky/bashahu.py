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


def bashahu(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, wv: np.ndarray):
    """
    Original Bashahu clear-sky model.

    References
    ----------
    Bashahu, M., & Laplaze, D. (1994). An atmospheric model for computing solar radiation.
    Renewable Energy, 4(4), 455-458.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    press : np.ndarray
        Local barometric pressure [mb]
    ang_alpha : np.ndarray
        Angstrom exponent [dimensionless]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient [dimensionless]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghBashahu, EbnBashahu, EdhBashahu] : list of np.ndarray
        EghBashahu : Global horizontal irradiance [W m-2]
        EbnBashahu : Direct normal irradiance [W m-2]
        EdhBashahu : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    ama = am * press / 1013.25

    # direct beam radiation
    Td = 5.228 / ama * (np.exp(-0.0002254 * ama) - np.exp(-0.1409 * ama)) * np.exp(-ama * ang_beta / (0.6489 ** ang_alpha)) + 0.00165 * ama + 0.2022 * (1 - ama * ang_beta / (2.875 ** ang_alpha))
    Ta = 0.1055 + 0.07053 * np.log10(ama * wv + 0.07854) * np.exp(-ama * ang_beta / (1.519 ** ang_alpha))
    EbnBashahu = Eext * (Td - Ta)

    # diffuse horizontal irradiance
    EdhBashahu = 0.5 * Eext * (1 - Td) * (np.cos(sza) ** (4 / 3))

    # global horizontal irradiance
    EghBashahu = EbnBashahu * np.cos(sza) + EdhBashahu

    # Quality control
    lower = 0
    EbnBashahu[EbnBashahu < lower] = 0
    EdhBashahu[EdhBashahu < lower] = 0
    EghBashahu[EghBashahu < lower] = 0

    return [EghBashahu, EbnBashahu, EdhBashahu]