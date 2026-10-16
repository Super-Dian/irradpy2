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


def yangwalsh(zsa: np.ndarray, datetime):
    """
    Original Yang & Walsh clear-sky model.

    References
    ----------
    Yang, D., Walsh, W. M., & Jirutitijaroen, P. (2014).
    Estimation and applications of clear sky global horizontal irradiance at the equator.
    Journal of Solar Energy Engineering, 136(3), 034505.

    Parameters
    ----------
    zsa : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input

    Returns
    -------
    Eghyangwalsh : np.ndarray
        Global horizontal irradiance [W m-2]
    """

    zsa = np.asarray(zsa, dtype=float)
    Dayth, Year = datetime_convert(datetime)

    # extraterrestrial irradiance
    Esc = 1362
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # global horizontal irradiance
    Eghyangwalsh = 0.8298 * Eext * (np.cos(zsa) ** 1.3585) * np.exp(-0.00135 * (np.pi / 2 - zsa) / np.pi * 180)

    # Quality control
    lower = 0
    Eghyangwalsh[Eghyangwalsh < lower] = 0

    return Eghyangwalsh