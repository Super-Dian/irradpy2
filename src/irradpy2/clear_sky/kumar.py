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


def kumar(sza: np.ndarray, datetime, press: np.ndarray):
    """
    Original Kumar clear-sky model.

    References
    ----------
    Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
    Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
    Solar Energy, 86(8), 2145-2169.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    press : np.ndarray
        Local barometric pressure [mb]

    Returns
    -------
    [Eghkumar, Ebnkumar, Edhkumar] : list of np.ndarray
        Eghkumar : Global horizontal irradiance [W m-2]
        Ebnkumar : Direct normal irradiance [W m-2]
        Edhkumar : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # absolute air mass
    mk = ((1229 + (614 * np.cos(sza)) ** 2) ** 0.5 - 614 * np.cos(sza)) * press / 1013.25

    # direct beam irradiance
    Ebnkumar = Eext * 0.56 * (np.exp(-0.65 * mk) + np.exp(-0.095 * mk))

    # diffuse horizontal irradiance
    Edhkumar = (0.2710 * Eext - 0.294 * Ebnkumar) * np.cos(sza)

    # global horizontal irradiance
    Eghkumar = Ebnkumar * np.cos(sza) + Edhkumar

    # Quality control
    lower = 0
    Ebnkumar[Ebnkumar < lower] = 0
    Edhkumar[Edhkumar < lower] = 0
    Eghkumar[Eghkumar < lower] = 0

    return [Eghkumar, Ebnkumar, Edhkumar]