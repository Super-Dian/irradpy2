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


def hs(sza: np.ndarray):
    """
    HS 1976 clear-sky model.

    References
    ----------
    Schulze, R. E. (1976).
    A physically based method of estimating solar radiation from suncards.

    Hourwitz, B. (1945).
    Insolation in relation to cloudiness and cloud density.

    Hourwitz, B. (1946).
    Insolation in relation to cloud type.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using
    routine hourly global irradiance measurements in Romania.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]

    Returns
    -------
    [EghHS, EbnHS, EdhHS] : list of np.ndarray
        EghHS : Global horizontal irradiance [W m-2]
        EbnHS : Direct normal irradiance [W m-2]
        EdhHS : Diffuse horizontal irradiance [W m-2]

    Notes
    -----
    Direct normal irradiance is based on the Hourwitz model, while diffuse
    horizontal irradiance is based on the Schulze formulation.
    """

    sza = np.asarray(sza, dtype=float)
    lower = 0

    # direct normal irradiance
    EbnHS = 1098 * np.exp(-0.057 / np.cos(sza))
    EbnHS[EbnHS < lower] = lower

    # diffuse horizontal irradiance
    EdhHS = 94.23 * np.cos(sza) ** 0.5
    EdhHS[EdhHS < lower] = lower

    # global horizontal irradiance
    EghHS = EbnHS * np.cos(sza) + EdhHS
    EghHS[EghHS < lower] = lower

    return [EghHS, EbnHS, EdhHS]