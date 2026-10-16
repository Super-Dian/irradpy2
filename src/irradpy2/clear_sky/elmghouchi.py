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

    return Dayth


def elmghouchi(sza: np.ndarray, datetime: np.ndarray):
    """
    Original El Mghouchi clear-sky model.

    References
    ----------
    El Mghouchi, Y., El Bouardi, A., Choulli, Z., & Ajzoul, T. (2014).
    New model to estimate and evaluate the solar radiation.
    International Journal of Sustainable Built Environment, 3(2), 225-234.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth automatically.

    Returns
    -------
    [EghElMghouchi, EbnElMghouchi, EdhElMghouchi] : list of np.ndarray
        EghElMghouchi : Global horizontal irradiance [W m-2]
        EbnElMghouchi : Direct normal irradiance [W m-2]
        EdhElMghouchi : Diffuse horizontal irradiance [W m-2]
    """

    # -------------------------
    # Make safe local arrays
    # -------------------------
    sza = np.asarray(sza, dtype=float)
    Dayth = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    Eext = Esc * (1 + 0.034 * np.cos(Dayth - 2))

    # turbidity atmospheric factor for clear skies
    TFactor = 0.796 - 0.01 * np.sin(0.986 * (Dayth + 284))

    # direct normal irradiance
    EbnElMghouchi = Eext * TFactor * np.exp(-0.13 / np.cos(sza))

    # diffuse horizontal irradiance
    EdhElMghouchi = 120 * TFactor * np.exp(-1 / (0.4511 + np.cos(sza)))

    # global horizontal irradiance
    EghElMghouchi = EbnElMghouchi * np.cos(sza) + EdhElMghouchi

    # Quality control
    lower = 0
    EbnElMghouchi[EbnElMghouchi < lower] = 0
    EdhElMghouchi[EdhElMghouchi < lower] = 0
    EghElMghouchi[EghElMghouchi < lower] = 0

    return [EghElMghouchi, EbnElMghouchi, EdhElMghouchi]