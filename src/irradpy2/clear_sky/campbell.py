import numpy as np
import pandas as pd


def datetime_convert(datetime):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def campbell(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray):
    """
    Original Campbell clear-sky model.

    References
    ----------
    Campbell, G. S., & Norman, J. M. (1998).
    An introduction to environmental biophysics. An introduction to environmental biophysics., (Ed. 2).

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    press : np.ndarray
        Local barometric pressure [mb]

    Returns
    -------
    [EghCampbell, EbnCampbell, EdhCampbell] : list of np.ndarray
        EghCampbell : Global horizontal irradiance [W m-2]
        EbnCampbell : Direct normal irradiance [W m-2]
        EdhCampbell : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = press / 1013 * (np.cos(sza)) ** -1

    # direct normal irradiance
    tau0 = 0.7
    EbnCampbell = Eext * tau0 ** am

    # diffuse horizontal irradiance
    EdhCampbell = 0.3 * Eext * np.cos(sza) * (1 - tau0 ** am)

    # global horizontal irradiance
    EghCampbell = EbnCampbell * np.cos(sza) + EdhCampbell

    # Quality control
    lower = 0
    EbnCampbell[EbnCampbell < lower] = 0
    EdhCampbell[EdhCampbell < lower] = 0
    EghCampbell[EghCampbell < lower] = 0

    return [EghCampbell, EbnCampbell, EdhCampbell]