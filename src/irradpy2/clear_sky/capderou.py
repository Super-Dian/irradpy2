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


def capderou(sza: np.ndarray, datetime: np.ndarray, latitude: np.ndarray, altitude: np.ndarray, press: np.ndarray):
    """
    Original Capderou clear-sky model.

    References
    ----------
    Capderou, M. (1987). Theoretical and experimental models solar atlas of Algeria (in French) Tome 1 and 2.
    Algeria: University Publications Office.

    Marif, Y., Chiba, Y., Belhadj, M. M., Zerrouki, M., & Benhammou, M. (2018).
    A clear sky irradiation assessment using a modified Algerian solar atlas model in Adrar city.
    Energy Reports, 4, 84-90.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    latitude : np.ndarray
        Location latitude [-90, 90]
    altitude : np.ndarray
        Location altitude [m]
    press : np.ndarray
        Local barometric pressure [mb]

    Returns
    -------
    [EghCapderou, EbnCapderou, EdhCapderou] : list of np.ndarray
        EghCapderou : Global horizontal irradiance [W m-2]
        EbnCapderou : Direct normal irradiance [W m-2]
        EdhCapderou : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    altitude = np.asarray(altitude, dtype=float)
    press = np.asarray(press, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    mA = press / 1013.25 * (np.cos(sza) + 0.15 * ((3.885 + (np.pi / 2 - sza) * 180 / np.pi) ** (-1.253))) ** (-1)
    mAc = (np.cos(sza) + 0.15 * ((3.885 + (np.pi / 2 - sza) * 180 / np.pi) ** (-1.253))) ** (-1)

    Ahe = np.sin(360 / 365 * (Dayth - 121))

    # the atmospheric turbidity caused by water vapor absorption
    T0 = 2.4 - 0.9 * np.sin(latitude) + 0.1 * Ahe * (2 + np.sin(latitude)) - 0.2 * altitude / 1000 - (1.22 + 0.14 * Ahe) * (1 - np.cos(sza))

    # the atmospheric turbidity corresponding to the molecular diffusion
    T1 = 0.89 ** (altitude / 1000)

    # the atmospheric turbidity relative to the aerosol diffusion coupled with a slight absorption
    T2 = (0.9 + 0.4 * Ahe) * 0.63 ** (altitude / 1000)

    # the atmospheric Linke turbidity factor
    Tlc = T0 + T1 + T2

    delta_rk = 1 / (9.4 + 0.9 * mA)

    # direct normal irradiance
    EbnCapderou = Eext * np.exp(-Tlc * mAc * delta_rk)

    # diffuse horizontal irradiance
    a = 1.1
    b = np.log(T1 + T2) - 2.8 + 1.02 * (1 - np.cos(sza)) ** 2
    EdhCapderou = Eext * np.exp(-1 + 1.06 * np.log(np.cos(sza)) + a - np.sqrt(a ** 2 + b ** 2))

    # global horizontal irradiance
    EghCapderou = EbnCapderou * np.cos(sza) + EdhCapderou

    # Quality control
    lower = 0
    EbnCapderou[EbnCapderou < lower] = 0
    EdhCapderou[EdhCapderou < lower] = 0
    EghCapderou[EghCapderou < lower] = 0

    return [EghCapderou, EbnCapderou, EdhCapderou]