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


def iqbal_a(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, ozone: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, albedo: np.ndarray):
    """
    Iqbal-A 1983 clear-sky model.

    References
    ----------
    Iqbal, M. (1983).
    An Introduction to Solar Radiation.
    Chapter 7.5, Model A.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input. It is used to calculate Dayth and Year automatically.
    pressure : np.ndarray
        Local barometric pressure [mb]
    wv : np.ndarray
        Total precipitable water vapour [cm]
    ozone : np.ndarray
        Total ozone amount [atm-cm]
    ang_alpha : np.ndarray
        Angstrom exponent
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    albedo : np.ndarray
        Ground albedo

    Returns
    -------
    [EghIQBALA, EbnIQBALA, EdhIQBALA] : list of np.ndarray
        EghIQBALA : Global horizontal irradiance [W m-2]
        EbnIQBALA : Direct normal irradiance [W m-2]
        EdhIQBALA : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    sza_deg = sza / np.pi * 180
    am = (np.cos(sza) + 0.15 * (93.885 - sza_deg) ** (-1.253)) ** (-1)
    ama = am * pressure / 1013.25

    # absorption ratio for water vapour
    U1 = am * wv * (pressure / 1013.25) ** 0.75
    aw = 2.9 * U1 / ((1 + 141.5 * U1) ** 0.635 + 5.925 * U1)

    # absorption ratio for ozone
    U3 = ozone * am
    ao = 0.02118 * U3 / (1 + 0.042 * U3 + 3.23e-4 * U3 ** 2) + 1.082 * U3 / ((1 + 138.6 * U3) ** 0.805) + 0.0658 * U3 / (1 + (103.6 * U3) ** 3)
    To = 1 - ao

    # transmittance due to Rayleigh scattering
    Tr = 0.972 - 0.08262 * ama + 0.00933 * ama ** 2 - 0.00095 * ama ** 3 + 0.000437 * ama ** 4

    # transmittance due to aerosol attenuation
    Ta = 0.12445 * ang_alpha - 0.0162 + (1.003 - 0.125 * ang_alpha) * np.exp(-ang_beta * ama * (1.089 * ang_alpha + 0.5123))

    # direct normal irradiance
    EbnIQBALA = Eext * (To * Tr - aw) * Ta

    # diffuse irradiance from the Rayleigh atmosphere
    Idr = Eext * np.cos(sza) * To * Ta * 0.5 * (1 - Tr)

    # diffuse irradiance from aerosol scattering
    Ida = Eext * np.cos(sza) * (To * Tr - aw) * 0.75 * (1 - Ta)

    # diffuse irradiance from multiple reflections between ground and atmosphere
    Q = EbnIQBALA * np.cos(sza) + Idr + Ida
    Ta166 = 0.12445 * ang_alpha - 0.0162 + (1.003 - 0.125 * ang_alpha) * np.exp(-ang_beta * (1.66 * pressure / 1013.25) * (1.089 * ang_alpha + 0.5123))
    pouA = 0.0685 + 0.17 * (1 - Ta166) * 0.95
    Idm = Q * albedo * pouA / (1 - albedo * pouA)

    # diffuse horizontal irradiance
    EdhIQBALA = Idr + Ida + Idm

    # global horizontal irradiance
    EghIQBALA = EbnIQBALA * np.cos(sza) + EdhIQBALA

    # Quality control
    lower = 0
    EbnIQBALA[EbnIQBALA < lower] = lower
    EbnIQBALA[np.imag(EbnIQBALA) != 0] = np.nan
    EdhIQBALA[EdhIQBALA < lower] = lower
    EdhIQBALA[np.imag(EdhIQBALA) != 0] = np.nan
    EghIQBALA[EghIQBALA < lower] = lower
    EghIQBALA[np.imag(EghIQBALA) != 0] = np.nan

    return [EghIQBALA, EbnIQBALA, EdhIQBALA]