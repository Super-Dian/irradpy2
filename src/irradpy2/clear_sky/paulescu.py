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


def paulescu(sza: np.ndarray, datetime, altitude: np.ndarray, press: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original Paulescu & Schlett clear-sky model.

    References
    ----------
    Paulescu, M., & Schlett, Z. (2003). A simplified but accurate spectral solar irradiance model.
    Theoretical and Applied Climatology, 75(3-4), 203-212.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    altitude : np.ndarray
        Location altitude [m]
    press : np.ndarray
        Local barometric pressure [mb]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghPaulescu, EbnPaulescu, EdhPaulescu] : list of np.ndarray
        EghPaulescu : Global horizontal irradiance [W m-2]
        EbnPaulescu : Direct normal irradiance [W m-2]
        EdhPaulescu : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    altitude = np.asarray(altitude, dtype=float)
    press = np.asarray(press, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1369
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = (1 - altitude * 10 ** -4) / (np.cos(sza) + 0.50572 * (((np.pi / 2 - sza) / np.pi * 180) + 6.07995) ** (-1.6364))

    # ozone
    To3 = np.exp(-am * ozone * (0.0184 - 0.0004 * am * ozone + 0.022 * (am * ozone) ** (-0.66)))

    # water vapour
    emw = am * wv
    Twv = np.exp(-emw * (-0.002 + 1.67e-5 * emw + 0.094 * emw ** (-0.693)))

    # trace gas absorption
    Tg = np.exp(-am * (-5.4e-5 - 3.8e-6 * am + 0.0099 * am ** (-0.62)))

    # Rayleigh scattering
    emp = am * press / 1013.25
    Tr = np.exp(-emp * (0.709 + 0.0013 * emp - 0.5856 * emp ** 0.058))

    # aerosol extinction
    emb = am * ang_beta
    Ta = np.exp(-emb * (1.053 - 0.083 * emb + 0.3345 * emb ** (-0.668)))
    Ta[Ta > 1] = 1

    # direct normal irradiance
    EbnPaulescu = Eext * Tr * Ta * Twv * To3 * Tg

    # diffuse horizontal irradiance
    gama = 0.432
    EdhPaulescu = gama * Eext * np.cos(sza) * (1 - Tr * Ta) * Twv * To3 * Tg

    # global horizontal irradiance
    EghPaulescu = EbnPaulescu * np.cos(sza) + EdhPaulescu

    # Quality control
    lower = 0
    EbnPaulescu[EbnPaulescu < lower] = 0
    EdhPaulescu[EdhPaulescu < lower] = 0
    EghPaulescu[EghPaulescu < lower] = 0

    return [EghPaulescu, EbnPaulescu, EdhPaulescu]