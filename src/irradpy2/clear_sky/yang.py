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


def yang(sza: np.ndarray, datetime, press: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original Yang clear-sky model.

    References
    ----------
    Yang, K., & Koike, T. (2005). A general model to estimate hourly and daily solar radiation for hydrological studies.
    Water Resources Research, 41(10).

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
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
    [EghYang, EbnYang, EdhYang] : list of np.ndarray
        EghYang : Global horizontal irradiance [W m-2]
        EbnYang : Direct normal irradiance [W m-2]
        EdhYang : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1361.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = 1 / (np.cos(sza) + 0.15 * (57.296 * (np.pi / 2 - sza) / np.pi * 180 + 3.885) ** -1.253)
    ama = am * press / 1013.25

    # the broadband radiative transmittance due to permanent gas absorption
    Tg = np.exp(-0.0117 * ama ** 0.3139)

    # the broadband radiative transmittance due to Rayleigh scattering
    Tr = np.exp(-0.008735 * ama * (0.547 + 0.014 * ama - 0.00038 * ama ** 2 + 4.6e-6 * ama ** 3) ** -4.08)

    # the broadband radiative transmittance due to water vapor absorption
    Tw = np.minimum(1, (0.909 - 0.036 * np.log(wv * am)))

    # the broadband radiative transmittance due to ozone absorption
    Toz = np.exp(-0.0365 * (am * ozone) ** 0.7136)

    # the broadband radiative transmittance due to aerosol extinction
    Ta = np.exp(-am * ang_beta * (0.6777 + 0.1464 * (am * ang_beta) - 0.00626 * (am * ang_beta) ** 2) ** -1.3)

    # the broadband solar beam radiative transmittance
    Tbclear = np.maximum(0, Toz * Tw * Tg * Tr * Ta - 0.013)

    # the broadband solar diffuse radiative transmittance
    Tdclear = 0.5 * (Toz * Tg * Tw * (1 - Ta * Tr) + 0.013)

    # direct beam irradiance
    EbnYang = Eext * Tbclear

    # diffuse horizontal irradiance
    EdhYang = Eext * np.cos(sza) * Tdclear

    # global horizontal irradiance
    EghYang = EbnYang * np.cos(sza) + EdhYang

    # Quality control
    lower = 0
    EbnYang[EbnYang < lower] = 0
    EdhYang[EdhYang < lower] = 0
    EghYang[EghYang < lower] = 0

    return [EghYang, EbnYang, EdhYang]