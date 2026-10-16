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


def mac2(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, wv: np.ndarray):
    """
    Original MAC2 clear-sky model.

    References
    ----------
    Davies, J. A., & McKay, D. C. (1982). Estimating solar irradiance and components.
    Solar Energy, 29(1), 55-64.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    press : np.ndarray
        Local barometric pressure [mb]
    albedo : np.ndarray
        Surface/ground albedo
    ang_alpha : np.ndarray
        Angstrom exponent
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [Eghmac, Ebnmac, Edhmac] : list of np.ndarray
        Eghmac : Global horizontal irradiance [W m-2]
        Ebnmac : Direct normal irradiance [W m-2]
        Edhmac : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    amm = 35 / ((1224 * (np.cos(sza)) ** 2 + 1) ** 0.5)
    amm[amm < 0.0] = np.nan

    # ozone transmittance
    ozone = 0.35
    XO = amm * (ozone * 10)
    AO = ((0.1082 * XO) / ((1 + 13.86 * XO) ** 0.805)) + ((0.00658 * XO) / (1 + (10.36 * XO) ** 3)) + (0.002118 * XO / (1 + 0.0042 * XO + 3.23e-6 * XO ** 2))
    TO = 1 - AO

    # Rayleigh transmittance
    amms = np.array([0.5, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6, 10, 30], dtype=float)
    TRs = np.array([.9385, .8973, .8830, .8696, .8572, .8455, .8344, .7872, .7673, .7493, .7328, .7177, .7037, .6907, .6108, .4364], dtype=float)
    TR = np.interp(amm, amms, TRs)

    # aerosols transmittance
    tA = ang_beta * (0.38 ** (-ang_alpha) * 0.2758 + 0.5 ** (-ang_alpha) * 0.35)
    TA = np.exp(-tA * amm)

    # water vapor transmittance
    XW = amm * wv * 10 * (press / 1013.25) ** 0.75
    AW = 0.29 * XW / (((1 + 14.15 * XW) ** 0.635) + 0.5925 * XW)

    # forward scatter
    szajiao = sza * 180 / np.pi
    szajiaos = np.array([0, 25.8, 36.9, 45.6, 53.1, 60.0, 66.4, 72.5, 78.5, 90], dtype=float)
    fs = np.array([.92, .91, .89, .86, .83, .78, .71, .67, .60, .60], dtype=float)
    f = np.interp(szajiao, szajiaos, fs)

    # direct beam irradiance
    Ebnmac = Eext * (TO * TR - AW) * TA

    # diffuse components from Rayleigh scatter
    DR = Eext * np.cos(sza) * TO * (1 - TR) / 2

    # diffuse components from scattering by aerosol
    DA = Eext * np.cos(sza) * (TO * TR - AW) * (1 - TA) * 0.75 * f

    # diffuse horizontal irradiance
    Taaa = 0.95 ** 1.66
    poub = 0.0685 + (1 - Taaa) * 0.75 * (1 - 0.83)
    Edhmac = poub * albedo * (Ebnmac * np.cos(sza) + DR + DA) / (1 - poub * albedo) + DR + DA

    # global horizontal irradiance
    Eghmac = (Ebnmac * np.cos(sza) + DR + DA) / (1 - poub * albedo)

    # Quality control
    lower = 0
    Ebnmac[Ebnmac < lower] = 0
    Edhmac[Edhmac < lower] = 0
    Eghmac[Eghmac < lower] = 0

    return [Eghmac, Ebnmac, Edhmac]