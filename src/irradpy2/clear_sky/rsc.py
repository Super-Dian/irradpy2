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


def rsc(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, ang_beta: np.ndarray, wv: np.ndarray):
    """
    Original RSC clear-sky model.

    References
    ----------
    Carroll, J. J. (1985). Global transmissivity and diffuse fraction of solar radiation for clear and cloudy skies as measured and as predicted by bulk transmissivity models.
    Solar Energy, 35(2), 105-118.
    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
    in Romania. Renewable Energy, 55, 85-103.

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
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghRSC, EbnRSC, EdhRSC] : list of np.ndarray
        EghRSC : Global horizontal irradiance [W m-2]
        EbnRSC : Direct normal irradiance [W m-2]
        EdhRSC : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1371
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    ama = am * press / 1013.25

    # Rayleigh transmittance
    gamma = 0.054 - 0.0088 * ama + 1.08e-3 * ama ** 2 - 5.1e-5 * ama ** 3
    TR = 10 ** (-ama * gamma)

    # aerosol and water vapour transmittance
    Ta = 10 ** (-0.02 * ama)
    Tp = 10 ** (-0.666 * ang_beta * ama)
    Tw = 10 ** (-ama * (0.04 * wv ** 0.1 + 0.01 * wv))

    # direct normal irradiance
    EbnRSC = Eext * Ta * Tw * Tp * TR
    Eb = EbnRSC * np.cos(sza)

    # diffuse horizontal irradiance
    Ew = Eext * np.cos(sza) * Ta * Tw
    Ef = (0.5 + 0.3 * ang_beta) * (np.cos(sza) ** (1 / 3)) * (Ew - Eb)
    f3 = (0.98 + 0.1 * albedo + 0.36 * ang_beta * (albedo - 0.25)) * (Eb + Ef) - (Eb + Ef)
    EdhRSC = Ef + f3

    # global horizontal irradiance
    EghRSC = EbnRSC * np.cos(sza) + EdhRSC

    # Quality control
    lower = 0
    EbnRSC[EbnRSC < lower] = 0
    EdhRSC[EdhRSC < lower] = 0
    EghRSC[EghRSC < lower] = 0

    return [EghRSC, EbnRSC, EdhRSC]