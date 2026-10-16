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


def bird(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, albedo: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original Bird clear-sky model.

    References
    ----------
    Bird, R. E., & Hulstrom, R. L. (1981). Simplified clear sky model for direct and diffuse insolation on horizontal surfaces
    (No. SERI/TR-642-761). Solar Energy Research Inst., Golden, CO (USA).

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    press : np.ndarray
        Local barometric pressure [mb]
    albedo : np.ndarray
        Surface/ground albedo
    ang_alpha : np.ndarray
        Angstrom exponent
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [Eghbird, Ebnbird, Edhbird] : list of np.ndarray
        Eghbird : Global horizontal irradiance [W m-2]
        Ebnbird : Direct normal irradiance [W m-2]
        Edhbird : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eex = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** -1.25) ** -1
    ama = am * press / 1013.25

    # transmittance of Rayleigh scattering
    TR = np.exp(-0.0903 * (ama ** 0.84) * (1 + ama - ama ** 1.01))

    # transmittance of ozone absorptance
    XO = ozone * am
    TO = 1 - 0.1611 * XO * (1 + 139.48 * XO) ** (-0.3035) - 0.002715 * XO * (1 + 0.044 * XO + 0.0003 * XO ** 2) ** (-1)

    # transmittance of absorptance by uniformly mixed gases
    TUM = np.exp(-0.0127 * ama ** 0.26)

    # transmittance of water vapor absorptance
    XW = wv * am
    TW = 1 - 2.4959 * XW * ((1 + 79.034 * XW) ** 0.6828 + 6.385 * XW) ** (-1)

    # transmittance of aerosol absorptance and scattering
    broadaod = ang_beta * (0.2758 * 0.38 ** (-ang_alpha) + 0.35 * 0.5 ** (-ang_alpha))
    TA = np.exp((-(broadaod ** 0.873)) * (1 + broadaod - broadaod ** 0.7088) * am ** 0.9108)

    # direct beam irradiance
    Ebnbird = Eex * 0.9662 * TA * TW * TUM * TO * TR

    # transmittance of aerosol absorptance
    K1 = 0.1
    TAA = 1 - K1 * (1 - am + am ** 1.06) * (1 - TA)

    # transmittance of aerosol scattering
    TAS = TA / TAA

    BA = 0.84
    Eas = Eex * np.cos(sza) * 0.79 * TO * TUM * TW * TAA * (0.5 * (1 - TR) + BA * (1 - TAS)) / (1 - am + am ** 1.02)

    # sky and ground albedo
    rs = 0.0685 + (1 - BA) * (1 - TAS)
    rs[rs < 0] = 0
    rs[rs > 1] = 1
    rg = albedo

    # global horizontal irradiance
    Eghbird = (Ebnbird * np.cos(sza) + Eas) / (1 - rg * rs)

    # diffuse horizontal irradiance
    Edhbird = Eghbird - Ebnbird * np.cos(sza)

    # Quality control
    lower = 0
    Ebnbird[Ebnbird < lower] = 0
    Edhbird[Edhbird < lower] = 0
    Eghbird[Eghbird < lower] = 0

    return [Eghbird, Ebnbird, Edhbird]