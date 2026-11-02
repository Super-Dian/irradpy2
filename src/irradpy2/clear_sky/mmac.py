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


def mmac(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, broadbandaod: np.ndarray, wv: np.ndarray):
    """
    Original MMAC clear-sky model.

    References
    ----------
    Gueymard, C. (1993). Critical analysis and performance assessment of clear sky solar irradiance models using theoretical and measured data.
    Solar Energy, 51(2), 121-138.
    Gueymard, C. A. (2003). Direct solar transmittance and irradiance predictions with broadband models. Part I: detailed theoretical performance assessment.
    Solar Energy, 74(5), 355-379.
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
    broadbandaod : np.ndarray
        Broadband aerosol optical depth [dimensionless]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [Eghmmac, Ebnmmac, Edhmmac] : list of np.ndarray
        Eghmmac : Global horizontal irradiance [W m-2]
        Ebnmmac : Direct normal irradiance [W m-2]
        Edhmmac : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    broadbandaod = np.asarray(broadbandaod, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    m = 35 / ((1224 * (np.cos(sza)) ** 2 + 1) ** 0.5)
    m[m < 0.0] = np.nan
    ma = m * press / 1013.25

    # aerosol transmittance
    Ta = np.exp(-ma * broadbandaod)

    # Rayleigh transmittance
    TR = 1 / (1 + 1 / np.exp(2.12182 - 0.791532 * np.log(ma) + 0.024761 * (np.log(ma)) ** 2))

    # ozone transmittance
    XO = m * (0.35 * 10)
    AO = ((0.1082 * XO) / ((1 + 13.86 * XO) ** 0.805)) + ((0.00658 * XO) / (1 + (10.36 * XO) ** 3)) + (0.002118 * XO / (1 + 0.0042 * XO + 3.23e-6 * XO ** 2))
    TO = 1 - AO

    # water vapor transmittance
    XW = 10 * m * wv * (press / 1013.25) ** 0.75
    AW = 0.29 * XW / ((1 + 14.15 * XW) ** 0.635 + 0.5925 * XW)

    # direct beam irradiance
    Ebnmmac = Eext * (TO * TR - AW) * Ta

    # forward aerosol scatterance
    Ba = 0.93 - 0.21 * np.log(m)

    # aerosol single-scattering albedo
    wa = 0.98
    Taaa = 0.95 ** 1.66
    Baa = 0.93 - 0.21 * np.log(1.66)

    # diffuse components from scattering by aerosol
    Ida = Eext * np.cos(sza) * wa * Ba * (TO * TR - AW) * (1 - Ta)

    # diffuse components from Rayleigh scatter
    Idr = Eext * np.cos(sza) * (0.5 * TO * Taaa * (1 - TR))

    # diffuse horizontal irradiance without albedos
    Edh = Idr + Ida

    # diffuse horizontal irradiance with albedos
    poub = 0.0685 + (1 - Taaa) * wa * (1 - Baa)
    Edhmmac = poub * albedo * (Ebnmmac * np.cos(sza) + Edh) / (1 - poub * albedo) + Edh

    # global horizontal irradiance
    Eghmmac = (Ebnmmac * np.cos(sza) + Edh) / (1 - poub * albedo)

    # Quality control
    lower = 0
    Ebnmmac[Ebnmmac < lower] = 0
    Edhmmac[Edhmmac < lower] = 0
    Eghmmac[Eghmmac < lower] = 0

    return [Eghmmac, Ebnmmac, Edhmmac]