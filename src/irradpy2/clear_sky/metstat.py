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


def metstat(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, broadbandaod: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original METSTAT clear-sky model.

    References
    ----------
    Maxwell, E. L. (1998). METSTAT—The solar radiation model used in the production of the National Solar Radiation Data Base (NSRDB).
    Solar Energy, 62(4), 263-279.
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
    broadbandaod : np.ndarray
        Broadband aerosol optical depth [dimensionless]
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghMetstat, EbnMetstat, EdhMetstat] : list of np.ndarray
        EghMetstat : Global horizontal irradiance [W m-2]
        EbnMetstat : Direct normal irradiance [W m-2]
        EdhMetstat : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    broadbandaod = np.asarray(broadbandaod, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    ama = am * press / 1013.25

    # water vapour transmittance
    xw = wv * am
    Tw = 1 - 1.668 * xw / (4.042 * xw + (1 + 54.6 * xw) ** 0.637)

    # Rayleigh scattering transmittance
    TR = np.exp(-0.0903 * (ama ** 0.84) * (1 + ama - ama ** 1.01))

    # ozone absorption transmittance
    XO = ozone * am
    To = 1 - 0.1611 * XO * (1 + 139.48 * XO) ** (-0.3035) - 0.002715 * XO * (1 + 0.044 * XO + 0.0003 * XO ** 2) ** (-1)

    # transmittance from absorption by uniformly mixed gases
    Tum = np.exp(-0.0127 * ama ** 0.26)

    # total aerosol transmittance
    TA = np.exp(-broadbandaod * am)

    # direct normal irradiance
    Kn = 0.9751 * TR * To * Tum * Tw * TA
    EbnMetstat = Eext * Kn

    # diffuse and global transmittance terms
    omega = 0.9
    Ba = 0.84
    Taa = 1 - (1 - omega) * (1 - am + am ** 1.06) * (1 - TA)
    Tas = TA / Taa
    Tabs = To * Tum * Taa
    Tsr = 0.5 * (1 - TR) * Tabs
    Tsa = Ba * (1 - TA) * Tabs
    Fam = 0.38 + 0.925 * np.exp(-0.851 * am)
    rhos = 0.0685 + (1 - Ba) * (1 - Tas)
    Ts0 = (Tsr + Tsa) * Fam
    Tsg = (Kn + Ts0) * (albedo - 0.2) * rhos
    Td = Ts0 + Tsg
    XKt = Kn + Td

    # diffuse horizontal irradiance
    EdhMetstat = Td * Eext * np.cos(sza)

    # global horizontal irradiance
    EghMetstat = XKt * Eext * np.cos(sza)

    # Quality control
    lower = 0
    EbnMetstat[EbnMetstat < lower] = 0
    EdhMetstat[EdhMetstat < lower] = 0
    EghMetstat[EghMetstat < lower] = 0

    return [EghMetstat, EbnMetstat, EdhMetstat]