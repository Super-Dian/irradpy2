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


def csr(sza: np.ndarray, datetime, pressure: np.ndarray, broadbandaod: np.ndarray, wv: np.ndarray, ozone: np.ndarray, albedo: np.ndarray):
    """
    CSR 2012 clear-sky model.

    References
    ----------
    Maxwell, E. L. (1998).
    METSTAT—The solar radiation model used in the production of the
    National Solar Radiation Data Base.

    Gueymard, C. A. (2012).
    Clear-sky irradiance predictions for solar resource mapping and
    large-scale applications: Improved validation methodology and
    detailed performance analysis of 18 broadband radiative models.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    pressure : np.ndarray
        Local barometric pressure [mb]
    broadbandaod : np.ndarray
        Broadband aerosol optical depth
    wv : np.ndarray
        Total precipitable water vapour [cm]
    ozone : np.ndarray
        Total ozone amount [atm-cm]
    albedo : np.ndarray
        Ground albedo

    Returns
    -------
    [EghCSR, EbnCSR, EdhCSR] : list of np.ndarray
        EghCSR : Global horizontal irradiance [W m-2]
        EbnCSR : Direct normal irradiance [W m-2]
        EdhCSR : Diffuse horizontal irradiance [W m-2]

    Notes
    -----
    The only difference between CSR and METSTAT is the C_coeff term
    used in the direct normal irradiance calculation.
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    pressure = np.asarray(pressure, dtype=float)
    broadbandaod = np.asarray(broadbandaod, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1369
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    sza_deg = sza / np.pi * 180
    am = (np.cos(sza) + 0.15 * (93.885 - sza_deg) ** (-1.253)) ** (-1)
    ama = am * pressure / 1013.25

    # water vapour transmittance
    xw = wv * am
    Tw = 1 - 1.668 * xw / (4.042 * xw + (1 + 54.6 * xw) ** 0.637)

    # Rayleigh scattering transmittance
    TR = np.exp(-0.0903 * ama ** 0.84 * (1 + ama - ama ** 1.01))

    # ozone absorption transmittance
    XO = ozone * am
    To = 1 - 0.1611 * XO * (1 + 139.48 * XO) ** (-0.3035) - 0.002715 * XO * (1 + 0.044 * XO + 0.0003 * XO ** 2) ** (-1)

    # transmittance from absorption by uniformly mixed gases
    Tum = np.exp(-0.0127 * ama ** 0.26)

    # total aerosol transmittance
    TA = np.exp(-broadbandaod * am)

    # direct normal irradiance
    C_coeff = (50 + np.abs(np.cos(Dayth / 365))) / 49.25
    Kn = C_coeff * TR * To * Tum * Tw * TA
    EbnCSR = Eext * Kn

    # diffuse horizontal irradiance
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
    EdhCSR = Eext * Td * np.cos(sza)

    # global horizontal irradiance
    XKt = Kn + Td
    EghCSR = Eext * XKt * np.cos(sza)

    # Quality control
    lower = 0
    EbnCSR[EbnCSR < lower] = lower
    EdhCSR[EdhCSR < lower] = lower
    EghCSR[EghCSR < lower] = lower

    return [EghCSR, EbnCSR, EdhCSR]