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


def mrmv6(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, aod550: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original MRMv6.1 clear-sky model.

    References
    ----------
    Kambezidis, H. D., & Psiloglou, B. E. (2008). The meteorological radiation model (MRM): advancements and applications.
    In Modeling solar radiation at the earth’s surface (pp. 357-392). Springer, Berlin, Heidelberg.
    Kambezidis, H. D., Psiloglou, B. E., Karagiannis, D., Dumka, U. C., & Kaskaoutis, D. G. (2016).
    Recent improvements of the Meteorological Radiation Model for solar irradiance estimates under all-sky conditions.
    Renewable Energy, 93, 142-158.
    Kambezidis, H. D., Psiloglou, B. E., Karagiannis, D., Dumka, U. C., & Kaskaoutis, D. G. (2017).
    Meteorological Radiation Model (MRM v6.1): Improvements in diffuse radiation estimates and a new approach for implementation of cloud products.
    Renewable and Sustainable Energy Reviews, 74, 616-637.
    Author's Fortran code.

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
    aod550 : np.ndarray
        Aerosol optical depth at 550 nm [dimensionless]
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghMRM61, EbnMRM61, EdhMRM61] : list of np.ndarray
        EghMRM61 : Global horizontal irradiance [W m-2]
        EbnMRM61 : Direct normal irradiance [W m-2]
        EdhMRM61 : Diffuse horizontal irradiance [W m-2]

    Notes
    -----
    This model requires the external file 'AEROSOL-TABLE.txt' in the current working directory.
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    aod550 = np.asarray(aod550, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1360.8
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eex = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    m = (np.cos(sza) + 0.50572 * (((np.pi / 2 - sza) / np.pi * 180) + 6.07995) ** -1.6364) ** -1
    ma = m * press / 1013.25

    # broadband transmission function for water vapour
    Tw = 1 - (3.0140 * m * wv / ((1 + 119.3 * m * wv) ** 0.644 + 5.814 * m * wv))

    # broadband transmission function for Rayleigh scattering
    Tr = np.exp(-(0.1128 * ma ** 0.8346 * (0.9341 - ma ** 0.9868 + 0.9391 * ma)))

    # broadband transmission function for absorption by ozone
    To = 1 - (0.2554 * m * ozone / ((1 + 6107.26 * m * ozone) ** 0.204 + 0.471 * m * ozone))

    # broadband transmission function for absorption by uniformly mixed gases
    Tco2 = 1 - (0.07210 * ma * 350 / ((1 + 377.890 * ma * 350) ** 0.5855 + 3.1709 * ma * 350))
    Tco = 1 - (0.0062 * ma * 0.075 / ((1 + 243.670 * ma * 0.075) ** 0.4246 + 1.7222 * ma * 0.075))
    Tn2o = 1 - (0.0326 * ma * 0.28 / ((1 + 107.413 * ma * 0.28) ** 0.5501 + 0.9093 * ma * 0.28))
    Tch4 = 1 - (0.0192 * ma * 1.6 / ((1 + 166.095 * ma * 1.6) ** 0.4221 + 0.7186 * ma * 1.6))
    To2 = 1 - (0.0003 * ma * 2.095e5 / ((1 + 476.934 * ma * 2.095e5) ** 0.4892 + 0.1261 * ma * 2.095e5))
    Tmg = Tco2 * Tco * Tn2o * Tch4 * To2

    # broadband transmission function for aerosol total extinction
    aerosol_table = np.loadtxt("AEROSOL-TABLE.txt", delimiter=",")
    wavelength = aerosol_table[:, 0]
    ET = aerosol_table[:, 1]
    AODtable = aerosol_table[:, 2:6]
    SSAtable = aerosol_table[:, 6:10]

    k = aod550 - aod550 + 4
    k[aod550 >= 0.3] = 1
    k[(aod550 >= 0.2) & (aod550 < 0.3)] = 2
    k[(aod550 <= 0.1) & (aod550 < 0.2)] = 3

    AODt = aod550 - aod550
    SSA = aod550 - aod550
    for i in range(len(k)):
        AODt[i] = AODtable[391, int(k[i]) - 1]

    RAOD = aod550 / AODt
    end = len(ET)
    sumetl = np.sum(((ET[1:end] + ET[0:end - 1]) / 2) * (wavelength[1:end] - wavelength[0:end - 1]))
    sumletl = np.array([0.0] * len(aod550), dtype=float)
    sumtletl = np.array([0.0] * len(aod550), dtype=float)

    for t in range(len(aod550)):
        sumletl[t] = np.sum(((np.exp(-ma[t] * RAOD[t] * AODtable[1:end, int(k[t]) - 1]) * ET[1:end]) + (np.exp(-ma[t] * RAOD[t] * AODtable[0:end - 1, int(k[t]) - 1]) * ET[0:end - 1])) / 2 * (wavelength[1:end] - wavelength[0:end - 1]))
        sumtletl[t] = np.sum(((np.exp(-ma[t] * RAOD[t] * AODtable[1:end, int(k[t]) - 1] * SSAtable[1:end, int(k[t]) - 1]) * ET[1:end]) + (np.exp(-ma[t] * RAOD[t] * AODtable[0:end - 1, int(k[t]) - 1] * SSAtable[0:end - 1, int(k[t]) - 1]) * ET[0:end - 1])) / 2 * (wavelength[1:end] - wavelength[0:end - 1]))

    sumletl[0] = sumetl
    sumtletl[0] = sumetl

    Taer = sumletl / sumetl
    Tas = sumtletl / sumetl
    Taa = Taer / Tas

    # direct horizontal/beam irradiance
    Ebh = Eex * np.cos(sza) * Tw * Tr * To * Tmg * Taer
    EbnMRM61 = Eex * Tw * Tr * To * Tmg * Taer

    # calculation of diffuse and global radiation from Fortran
    Tc = 1
    Ra = 0.0685 + 0.26 * (1 - Taer) + 0.4 * (1 - Tc)
    Rsa = Ra * albedo
    Ebhc = Ebh * Tc

    # calculation of Mie aerosol forward scattering percentage
    fa = np.abs(-8e-5 * (np.pi / 2 - sza) ** 2 + 0.0117 * (np.pi / 2 - sza) + 0.5)
    HDR3 = Eex * np.cos(sza) * Tw * To * Tmg * Taa * (np.sqrt(0.5 * fa) * (1.0 - Tas * Tr))
    HDR3c = (HDR3 * Tc) + (0.32 * (1.0 - Tc) * (Ebh + HDR3))
    HDA3c = (Ebhc + HDR3c) * (Rsa / (1.0 - Rsa))

    # diffuse horizontal irradiance
    EdhMRM61 = HDR3c + HDA3c

    # global horizontal irradiance
    EghMRM61 = EbnMRM61 * np.cos(sza) + EdhMRM61

    # Quality control
    lower = 0
    EbnMRM61[EbnMRM61 < lower] = 0
    EdhMRM61[EdhMRM61 < lower] = 0
    EghMRM61[EghMRM61 < lower] = 0

    return [EghMRM61, EbnMRM61, EdhMRM61]