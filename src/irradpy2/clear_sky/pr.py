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


def pr(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original PR clear-sky model.

    References
    ----------
    Gueymard, C. A. (2003). Direct solar transmittance and irradiance predictions with broadband models. Part I: detailed theoretical performance assessment.
    Solar Energy, 74(5), 355-379.
    Psiloglou, B. E., Santamouris, M., & Asimakopoulos, D. N. (2000). Atmospheric broadband model for computation of solar radiation at the Earth’s surface.
    Application to Mediterranean climate. Pure and Applied Geophysics, 157(5), 829-860.
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
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghPR, EbnPR, EdhPR] : list of np.ndarray
        EghPR : Global horizontal irradiance [W m-2]
        EbnPR : Direct normal irradiance [W m-2]
        EdhPR : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1373
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    amky = (np.cos(sza) + 0.50572 * (((np.pi / 2 - sza) / np.pi * 180) + 6.07995) ** -1.6364) ** -1
    amkyp = amky * press / 1013.25

    # transmittances from Psiloglou2000
    Trr = np.exp(-0.1128 * (amkyp ** 0.8346) * (0.9341 - amkyp ** 0.9868 + 0.9391 * amkyp))
    ako3 = amky * ozone
    Tro = 1 - 0.2554 * ako3 / ((1 + 6107.26 * ako3) ** 0.204 + 0.471 * ako3)
    akco2 = amkyp * 330
    TrCO2 = 1 - 0.0721 * akco2 / ((1 + 377.89 * akco2) ** 0.5855 + 3.1709 * akco2)
    akco = amkyp * 0.075
    TrCO = 1 - 0.0062 * akco / ((1 + 243.67 * akco) ** 0.4246 + 1.7222 * akco)
    akch4 = amkyp * 1.6
    TrCH4 = 1 - 0.0192 * akch4 / ((1 + 166.095 * akch4) ** 0.4221 + 0.7186 * akch4)
    akn2o = amkyp * 0.28
    TrN2O = 1 - 0.0326 * akn2o / ((1 + 107.413 * akn2o) ** 0.5501 + 0.9093 * akn2o)
    ako2 = amkyp * 2.095e5
    TrO2 = 1 - 0.0003 * ako2 / ((1 + 476.934 * ako2) ** 0.4892 + 0.1261 * ako2)
    Trmg = TrCO2 * TrCO * TrCH4 * TrN2O * TrO2
    akw = amky * wv
    Trw = 1 - 3.014 * akw / ((1 + 119.3 * akw) ** 0.644 + 5.814 * akw)

    # aerosol transmittance from REST
    Masa = amky
    beta2 = ang_beta * ang_beta
    b0 = 1.6933
    bx = 1 + 0.42003 * ang_beta
    b1 = (-0.013029 + 0.13126 * ang_beta) / bx
    b2 = (-0.0083581 + 0.40323 * ang_beta + 0.123 * beta2) / bx
    TauA = ang_beta * (b0 + Masa * b1) / (1 + b2 * Masa)
    Tra = np.exp(-Masa * TauA)

    # direct normal radiation
    EbnPR = Eext * Trr * Trmg * Tro * Trw * Tra
    Eb = EbnPR * np.cos(sza)

    # diffuse horizontal radiation
    Traa = 1 - 1.405e-3 * amky - 9.013e-5 * amky ** 2 + 2.2e-6 * amky ** 3
    Tras = Tra / Traa
    ros = 0.08503
    Ed1 = Eext * np.cos(sza) * Trmg * Tro * Trw * Traa * (1 - Tras * Trr) / 2
    Ed2 = (Ed1 + Eb) * albedo * ros / (1 - albedo * ros)
    EdhPR = Ed1 + Ed2

    # global horizontal irradiance
    EghPR = EbnPR * np.cos(sza) + EdhPR

    # Quality control
    lower = 0
    EbnPR[EbnPR < lower] = 0
    EdhPR[EdhPR < lower] = 0
    EghPR[EghPR < lower] = 0

    return [EghPR, EbnPR, EdhPR]