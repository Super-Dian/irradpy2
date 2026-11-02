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


def mrmv5(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original MRMv5 clear-sky model.

    References
    ----------
    Kambezidis, H. D., & Psiloglou, B. E. (2008). The meteorological radiation model (MRM): advancements and applications.
    In Modeling solar radiation at the earth’s surface (pp. 357-392). Springer, Berlin, Heidelberg.

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
    [EghMRM5, EbnMRM5, EdhMRM5] : list of np.ndarray
        EghMRM5 : Global horizontal irradiance [W m-2]
        EbnMRM5 : Direct normal irradiance [W m-2]
        EdhMRM5 : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = (np.cos(sza) + 0.50572 * (((np.pi / 2 - sza) / np.pi * 180) + 6.07995) ** -1.6364) ** -1
    ama = am * press / 1013.25

    # atmospheric gases transmittance function
    Tco2 = 1 - (0.07210 * ama * 350 / ((1 + 377.890 * ama * 350) ** 0.5855 + 3.1709 * ama * 350))
    Tco = 1 - (0.0062 * ama * 0.075 / ((1 + 243.670 * ama * 0.075) ** 0.4246 + 1.7222 * ama * 0.075))
    Tn2o = 1 - (0.0326 * ama * 0.28 / ((1 + 107.413 * ama * 0.28) ** 0.5501 + 0.9093 * ama * 0.28))
    Tch4 = 1 - (0.0192 * ama * 1.6 / ((1 + 166.095 * ama * 1.6) ** 0.4221 + 0.7186 * ama * 1.6))
    To2 = 1 - (0.0003 * ama * 2.095e5 / ((1 + 476.934 * ama * 2.095e5) ** 0.4892 + 0.1261 * ama * 2.095e5))

    # the transmittance function for the mixed gases
    Tmg = Tco2 * Tco * Tn2o * Tch4 * To2

    # the transmittance function for water vapour
    lw = wv
    Tw = 1 - (3.0140 * am * lw / ((1 + 119.300 * am * lw) ** 0.6440 + 5.8140 * am * lw))

    # the transmittance function for ozone
    To = 1 - (0.2554 * am * ozone / ((1 + 6107.260 * am * ozone) ** 0.2040 + 0.4710 * am * ozone))

    # the transmittance function for Rayleigh scattering
    Tr = np.exp(-0.1128 * ama ** 0.8346 * (0.9341 - ama ** 0.9868 + 0.9391 * ama))

    # the Mie scattering transmittance function
    Ta = np.exp(-am * ang_beta * (0.6777 + 0.1464 * (am * ang_beta) - 0.00626 * (am * ang_beta) ** 2) ** -1.3)

    # the aerosol absorption function
    Taa = 1 - 0.1 * (1 - am + am ** 1.06) * (1 - Ta)

    # the aerosol scattering function
    Tas = Ta / Taa

    # direct beam irradiance
    EbnMRM5 = Eext * Ta * Tr * To * Tw * Tmg

    # the circumsolar diffuse radiation produced by a single-scattering mode of molecules and aerosols
    Ids = Eext * np.cos(sza) * Taa * To * Tw * Tmg * 0.5 * (1 - Tas * Tr)

    # the diffuse component reflected by the ground and backscattered by the atmosphere
    Ta166 = np.exp(-1.66 * ang_beta * (0.6777 + 0.1464 * (1.66 * ang_beta) - 0.00626 * (1.66 * ang_beta) ** 2) ** -1.3)
    pa = 0.0685 + 0.16 * (1 - Ta166)
    Idm = (EbnMRM5 * np.cos(sza) + Ids) * (albedo * pa / (1 - albedo * pa))

    # diffuse horizontal irradiance
    EdhMRM5 = Ids + Idm

    # global horizontal irradiance
    EghMRM5 = EbnMRM5 * np.cos(sza) + EdhMRM5

    # Quality control
    lower = 0
    EbnMRM5[EbnMRM5 < lower] = 0
    EdhMRM5[EdhMRM5 < lower] = 0
    EghMRM5[EghMRM5 < lower] = 0

    return [EghMRM5, EbnMRM5, EdhMRM5]